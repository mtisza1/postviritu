"""Sequence-search backends.

Defines an :class:`Aligner` interface that aligns query sequences (the EsViritu
consensus genomes) against a prepared reference database and returns a tidy
polars DataFrame of hits. The mmseqs2 backend is implemented here; the abstract
interface allows a BLASTN backend to be added later without refactoring the
downstream taxonomy/reassignment logic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

# Columns every aligner must return. ``query`` is the EsViritu Accession,
# ``target`` is the DB sequence id, ``taxid`` is the NCBI taxid of the target.
HIT_COLUMNS = [
    "query",
    "target",
    "taxid",
    "pct_identity",  # fraction 0-1
    "aln_length",
    "query_length",
    "query_coverage",  # fraction 0-1
    "evalue",
    "bitscore",
]

HIT_SCHEMA = {
    "query": pl.Utf8,
    "target": pl.Utf8,
    "taxid": pl.Utf8,
    "pct_identity": pl.Float64,
    "aln_length": pl.Int64,
    "query_length": pl.Int64,
    "query_coverage": pl.Float64,
    "evalue": pl.Float64,
    "bitscore": pl.Float64,
}


def empty_hits() -> pl.DataFrame:
    """Return an empty hit table with the canonical schema."""
    return pl.DataFrame(schema=HIT_SCHEMA)


class Aligner(ABC):
    """Abstract sequence-search backend."""

    @abstractmethod
    def search(
        self,
        query_fasta: str,
        threads: int = 1,
        exclude_taxids: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Align ``query_fasta`` against the backend's DB.

        ``exclude_taxids`` optionally removes the given taxids (and descendants)
        from the searchable database, used for the second-round tie check.
        Returns a polars DataFrame with :data:`HIT_COLUMNS`.
        """
        raise NotImplementedError


class Mmseqs2Aligner(Aligner):
    """mmseqs2 ``easy-search`` backend against a taxonomy-aware target DB."""

    # mmseqs format-output field order (must match parsing below).
    _FORMAT_FIELDS = [
        "query",
        "target",
        "taxid",
        "fident",
        "alnlen",
        "qlen",
        "qcov",
        "evalue",
        "bits",
    ]

    def __init__(
        self,
        target_db: str,
        sensitivity: float = 5.7,
        max_seqs: int = 300,
        mmseqs_bin: str = "mmseqs",
        extra_args: Optional[List[str]] = None,
    ):
        self.target_db = target_db
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs
        self.mmseqs_bin = mmseqs_bin
        self.extra_args = extra_args or []

    def search(
        self,
        query_fasta: str,
        threads: int = 1,
        exclude_taxids: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        if shutil.which(self.mmseqs_bin) is None:
            raise RuntimeError(
                f"'{self.mmseqs_bin}' not found on PATH. Install mmseqs2 "
                "(e.g. `conda install -c bioconda mmseqs2`)."
            )
        tmp_root = tempfile.mkdtemp(prefix="postviritu_mmseqs_")
        result_m8 = os.path.join(tmp_root, "result.m8")
        mmseqs_tmp = os.path.join(tmp_root, "tmp")
        os.makedirs(mmseqs_tmp, exist_ok=True)
        cmd = [
            self.mmseqs_bin,
            "easy-search",
            query_fasta,
            self.target_db,
            result_m8,
            mmseqs_tmp,
            "--threads",
            str(threads),
            "-s",
            str(self.sensitivity),
            "--max-seqs",
            str(self.max_seqs),
            "--search-type",
            "3",
            "--format-output",
            ",".join(self._FORMAT_FIELDS),
            *self.extra_args,
        ]
        if exclude_taxids:
            # '!taxid' negates, i.e. excludes that taxon (and descendants).
            taxon_list = ",".join("!" + str(t) for t in exclude_taxids)
            cmd += ["--taxon-list", taxon_list]
        try:
            subprocess.run(cmd, check=True)
            return parse_m8(result_m8, self._FORMAT_FIELDS)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


def parse_m8(path: str, fields: List[str]) -> pl.DataFrame:
    """Parse an mmseqs/BLAST tabular output into the canonical hit schema.

    ``fields`` lists the column names as emitted, using mmseqs field names
    (``fident``, ``qcov``, ``bits`` ...). The query is normalised so a
    ``_consensus`` suffix is stripped to recover the EsViritu Accession.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return empty_hits()

    raw = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=fields,
        infer_schema_length=10000,
        schema_overrides={"taxid": pl.Utf8},
    )

    rename = {
        "fident": "pct_identity",
        "alnlen": "aln_length",
        "qlen": "query_length",
        "qcov": "query_coverage",
        "bits": "bitscore",
    }
    raw = raw.rename({k: v for k, v in rename.items() if k in raw.columns})

    # Strip the EsViritu consensus suffix from the query name.
    raw = raw.with_columns(
        pl.col("query").str.replace(r"_consensus$", "").alias("query")
    )

    # Ensure all canonical columns exist (fill missing with nulls).
    for col, dtype in HIT_SCHEMA.items():
        if col not in raw.columns:
            raw = raw.with_columns(pl.lit(None).cast(dtype).alias(col))

    return raw.select(HIT_COLUMNS).cast(HIT_SCHEMA, strict=False)


def filter_hits(
    hits: pl.DataFrame,
    min_identity: float = 0.9,
    min_aln_fraction: float = 0.5,
    max_evalue: float = 1e-10,
) -> pl.DataFrame:
    """Keep only acceptable hits and drop rows with a missing/0 taxid."""
    if hits.is_empty():
        return hits
    return hits.filter(
        (pl.col("pct_identity") >= min_identity)
        & (pl.col("query_coverage") >= min_aln_fraction)
        & (pl.col("evalue") <= max_evalue)
        & pl.col("taxid").is_not_null()
        & (pl.col("taxid") != "0")
        & (pl.col("taxid") != "")
    )
