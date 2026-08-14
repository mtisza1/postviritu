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
from typing import Dict, Iterable, List, Optional, Tuple

import polars as pl

from .io_esviritu import parse_consensus_fasta, write_fasta

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
    "qaln",
    "taln",
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
    "qaln": pl.Utf8,
    "taln": pl.Utf8,
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
        tmp_dir: Optional[str] = None,
        keep: bool = False,
        result_name: str = "result.m8",
        query_nonN_len: Optional[Dict[str, int]] = None,
    ) -> pl.DataFrame:
        """Align ``query_fasta`` against the backend's DB.

        ``exclude_taxids`` optionally removes the given taxids (and descendants)
        from the searchable database, used for the second-round tie check.
        ``tmp_dir`` is the working directory for intermediate files (created if
        needed); when ``None`` a system temp directory is used. ``keep``
        preserves the tabular alignment output (named ``result_name``) under
        ``tmp_dir`` for later inspection instead of deleting it.
        ``query_nonN_len`` maps each query Accession to its non-N residue count;
        when provided, ``pct_identity`` and ``query_coverage`` are recomputed to
        exclude N positions (see :func:`parse_m8`).
        Returns a polars DataFrame with :data:`HIT_COLUMNS`.
        """
        raise NotImplementedError


class Mmseqs2Aligner(Aligner):
    """mmseqs2 ``easy-search`` backend against a taxonomy-aware target DB."""

    # mmseqs format-output field order (must match parsing below).
    # ``qaln``/``taln`` (gapped aligned sequences) are emitted so identity and
    # coverage can be recomputed excluding N positions on both query and target.
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
        "qaln",
        "taln",
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
        tmp_dir: Optional[str] = None,
        keep: bool = False,
        result_name: str = "result.m8",
        query_nonN_len: Optional[Dict[str, int]] = None,
    ) -> pl.DataFrame:
        if shutil.which(self.mmseqs_bin) is None:
            raise RuntimeError(
                f"'{self.mmseqs_bin}' not found on PATH. Install mmseqs2 "
                "(e.g. `conda install -c bioconda mmseqs2`)."
            )
        if tmp_dir is None:
            tmp_root = tempfile.mkdtemp(prefix="postviritu_mmseqs_")
        else:
            tmp_root = tmp_dir
            os.makedirs(tmp_root, exist_ok=True)
        result_m8 = os.path.join(tmp_root, result_name)
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
            return parse_m8(result_m8, self._FORMAT_FIELDS, query_nonN_len)
        finally:
            if keep:
                # Preserve the tabular alignment output; drop only the
                # mmseqs2 internal working directory.
                shutil.rmtree(mmseqs_tmp, ignore_errors=True)
            else:
                shutil.rmtree(tmp_root, ignore_errors=True)


def parse_m8(
    path: str,
    fields: List[str],
    query_nonN_len: Optional[Dict[str, int]] = None,
) -> pl.DataFrame:
    """Parse a mmseqs2 tabular output into the canonical hit schema.

    ``fields`` lists the column names as emitted, using mmseqs field names
    (``fident``, ``qcov``, ``bits`` ...). The query is normalised so a
    ``_consensus`` suffix is stripped to recover the EsViritu Accession.

    When the gapped aligned sequences (``qaln``/``taln``) are present,
    ``pct_identity`` and ``query_coverage`` are recomputed to exclude N
    positions: identity is matching residues over alignment columns where
    neither query nor target is N (or a gap), and coverage is the count of
    aligned non-N query residues over ``query_nonN_len`` (the query's total
    non-N residue count). Columns left untouched fall back to the values mmseqs
    reported.
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

    # Recompute identity/coverage excluding N positions when the aligned
    # sequences are available (query name must already be the Accession so it
    # matches the keys of ``query_nonN_len``).
    if "qaln" in raw.columns and "taln" in raw.columns:
        raw = _recompute_nonN_metrics(raw, query_nonN_len)

    # Ensure all canonical columns exist (fill missing with nulls).
    for col, dtype in HIT_SCHEMA.items():
        if col not in raw.columns:
            raw = raw.with_columns(pl.lit(None).cast(dtype).alias(col))

    return raw.select(HIT_COLUMNS).cast(HIT_SCHEMA, strict=False)


def _recompute_nonN_metrics(
    raw: pl.DataFrame, query_nonN_len: Optional[Dict[str, int]]
) -> pl.DataFrame:
    """Recompute ``pct_identity`` and ``query_coverage`` excluding N positions.

    Identity is computed over alignment columns where neither the query nor
    target residue is a gap or N. Coverage is the number of aligned non-N query
    residues divided by the query's total non-N residue count
    (``query_nonN_len``); when that count is unavailable for a query, the
    mmseqs-reported coverage is kept.
    """
    query_nonN_len = query_nonN_len or {}
    qalns = raw["qaln"].to_list()
    talns = raw["taln"].to_list()
    queries = raw["query"].to_list()
    orig_cov = (
        raw["query_coverage"].to_list()
        if "query_coverage" in raw.columns
        else [None] * len(queries)
    )

    new_pid: List[float] = []
    new_cov: List[Optional[float]] = []
    for qaln, taln, query, ocov in zip(qalns, talns, queries, orig_cov):
        matches = 0
        id_denom = 0
        aligned_nonN_query = 0
        if qaln and taln:
            for cq, ct in zip(qaln, taln):
                if cq == "-":
                    continue
                cq_u = cq.upper()
                q_is_n = cq_u == "N"
                if not q_is_n:
                    aligned_nonN_query += 1
                if ct == "-" or q_is_n or ct.upper() == "N":
                    continue
                id_denom += 1
                if cq_u == ct.upper():
                    matches += 1
        new_pid.append(matches / id_denom if id_denom else 0.0)
        total = query_nonN_len.get(query)
        if total:
            new_cov.append(aligned_nonN_query / total)
        else:
            new_cov.append(ocov)

    return raw.with_columns(
        pl.Series("pct_identity", new_pid, dtype=pl.Float64),
        pl.Series("query_coverage", new_cov, dtype=pl.Float64),
    )


def _chunked(items: List[Tuple[str, str]], n: int) -> Iterable[List[Tuple[str, str]]]:
    """Yield successive chunks of size ``n`` from ``items``."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def parse_blastn(
    path: str,
    fields: List[str],
    query_nonN_len: Optional[Dict[str, int]] = None,
) -> pl.DataFrame:
    """Parse a BLASTN tabular output (``-outfmt 6``) into the canonical hit schema.

    ``fields`` lists the column names as emitted by BLASTN. Percentages
    (``pident``, ``qcovs``) are converted to fractions, ``staxids`` is reduced
    to the first taxid, and the EsViritu ``_consensus`` suffix is stripped from
    the query name. When gapped aligned sequences (``qseq``/``sseq``) are
    present, identity and coverage are recomputed to exclude N positions, just
    as for the mmseqs2 backend.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return empty_hits()

    raw = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=fields,
        infer_schema_length=10000,
        schema_overrides={"staxids": pl.Utf8},
    )

    rename = {
        "qseqid": "query",
        "sseqid": "target",
        "staxids": "taxid_raw",
        "length": "aln_length",
        "qlen": "query_length",
        "qseq": "qaln",
        "sseq": "taln",
    }
    raw = raw.rename({k: v for k, v in rename.items() if k in raw.columns})

    # BLASTN reports percentages; downstream expects fractions.
    if "pident" in raw.columns:
        raw = raw.with_columns((pl.col("pident") / 100.0).alias("pct_identity"))
    if "qcovs" in raw.columns:
        raw = raw.with_columns((pl.col("qcovs") / 100.0).alias("query_coverage"))

    # ``staxids`` may contain multiple semicolon-separated taxids.
    if "taxid_raw" in raw.columns:
        raw = raw.with_columns(
            pl.col("taxid_raw").str.extract(r"^(\d+)").alias("taxid")
        )

    # Strip the EsViritu consensus suffix from the query name.
    raw = raw.with_columns(
        pl.col("query").str.replace(r"_consensus$", "").alias("query")
    )

    # Recompute identity/coverage excluding N positions when the aligned
    # sequences are available.
    if "qaln" in raw.columns and "taln" in raw.columns:
        raw = _recompute_nonN_metrics(raw, query_nonN_len)

    # Ensure all canonical columns exist (fill missing with nulls).
    for col, dtype in HIT_SCHEMA.items():
        if col not in raw.columns:
            raw = raw.with_columns(pl.lit(None).cast(dtype).alias(col))

    return raw.select(HIT_COLUMNS).cast(HIT_SCHEMA, strict=False)


class BlastnAligner(Aligner):
    """NCBI BLASTN ``-remote`` backend against the ``nt`` database."""

    # BLASTN -outfmt 6 field order (must match parsing above).
    _FORMAT_FIELDS = [
        "qseqid",
        "sseqid",
        "staxids",
        "pident",
        "length",
        "qlen",
        "qcovs",
        "evalue",
        "bitscore",
        "qseq",
        "sseq",
    ]

    def __init__(
        self,
        db: str = "nt",
        blastn_bin: str = "blastn",
        max_target_seqs: int = 300,
        batch_size: int = 3,
        extra_args: Optional[List[str]] = None,
    ):
        self.db = db
        self.blastn_bin = blastn_bin
        self.max_target_seqs = max_target_seqs
        self.batch_size = batch_size
        self.extra_args = extra_args or []

    def search(
        self,
        query_fasta: str,
        threads: int = 1,
        exclude_taxids: Optional[List[str]] = None,
        tmp_dir: Optional[str] = None,
        keep: bool = False,
        result_name: str = "result.m8",
        query_nonN_len: Optional[Dict[str, int]] = None,
    ) -> pl.DataFrame:
        if shutil.which(self.blastn_bin) is None:
            raise RuntimeError(
                f"'{self.blastn_bin}' not found on PATH. Install BLAST+ "
                "(e.g. `conda install -c bioconda blast`)."
            )

        if tmp_dir is None:
            tmp_root = tempfile.mkdtemp(prefix="postviritu_blastn_")
        else:
            tmp_root = tmp_dir
            os.makedirs(tmp_root, exist_ok=True)

        result_path = os.path.join(tmp_root, result_name)
        batch_out_paths: List[str] = []

        try:
            records = list(parse_consensus_fasta(query_fasta).items())
            if not records:
                return empty_hits()

            for i, batch in enumerate(_chunked(records, self.batch_size)):
                batch_fasta = os.path.join(tmp_root, f"batch_{i}.fasta")
                batch_out = os.path.join(tmp_root, f"batch_{i}.tsv")
                write_fasta(dict(batch), batch_fasta)

                cmd = [
                    self.blastn_bin,
                    "-query",
                    batch_fasta,
                    "-db",
                    self.db,
                    "-remote",
                    "-outfmt",
                    f"6 {' '.join(self._FORMAT_FIELDS)}",
                    "-out",
                    batch_out,
                    "-max_target_seqs",
                    str(self.max_target_seqs),
                    *self.extra_args,
                ]
                subprocess.run(cmd, check=True)
                batch_out_paths.append(batch_out)

            # Concatenate per-batch tabular outputs into a single result file.
            with open(result_path, "w") as out_fh:
                for batch_out in batch_out_paths:
                    if os.path.isfile(batch_out) and os.path.getsize(batch_out) > 0:
                        with open(batch_out) as in_fh:
                            shutil.copyfileobj(in_fh, out_fh)

            hits = parse_blastn(result_path, self._FORMAT_FIELDS, query_nonN_len)

            if exclude_taxids:
                # BLASTN -remote does not support -negative_taxids, so filter the
                # tabular results locally (exact taxid match).
                exclude = [str(t) for t in exclude_taxids]
                hits = hits.filter(~pl.col("taxid").is_in(exclude))

            return hits
        finally:
            if keep:
                # Keep the final concatenated result; remove per-batch files.
                for batch_out in batch_out_paths:
                    batch_fasta = batch_out.replace(".tsv", ".fasta")
                    for p in [batch_out, batch_fasta]:
                        if os.path.isfile(p):
                            os.remove(p)
            else:
                shutil.rmtree(tmp_root, ignore_errors=True)


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
