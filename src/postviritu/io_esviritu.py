"""I/O for EsViritu per-sample output files.

Handles discovery of sample prefixes in an EsViritu output directory, loading of
the relevant TSV/YAML files into polars DataFrames, parsing of consensus FASTA
headers, and writing of the rewritten output tables.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import polars as pl
import yaml

# Canonical EsViritu file-name suffixes (relative to a sample PREFIX).
INFO_SUFFIX = ".detected_virus.info.tsv"
ASSEMBLY_SUFFIX = ".detected_virus.assembly_summary.tsv"
TAX_PROFILE_SUFFIX = ".tax_profile.tsv"
COVERAGE_SUFFIX = ".virus_coverage_windows.tsv"
CONSENSUS_SUFFIX = "_final_consensus.fasta"
PARAMS_SUFFIX = "_esviritu.params.yaml"
READSTATS_SUFFIX = "_esviritu.readstats.yaml"
REPORT_SUFFIX = ".postviritu_report.html"

CONSENSUS_HEADER_SUFFIX = "_consensus"

# Canonical nucleotide bases used for alignment-fraction calculations.
CANONICAL_BASES = frozenset("ATCG")


def canonical_base_count(seq: str) -> int:
    """Return the number of canonical (A, T, C, G) bases in ``seq``.

    Matching is case-insensitive so both lower- and upper-case FASTA sequences
    are counted correctly.
    """
    return sum(1 for base in seq if base.upper() in CANONICAL_BASES)


# The 8 EsViritu taxonomy ranks, in order.
TAX_RANKS = [
    "kingdom",
    "phylum",
    "tclass",
    "order",
    "family",
    "genus",
    "species",
    "subspecies",
]

# Rank -> EsViritu prefix used in the lineage strings.
RANK_PREFIXES = {
    "kingdom": "k__",
    "phylum": "p__",
    "tclass": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
    "subspecies": "t__",
}


@dataclass
class SamplePaths:
    """Resolved file paths for a single EsViritu sample prefix."""

    prefix: str
    directory: str

    @property
    def info(self) -> str:
        return os.path.join(self.directory, self.prefix + INFO_SUFFIX)

    @property
    def assembly_summary(self) -> str:
        return os.path.join(self.directory, self.prefix + ASSEMBLY_SUFFIX)

    @property
    def tax_profile(self) -> str:
        return os.path.join(self.directory, self.prefix + TAX_PROFILE_SUFFIX)

    @property
    def coverage_windows(self) -> str:
        return os.path.join(self.directory, self.prefix + COVERAGE_SUFFIX)

    @property
    def consensus(self) -> str:
        return os.path.join(self.directory, self.prefix + CONSENSUS_SUFFIX)

    @property
    def params(self) -> str:
        return os.path.join(self.directory, self.prefix + PARAMS_SUFFIX)

    @property
    def readstats(self) -> str:
        return os.path.join(self.directory, self.prefix + READSTATS_SUFFIX)

    @property
    def report(self) -> str:
        return os.path.join(self.directory, self.prefix + REPORT_SUFFIX)

    def missing_required(self) -> List[str]:
        """Return the list of required input files that do not exist."""
        required = [self.info, self.consensus]
        return [p for p in required if not os.path.isfile(p)]


def discover_sample_prefixes(directory: str) -> List[str]:
    """Find sample prefixes in a directory by scanning for info.tsv files."""
    pattern = os.path.join(directory, "*" + INFO_SUFFIX)
    prefixes = []
    for fp in sorted(glob.glob(pattern)):
        base = os.path.basename(fp)
        prefixes.append(base[: -len(INFO_SUFFIX)])
    return prefixes


def iter_samples(
    directory: str, sample_id: Optional[str] = None
) -> Iterator[SamplePaths]:
    """Yield SamplePaths for all samples (or a single sample_id) in a directory."""
    if sample_id:
        yield SamplePaths(prefix=sample_id, directory=directory)
        return
    for prefix in discover_sample_prefixes(directory):
        yield SamplePaths(prefix=prefix, directory=directory)


def read_tsv(path: str) -> pl.DataFrame:
    """Read a TSV with EsViritu-friendly schema inference."""
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
        schema_overrides={"Segment": pl.Utf8},
    )


def write_tsv(df: pl.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.write_csv(path, separator="\t")


def load_params(path: str) -> Dict:
    """Load params.yaml; returns {} if missing."""
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def get_thresholds(params: Dict) -> Tuple[float, float]:
    """Return (spthresh, subspthresh) with EsViritu defaults if absent."""
    spthresh = float(params.get("spthresh", 0.90))
    subspthresh = float(params.get("subspthresh", 0.95))
    return spthresh, subspthresh


def consensus_accession(header: str, sample: Optional[str] = None) -> str:
    """Recover the info-table Accession from a consensus FASTA record name.

    EsViritu >= 1.3 names records ``{Accession}_{sample}_consensus``; older
    versions use ``{Accession}_consensus``. Both suffixes are stripped.
    """
    if sample:
        suffix = f"_{sample}{CONSENSUS_HEADER_SUFFIX}"
        if header.endswith(suffix):
            return header[: -len(suffix)]
    return header.removesuffix(CONSENSUS_HEADER_SUFFIX)


def parse_consensus_fasta(path: str, sample: Optional[str] = None) -> Dict[str, str]:
    """Parse a consensus FASTA into {Accession: sequence}.

    Record names are mapped back to info-table Accessions with
    :func:`consensus_accession`; pass ``sample`` (the EsViritu sample name)
    to handle the EsViritu >= 1.3 header format.
    """
    sequences: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                current = consensus_accession(line[1:].split()[0], sample)
                sequences[current] = []
            elif current is not None:
                sequences[current].append(line)
    return {acc: "".join(parts) for acc, parts in sequences.items()}


def write_fasta(sequences: Dict[str, str], path: str) -> None:
    """Write {name: sequence} to a FASTA file (60 chars per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
