import os

import polars as pl
import pytest

from postviritu.aligner import HIT_SCHEMA
from postviritu.taxonomy import map_ranks_to_esviritu, unclassified_lineage

EXAMPLE_PREFIX = "AYWM5R.p2126"


def _make_example_data(root: str, prefix: str) -> str:
    """Create a minimal synthetic EsViritu sample directory for integration tests."""
    from postviritu.io_esviritu import write_fasta

    os.makedirs(root, exist_ok=True)

    info = pl.DataFrame(
        [
            {
                "sample_ID": "AYWM5R.p2126",
                "Name": "OR777233.1",
                "description": ".",
                "Length": 30000,
                "Segment": "1",
                "Accession": "OR777233.1",
                "Assembly": "asm1",
                "Asm_length": 30000,
                "kingdom": "k__Viruses",
                "phylum": "p__",
                "tclass": "c__",
                "order": "o__",
                "family": "f__",
                "genus": "g__",
                "species": "s__OrigSp1",
                "subspecies": "t__",
                "RPKMF": 100.0,
                "read_count": 100,
                "covered_bases": 1000,
                "mean_coverage": 2.0,
                "avg_read_identity": 0.97,
                "Pi": 0.0,
                "filtered_reads_in_sample": 1_000_000,
            },
            {
                "sample_ID": "AYWM5R.p2126",
                "Name": "Y15173.1",
                "description": ".",
                "Length": 30000,
                "Segment": "2",
                "Accession": "Y15173.1",
                "Assembly": "asm1",
                "Asm_length": 30000,
                "kingdom": "k__Viruses",
                "phylum": "p__",
                "tclass": "c__",
                "order": "o__",
                "family": "f__",
                "genus": "g__",
                "species": "s__OrigSp2",
                "subspecies": "t__",
                "RPKMF": 100.0,
                "read_count": 100,
                "covered_bases": 1000,
                "mean_coverage": 2.0,
                "avg_read_identity": 0.97,
                "Pi": 0.0,
                "filtered_reads_in_sample": 1_000_000,
            },
        ],
        schema_overrides={"Segment": pl.Utf8},
    )
    info.write_csv(
        os.path.join(root, f"{prefix}.detected_virus.info.tsv"), separator="\t"
    )

    # Coverage windows file (passed through unchanged).
    open(os.path.join(root, f"{prefix}.virus_coverage_windows.tsv"), "w").close()

    # Params YAML used by get_thresholds.
    import yaml

    with open(os.path.join(root, f"{prefix}_esviritu.params.yaml"), "w") as fh:
        yaml.safe_dump({"spthresh": 0.9, "subspthresh": 0.95}, fh)

    # Consensus FASTA with two accessions.
    consensus = os.path.join(root, f"{prefix}_final_consensus.fasta")
    write_fasta(
        {"OR777233.1": "ACGT" * 100, "Y15173.1": "TGCA" * 100},
        consensus,
    )
    return root


@pytest.fixture(scope="session")
def example_data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("example_data")
    _make_example_data(str(root), EXAMPLE_PREFIX)
    return str(root)


@pytest.fixture
def example_prefix():
    return EXAMPLE_PREFIX


class FakeTaxonomy:
    """In-memory stand-in for the taxonkit-backed Taxonomy class.

    ``rank_table`` maps taxid -> {ncbi_rank: name}. ``lca_table`` maps a
    frozenset of taxids -> taxid for deterministic LCA results.
    """

    def __init__(self, rank_table=None, lca_table=None, genotype_table=None):
        self.rank_table = rank_table or {}
        self.lca_table = lca_table or {}
        self.genotype_table = genotype_table or {}

    def esviritu_lineage(self, taxid, accession=None):
        rmap = self.rank_table.get(taxid)
        if not rmap:
            return unclassified_lineage()
        lineage = map_ranks_to_esviritu(rmap)
        if accession in self.genotype_table:
            lineage["subspecies"] = "t__" + self.genotype_table[accession]
        return lineage

    def lca(self, taxids):
        clean = [t for t in dict.fromkeys(taxids) if t and t != "0"]
        if not clean:
            return None
        if len(clean) == 1:
            return clean[0]
        return self.lca_table.get(frozenset(clean), clean[0])


class FakeAligner:
    """Returns a preset hit table regardless of the query."""

    def __init__(self, hits, second_round=None):
        self._hits = hits
        self._second_round = second_round if second_round is not None else _empty()

    def search(
        self,
        query_fasta,
        threads=1,
        exclude_taxids=None,
        tmp_dir=None,
        keep=False,
        result_name="result.m8",
        query_nonN_len=None,
    ):
        if exclude_taxids:
            return self._second_round
        return self._hits


def _empty():
    return pl.DataFrame(schema=HIT_SCHEMA)


def make_hits(rows):
    """Build a hit DataFrame from a list of dicts (canonical schema)."""
    return pl.DataFrame(rows, schema=HIT_SCHEMA)


@pytest.fixture
def fake_taxonomy_cls():
    return FakeTaxonomy


@pytest.fixture
def fake_aligner_cls():
    return FakeAligner
