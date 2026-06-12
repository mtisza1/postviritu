import os

import polars as pl
import pytest

from postviritu.aligner import HIT_SCHEMA
from postviritu.taxonomy import map_ranks_to_esviritu, unclassified_lineage

EXAMPLE_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_data"
)
EXAMPLE_PREFIX = "AYWM5R.p2126"


@pytest.fixture
def example_data_dir():
    return EXAMPLE_DATA


@pytest.fixture
def example_prefix():
    return EXAMPLE_PREFIX


class FakeTaxonomy:
    """In-memory stand-in for the taxonkit-backed Taxonomy class.

    ``rank_table`` maps taxid -> {ncbi_rank: name}. ``lca_table`` maps a
    frozenset of taxids -> taxid for deterministic LCA results.
    """

    def __init__(self, rank_table=None, lca_table=None):
        self.rank_table = rank_table or {}
        self.lca_table = lca_table or {}

    def esviritu_lineage(self, taxid):
        rmap = self.rank_table.get(taxid)
        if not rmap:
            return unclassified_lineage()
        return map_ranks_to_esviritu(rmap)

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
