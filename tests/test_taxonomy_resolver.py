"""Tests for the pytaxonkit-backed Taxonomy resolver, using a fake module.

These avoid needing the taxonkit binary / pytaxonkit installed by monkeypatching
``Taxonomy._pytaxonkit`` to return an in-memory stand-in.
"""

from collections import namedtuple

import pytest

from postviritu.taxonomy import Taxonomy

_Row = namedtuple("Row", ["TaxID", "FullLineage", "FullLineageRanks"])


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def itertuples(self, index=False):
        return iter(self._rows)


class FakePyTaxonkit:
    """Mimics the subset of pytaxonkit used by Taxonomy."""

    def __init__(self, lineage_rows=None, lca_result=0):
        self._lineage_rows = lineage_rows or {}
        self._lca_result = lca_result
        self.last_lineage_call = None
        self.last_lca_call = None

    def lineage(self, ids, data_dir=None, threads=None):
        self.last_lineage_call = (list(ids), data_dir, threads)
        rows = []
        for taxid in ids:
            names, ranks = self._lineage_rows.get(str(taxid), (float("nan"), float("nan")))
            rows.append(_Row(int(taxid), names, ranks))
        return _FakeDF(rows)

    def lca(self, ids, skip_deleted=False, skip_unfound=False, data_dir=None, threads=None):
        self.last_lca_call = (list(ids), skip_deleted, skip_unfound, data_dir, threads)
        return self._lca_result


def _patch(monkeypatch, fake):
    monkeypatch.setattr(Taxonomy, "_pytaxonkit", staticmethod(lambda: fake))


def test_rank_maps_parses_full_lineage(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Preplasmiviricota;Tectiliviricetes;Rowavirales;"
                "Adenoviridae;Mastadenovirus;Human mastadenovirus F",
                "acellular root;phylum;class;order;family;genus;species",
            )
        }
    )
    _patch(monkeypatch, fake)
    tax = Taxonomy(data_dir="/tmp/taxdump", threads=2)
    rmap = tax.rank_maps(["999"])["999"]
    assert rmap["acellular root"] == "Viruses"
    assert rmap["family"] == "Adenoviridae"
    assert rmap["species"] == "Human mastadenovirus F"
    # data_dir/threads forwarded to pytaxonkit.lineage.
    assert fake.last_lineage_call == (["999"], "/tmp/taxdump", 2)


def test_esviritu_lineage_maps_acellular_root_to_kingdom(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Adenoviridae;Mastadenovirus;Human mastadenovirus F",
                "acellular root;family;genus;species",
            )
        }
    )
    _patch(monkeypatch, fake)
    tax = Taxonomy()
    lineage = tax.esviritu_lineage("999")
    assert lineage["kingdom"] == "k__Viruses"
    assert lineage["phylum"] == "p__unclassified_Viruses"
    assert lineage["family"] == "f__Adenoviridae"
    assert lineage["species"] == "s__Human mastadenovirus F"
    assert lineage["subspecies"] == "t__Human mastadenovirus F"


def test_esviritu_lineage_unknown_taxid_is_unclassified(monkeypatch):
    fake = FakePyTaxonkit(lineage_rows={})  # NaN lineage -> empty map
    _patch(monkeypatch, fake)
    tax = Taxonomy()
    lineage = tax.esviritu_lineage("123456789")
    assert lineage["kingdom"] == "k__unclassified_Viruses"
    assert lineage["species"] == "s__unclassified_Viruses"


def test_lineage_cache_avoids_second_call(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={"999": ("Viruses", "acellular root")}
    )
    _patch(monkeypatch, fake)
    tax = Taxonomy()
    tax.rank_maps(["999"])
    fake.last_lineage_call = None
    tax.rank_maps(["999"])  # cached -> no new pytaxonkit call
    assert fake.last_lineage_call is None


def test_lca_single_taxid_short_circuits(monkeypatch):
    fake = FakePyTaxonkit(lca_result=0)
    _patch(monkeypatch, fake)
    tax = Taxonomy()
    assert tax.lca(["100"]) == "100"
    assert fake.last_lca_call is None  # not invoked for a single taxid


def test_lca_multiple_returns_string(monkeypatch):
    fake = FakePyTaxonkit(lca_result=10239)
    _patch(monkeypatch, fake)
    tax = Taxonomy(data_dir="/tmp/td")
    assert tax.lca(["200", "300"]) == "10239"
    ids, skip_del, skip_unf, data_dir, threads = fake.last_lca_call
    assert ids == [200, 300]
    assert skip_del and skip_unf
    assert data_dir == "/tmp/td"


def test_lca_invalid_returns_none(monkeypatch):
    fake = FakePyTaxonkit(lca_result=0)
    _patch(monkeypatch, fake)
    tax = Taxonomy()
    assert tax.lca(["1", "2"]) is None
