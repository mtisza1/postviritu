"""Tests for the pytaxonkit-backed Taxonomy resolver, using a fake module.

These avoid needing the taxonkit binary / pytaxonkit installed by monkeypatching
``Taxonomy._pytaxonkit`` to return an in-memory stand-in.
"""

from collections import namedtuple

import pytest
import requests

from postviritu.taxonomy import Taxonomy

_Row = namedtuple(
    "Row", ["TaxID", "FullLineage", "FullLineageRanks", "Name", "Rank"]
)


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
            values = self._lineage_rows.get(
                str(taxid), (float("nan"), float("nan"), None, None)
            )
            if len(values) == 2:
                values = (*values, None, None)
            rows.append(_Row(int(taxid), *values))
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


def test_esviritu_lineage_uses_terminal_no_rank_below_species(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Mastadenovirus;Human mastadenovirus F;Human adenovirus 41 isolate Tak",
                "acellular root;genus;species;no rank",
                "Human adenovirus 41 isolate Tak",
                "no rank",
            )
        }
    )
    _patch(monkeypatch, fake)

    lineage = Taxonomy().esviritu_lineage("999")

    assert lineage["species"] == "s__Human mastadenovirus F"
    assert lineage["subspecies"] == "t__Human adenovirus 41 isolate Tak"


def test_esviritu_lineage_includes_terminal_strain_missing_from_full_lineage(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Mastadenovirus;Human mastadenovirus F",
                "acellular root;genus;species",
                "Human adenovirus 41 strain Dugan",
                "strain",
            )
        }
    )
    _patch(monkeypatch, fake)

    lineage = Taxonomy().esviritu_lineage("999")

    assert lineage["species"] == "s__Human mastadenovirus F"
    assert lineage["subspecies"] == "t__Human adenovirus 41 strain Dugan"


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


class _VVResponse:
    def __init__(self, docs):
        self.docs = docs

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": {"docs": self.docs}}


def test_virus_lineage_uses_vvsearch_genotype_and_caches_accession(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Orthopoxvirus;Monkeypox virus",
                "acellular root;genus;species",
            )
        }
    )
    _patch(monkeypatch, fake)
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return _VVResponse([{"AccVer_s": "XCI56374.1", "Genotype_s": "IIb"}])

    monkeypatch.setattr("postviritu.taxonomy.requests.get", get)
    tax = Taxonomy()

    lineage = tax.esviritu_lineage("999", "gi|123|ref|XCI56374.1|")
    tax.esviritu_lineage("999", "XCI56374.1")

    assert lineage["subspecies"] == "t__IIb"
    assert len(calls) == 1
    assert calls[0][1]["params"]["q"] == 'AccVer_s:"XCI56374.1"'
    assert calls[0][1]["params"]["fq"] == 'SeqType_s:("Nucleotide")'


def test_vvsearch_empty_genotype_falls_back_to_taxdump(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Mastadenovirus;Human mastadenovirus F",
                "acellular root;genus;species",
            )
        }
    )
    _patch(monkeypatch, fake)
    monkeypatch.setattr(
        "postviritu.taxonomy.requests.get",
        lambda *args, **kwargs: _VVResponse([{"AccVer_s": "NC_001405.1"}]),
    )

    lineage = Taxonomy().esviritu_lineage("999", "NC_001405.1")

    assert lineage["subspecies"] == "t__Human mastadenovirus F"


def test_vvsearch_failure_falls_back_to_taxdump(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Viruses;Mastadenovirus;Human mastadenovirus F",
                "acellular root;genus;species",
            )
        }
    )
    _patch(monkeypatch, fake)

    def timeout(*args, **kwargs):
        raise requests.Timeout("unavailable")

    monkeypatch.setattr("postviritu.taxonomy.requests.get", timeout)

    lineage = Taxonomy().esviritu_lineage("999", "NC_001405.1")

    assert lineage["subspecies"] == "t__Human mastadenovirus F"


def test_nonvirus_lineage_does_not_query_vvsearch(monkeypatch):
    fake = FakePyTaxonkit(
        lineage_rows={
            "999": (
                "Eukaryota;Chordata;Homo sapiens",
                "superkingdom;phylum;species",
            )
        }
    )
    _patch(monkeypatch, fake)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("vvsearch2 should only be queried for viruses")

    monkeypatch.setattr("postviritu.taxonomy.requests.get", unexpected_request)

    lineage = Taxonomy().esviritu_lineage("999", "NC_000001.11")

    assert lineage["kingdom"] == "k__Eukaryota"
