import polars as pl

from conftest import FakeAligner, FakeTaxonomy, make_hits
from postviritu.io_esviritu import TAX_RANKS
from postviritu.reassign import (
    MODE_DISAGREE,
    MODE_SCRATCH,
    resolve_assemblies,
)


def _info_row(accession, assembly, species, subspecies):
    row = {
        "Accession": accession,
        "Assembly": assembly,
        "read_count": 10,
        "kingdom": "k__Viruses",
        "phylum": "p__OrigPhy",
        "tclass": "c__OrigCls",
        "order": "o__OrigOrd",
        "family": "f__OrigFam",
        "genus": "g__OrigGen",
        "species": species,
        "subspecies": subspecies,
    }
    return row


def _info_df():
    return pl.DataFrame(
        [
            _info_row("accA1", "asmA", "s__OrigSpeciesA", "t__OrigStrainA"),
            _info_row("accB1", "asmB", "s__OrigSpeciesB", "t__OrigStrainB"),
        ]
    )


def _taxonomy():
    return FakeTaxonomy(
        rank_table={
            "100": {
                "superkingdom": "Viruses",
                "family": "Adenoviridae",
                "genus": "Mastadenovirus",
                "species": "Human mastadenovirus F",
            },
            "50": {"superkingdom": "Viruses", "family": "Adenoviridae"},
        },
        lca_table={frozenset({"200", "300"}): "50"},
    )


def test_scratch_single_taxon_assigned():
    hits = make_hits(
        [
            {
                "query": "accA1",
                "target": "tgtA",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 1000,
                "query_length": 1000,
                "query_coverage": 0.95,
                "evalue": 1e-50,
                "bitscore": 500.0,
            }
        ]
    )
    res = resolve_assemblies(
        hits, _info_df(), _taxonomy(), mode=MODE_SCRATCH
    )
    a = res["asmA"]
    assert a.decision == "assigned"
    assert a.ambiguous is False
    assert a.lineage["species"] == "s__Human mastadenovirus F"
    # No hits for asmB -> fully unclassified in scratch mode.
    assert res["asmB"].decision == "no_hit_unclassified"
    assert res["asmB"].lineage["species"] == "s__unclassified_Viruses"


def test_scratch_combines_segments_for_provenance_score():
    hits = make_hits(
        [
            {
                "query": "accA1",
                "target": "tgtA",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 400,
                "query_length": 1000,
                "query_coverage": 0.4,
                "evalue": 1e-50,
                "bitscore": 500.0,
                "qaln": "A" * 400,
                "taln": "A" * 400,
                "qstart": 1,
                "qend": 400,
                "tstart": 1,
                "tend": 400,
            },
            {
                "query": "accA1",
                "target": "tgtA",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 400,
                "query_length": 1000,
                "query_coverage": 0.4,
                "evalue": 1e-40,
                "bitscore": 400.0,
                "qaln": "C" * 400,
                "taln": "C" * 400,
                "qstart": 601,
                "qend": 1000,
                "tstart": 601,
                "tend": 1000,
            },
        ]
    )

    resolution = resolve_assemblies(
        hits, _info_df(), _taxonomy(), mode=MODE_SCRATCH
    )["asmA"]

    assert resolution.hit_accession == "tgtA"
    assert resolution.bitscore == 900.0
    assert resolution.pct_identity == 1.0


def test_scratch_tie_uses_lca():
    hits = make_hits(
        [
            {
                "query": "accB1", "target": "t200", "taxid": "200",
                "pct_identity": 0.97, "aln_length": 900, "query_length": 1000,
                "query_coverage": 0.9, "evalue": 1e-40, "bitscore": 400.0,
            },
            {
                "query": "accB1", "target": "t300", "taxid": "300",
                "pct_identity": 0.96, "aln_length": 900, "query_length": 1000,
                "query_coverage": 0.9, "evalue": 1e-40, "bitscore": 400.0,
            },
        ]
    )
    res = resolve_assemblies(hits, _info_df(), _taxonomy(), mode=MODE_SCRATCH)
    b = res["asmB"]
    assert b.ambiguous is True
    assert b.decision == "lca_ambiguous"
    assert b.lineage["family"] == "f__Adenoviridae"
    # LCA at family -> species/genus become unclassified.
    assert b.lineage["species"].startswith("s__unclassified_")


def test_disagree_override_on_species_mismatch():
    hits = make_hits(
        [
            {
                "query": "accA1", "target": "tgtA", "taxid": "100",
                "pct_identity": 0.99, "aln_length": 1000, "query_length": 1000,
                "query_coverage": 0.95, "evalue": 1e-50, "bitscore": 500.0,
            }
        ]
    )
    res = resolve_assemblies(hits, _info_df(), _taxonomy(), mode=MODE_DISAGREE)
    a = res["asmA"]
    # New species (Human mastadenovirus F) != original (OrigSpeciesA) -> override.
    assert a.decision == "overridden"
    assert a.lineage["species"] == "s__Human mastadenovirus F"


def test_disagree_keeps_original_when_agree():
    info = pl.DataFrame(
        [_info_row("accA1", "asmA", "s__Human mastadenovirus F", "t__strain")]
    )
    hits = make_hits(
        [
            {
                "query": "accA1", "target": "tgtA", "taxid": "100",
                "pct_identity": 0.99, "aln_length": 1000, "query_length": 1000,
                "query_coverage": 0.95, "evalue": 1e-50, "bitscore": 500.0,
            }
        ]
    )
    res = resolve_assemblies(hits, info, _taxonomy(), mode=MODE_DISAGREE)
    a = res["asmA"]
    assert a.decision == "kept_original"
    assert a.lineage["species"] == "s__Human mastadenovirus F"
    assert a.lineage["subspecies"] == "t__strain"


def test_disagree_agreement_uses_vvsearch_genotype():
    info = pl.DataFrame(
        [_info_row("accA1", "asmA", "s__Human mastadenovirus F", "t__strain")]
    )
    hits = make_hits(
        [
            {
                "query": "accA1", "target": "tgtA", "taxid": "100",
                "pct_identity": 0.99, "aln_length": 1000, "query_length": 1000,
                "query_coverage": 0.95, "evalue": 1e-50, "bitscore": 500.0,
            }
        ]
    )
    taxonomy = _taxonomy()
    taxonomy.genotype_table["tgtA"] = "GII.4"

    resolution = resolve_assemblies(
        hits, info, taxonomy, mode=MODE_DISAGREE
    )["asmA"]

    assert resolution.decision == "kept_original"
    assert resolution.lineage["subspecies"] == "t__GII.4"


def test_disagree_no_hit_keeps_original():
    res = resolve_assemblies(
        make_hits([]), _info_df(), _taxonomy(), mode=MODE_DISAGREE
    )
    assert res["asmA"].decision == "no_hit_kept_original"
    assert res["asmA"].lineage["species"] == "s__OrigSpeciesA"


def test_disagree_realign_tie_detected():
    hits = make_hits(
        [
            {
                "query": "accA1", "target": "tgtA", "taxid": "100",
                "pct_identity": 0.99, "aln_length": 1000, "query_length": 1000,
                "query_coverage": 0.95, "evalue": 1e-50, "bitscore": 500.0,
            }
        ]
    )
    # Second-round (excluding taxid 100) returns an equally-good taxon 50.
    second = make_hits(
        [
            {
                "query": "accA1", "target": "tgtB", "taxid": "50",
                "pct_identity": 0.98, "aln_length": 1000, "query_length": 1000,
                "query_coverage": 0.95, "evalue": 1e-49, "bitscore": 499.0,
            }
        ]
    )
    tax = FakeTaxonomy(
        rank_table={
            "100": {"superkingdom": "Viruses", "family": "Adenoviridae",
                    "species": "Human mastadenovirus F"},
            "lca100_50": {"superkingdom": "Viruses", "family": "Adenoviridae"},
        },
        lca_table={frozenset({"100", "50"}): "lca100_50"},
    )
    aligner = FakeAligner(hits, second_round=second)
    res = resolve_assemblies(
        hits,
        _info_df(),
        tax,
        mode=MODE_DISAGREE,
        aligner=aligner,
        consensus_seqs={"accA1": "ACGT" * 50},
        realign_ties=True,
    )
    a = res["asmA"]
    assert a.ambiguous is True
    assert a.decision == "lca_ambiguous_realigned"


def _hit(query="accA1", target="tgtA", taxid="100", identity=0.99, bitscore=500.0):
    return {
        "query": query,
        "target": target,
        "taxid": taxid,
        "pct_identity": identity,
        "aln_length": 1000,
        "query_length": 1000,
        "query_coverage": 0.95,
        "evalue": 1e-50,
        "bitscore": bitscore,
    }


def test_disagree_override_carries_genotype():
    """An overridden call keeps the enriched subspecies, not just the agreeing one."""
    info = pl.DataFrame([_info_row("accA1", "asmA", "s__OrigSpeciesA", "t__strain")])
    taxonomy = _taxonomy()
    taxonomy.genotype_table["tgtA"] = "F41"

    resolution = resolve_assemblies(
        make_hits([_hit()]), info, taxonomy, mode=MODE_DISAGREE
    )["asmA"]

    assert resolution.decision == "overridden"
    assert resolution.lineage["species"] == "s__Human mastadenovirus F"
    assert resolution.lineage["subspecies"] == "t__F41"


def test_ambiguous_lca_skips_genotype_lookup():
    """An LCA call is not a subspecies-level claim, so no genotype is fetched."""
    info = pl.DataFrame([_info_row("accA1", "asmA", "s__OrigSpeciesA", "t__strain")])
    taxonomy = _taxonomy()
    taxonomy.genotype_table["tgtA"] = "F41"

    resolution = resolve_assemblies(
        make_hits(
            [
                _hit(target="tgtA", taxid="200", bitscore=500.0),
                _hit(target="tgtB", taxid="300", bitscore=499.0),
            ]
        ),
        info,
        taxonomy,
        mode=MODE_SCRATCH,
    )["asmA"]

    assert resolution.ambiguous is True
    assert resolution.lineage["subspecies"] != "t__F41"
    assert taxonomy.genotype_queries == []


def test_low_identity_skips_genotype_lookup():
    """Below subspthresh no subspecies is asserted, so no request is spent."""
    info = pl.DataFrame([_info_row("accA1", "asmA", "s__OrigSpeciesA", "t__strain")])
    taxonomy = _taxonomy()
    taxonomy.genotype_table["tgtA"] = "F41"

    resolution = resolve_assemblies(
        make_hits([_hit(identity=0.92)]),
        info,
        taxonomy,
        mode=MODE_SCRATCH,
        subspthresh=0.95,
    )["asmA"]

    assert resolution.lineage["subspecies"] != "t__F41"
    assert taxonomy.genotype_queries == []


def test_identity_at_threshold_performs_genotype_lookup():
    info = pl.DataFrame([_info_row("accA1", "asmA", "s__OrigSpeciesA", "t__strain")])
    taxonomy = _taxonomy()
    taxonomy.genotype_table["tgtA"] = "F41"

    resolution = resolve_assemblies(
        make_hits([_hit(identity=0.95)]),
        info,
        taxonomy,
        mode=MODE_SCRATCH,
        subspthresh=0.95,
    )["asmA"]

    assert resolution.lineage["subspecies"] == "t__F41"
    assert taxonomy.genotype_queries == ["tgtA"]
