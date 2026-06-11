import polars as pl

from conftest import FakeTaxonomy, make_hits
from postviritu.io_esviritu import TAX_RANKS
from postviritu.outputs import (
    PROVENANCE_COLUMNS,
    build_assembly_summary,
    build_tax_profile,
    rebuild_info,
)
from postviritu.reassign import resolve_assemblies

INFO_COLUMNS = [
    "sample_ID", "Name", "description", "Length", "Segment", "Accession",
    "Assembly", "Asm_length", *TAX_RANKS, "RPKMF", "read_count",
    "covered_bases", "mean_coverage", "avg_read_identity", "Pi",
    "filtered_reads_in_sample",
]


def _info_df():
    def row(acc, asm, seg, length, reads):
        return {
            "sample_ID": "S1", "Name": acc, "description": "d", "Length": length,
            "Segment": seg, "Accession": acc, "Assembly": asm,
            "Asm_length": length, "kingdom": "k__Viruses", "phylum": "p__OrigP",
            "tclass": "c__OrigC", "order": "o__OrigO", "family": "f__OrigF",
            "genus": "g__OrigG", "species": "s__OrigSp", "subspecies": "t__OrigSub",
            "RPKMF": 1.0, "read_count": reads, "covered_bases": 100,
            "mean_coverage": 2.0, "avg_read_identity": 0.97, "Pi": 0.0,
            "filtered_reads_in_sample": 1_000_000,
        }

    return pl.DataFrame(
        [
            row("seg1", "asmSeg", "RNA1", 5000, 50),
            row("seg2", "asmSeg", "RNA2", 5000, 30),
            row("solo", "asmSolo", None, 30000, 200),
        ],
        schema_overrides={"Segment": pl.Utf8},
    ).select(INFO_COLUMNS)


def _taxonomy():
    return FakeTaxonomy(
        rank_table={
            "100": {"superkingdom": "Viruses", "family": "Reoviridae",
                    "genus": "Rotavirus", "species": "Rotavirus A"},
            "200": {"superkingdom": "Viruses", "family": "Adenoviridae",
                    "genus": "Mastadenovirus", "species": "Human mastadenovirus F"},
        }
    )


def _hits():
    return make_hits(
        [
            {"query": "seg1", "target": "t100", "taxid": "100",
             "pct_identity": 0.99, "aln_length": 4800, "query_length": 5000,
             "query_coverage": 0.96, "evalue": 1e-50, "bitscore": 800.0},
            {"query": "solo", "target": "t200", "taxid": "200",
             "pct_identity": 0.99, "aln_length": 29000, "query_length": 30000,
             "query_coverage": 0.96, "evalue": 1e-99, "bitscore": 5000.0},
        ]
    )


def _resolve():
    return resolve_assemblies(_hits(), _info_df(), _taxonomy(), mode="scratch")


def test_rebuild_info_preserves_columns_and_adds_provenance():
    new_info = rebuild_info(_info_df(), _resolve())
    assert new_info.columns == INFO_COLUMNS + PROVENANCE_COLUMNS
    # Both segments of asmSeg get the same new species.
    seg_species = (
        new_info.filter(pl.col("Assembly") == "asmSeg")["species"].unique().to_list()
    )
    assert seg_species == ["s__Rotavirus A"]
    # Metrics unchanged.
    assert new_info.filter(pl.col("Accession") == "solo")["read_count"][0] == 200


def test_assembly_summary_groups_segments():
    new_info = rebuild_info(_info_df(), _resolve())
    asm = build_assembly_summary(new_info)
    seg_row = asm.filter(pl.col("Assembly") == "asmSeg").row(0, named=True)
    # Segment read counts summed (50 + 30) and accessions joined.
    assert seg_row["read_count"] == 80
    assert set(seg_row["Accession"].split(",")) == {"seg1", "seg2"}
    assert "RNA1" in seg_row["Segment"] and "RNA2" in seg_row["Segment"]


def test_tax_profile_schema_and_assembly_list():
    new_info = rebuild_info(_info_df(), _resolve())
    tax = build_tax_profile(new_info, _resolve(), spthresh=0.9, subspthresh=0.95)
    expected = [
        "sample_ID", "filtered_reads_in_sample", *TAX_RANKS,
        "read_count", "RPKMF", "avg_read_identity", "assembly_list",
    ]
    assert tax.columns == expected
    species = set(tax["species"].to_list())
    assert "s__Rotavirus A" in species
    assert "s__Human mastadenovirus F" in species
