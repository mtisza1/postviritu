import os

import polars as pl
import pytest
import yaml

from conftest import FakeAligner, FakeTaxonomy, make_hits
from postviritu.io_esviritu import SamplePaths, read_tsv, write_fasta
from postviritu.run import RunConfig, _split_info_by_taxa_filter, run_sample
from postviritu.taxonomy import TaxaFilter


def _info_row(accession, assembly, species, genus="g__OrigGen"):
    return {
        "sample_ID": "S1",
        "Name": accession,
        "description": ".",
        "Length": 30000,
        "Segment": None,
        "Accession": accession,
        "Assembly": assembly,
        "Asm_length": 30000,
        "kingdom": "k__Viruses",
        "phylum": "p__",
        "tclass": "c__",
        "order": "o__",
        "family": "f__",
        "genus": genus,
        "species": species,
        "subspecies": "t__",
        "RPKMF": 10.0,
        "read_count": 100,
        "covered_bases": 1000,
        "mean_coverage": 2.0,
        "avg_read_identity": 0.97,
        "Pi": 0.0,
        "filtered_reads_in_sample": 1_000_000,
    }


def _make_sample_dir(tmp_path, prefix, rows, consensus_seqs):
    """Create a minimal sample directory for filter tests."""
    root = str(tmp_path / "input")
    os.makedirs(root, exist_ok=True)

    info = pl.DataFrame(rows, schema_overrides={"Segment": pl.Utf8})
    info.write_csv(
        os.path.join(root, f"{prefix}.detected_virus.info.tsv"), separator="\t"
    )

    open(os.path.join(root, f"{prefix}.virus_coverage_windows.tsv"), "w").close()
    with open(os.path.join(root, f"{prefix}_esviritu.params.yaml"), "w") as fh:
        yaml.safe_dump({"spthresh": 0.9, "subspthresh": 0.95}, fh)

    write_fasta(consensus_seqs, os.path.join(root, f"{prefix}_final_consensus.fasta"))
    return root


def test_taxa_filter_matches_with_and_without_prefix():
    tf = TaxaFilter({"species": ["s__KeepSp", "Human mastadenovirus A"]})
    assert tf.matches({"species": "s__KeepSp"})
    assert tf.matches({"species": "s__Human mastadenovirus A"})
    assert not tf.matches({"species": "s__OtherSp"})


def test_taxa_filter_matches_case_insensitive():
    tf = TaxaFilter({"species": ["s__keepsp"]})
    assert tf.matches({"species": "s__KeepSp"})
    assert tf.matches({"species": "S__KEEPSP"})
    tf2 = TaxaFilter({"genus": ["Mastadenovirus"]})
    assert tf2.matches({"genus": "g__Mastadenovirus"})
    assert tf2.matches({"genus": "g__MASTADENOVIRUS"})


def test_taxa_filter_matches_any_rank():
    tf = TaxaFilter({"genus": ["g__Enterovirus"]})
    assert tf.matches({"genus": "g__Enterovirus", "species": "s__SomeSpecies"})
    assert not tf.matches({"genus": "g__Mastadenovirus"})


def test_taxa_filter_empty_matches_all():
    tf = TaxaFilter({})
    assert tf.matches({"species": "s__Anything"})
    tf2 = TaxaFilter()
    assert tf2.matches({"species": "s__Anything"})


def test_taxa_filter_class_alias():
    tf = TaxaFilter({"class": ["Tectiliviricetes"]})
    assert tf.matches({"tclass": "c__Tectiliviricetes"})


def test_taxa_filter_normalizes_rank_keys_and_values():
    tf = TaxaFilter({" Species ": [123]})
    assert tf.matches({"species": "s__123"})


def test_taxa_filter_rejects_unknown_rank():
    with pytest.raises(ValueError, match="Unknown taxonomy rank"):
        TaxaFilter({"notarank": ["s__Foo"]})


@pytest.mark.parametrize("values", [[], None, {}])
def test_taxa_filter_rejects_rank_with_no_taxa(values):
    # An include list naming a rank but no taxa would match nothing at all and
    # silently empty every sample; it must be an error, not a no-op.
    with pytest.raises(ValueError, match="lists no taxa"):
        TaxaFilter({"species": values})


def test_taxa_filter_allows_one_empty_rank_among_several():
    tf = TaxaFilter({"species": [], "genus": ["g__KeepGen"]})
    assert tf.include == {"genus": {"g__keepgen"}}
    assert tf.matches({"genus": "g__KeepGen"})
    assert not tf.matches({"species": "s__Anything"})


def test_taxa_filter_merges_class_and_tclass():
    tf = TaxaFilter({"class": ["Tectiliviricetes"], "tclass": ["Caudoviricetes"]})
    assert tf.matches({"tclass": "c__Tectiliviricetes"})
    assert tf.matches({"tclass": "c__Caudoviricetes"})


def test_taxa_filter_rejects_entry_with_another_ranks_prefix():
    with pytest.raises(ValueError, match="carries the 's__' "):
        TaxaFilter({"genus": ["s__Human mastadenovirus A"]})


def test_taxa_filter_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        TaxaFilter(["s__Foo"])


def test_taxa_filter_from_yaml_rejects_top_level_list(tmp_path):
    path = tmp_path / "filter.yaml"
    path.write_text('- "s__KeepSp"\n- "s__OtherSp"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        TaxaFilter.from_yaml(str(path))


def test_taxa_filter_from_yaml_empty_file_matches_all(tmp_path):
    path = tmp_path / "filter.yaml"
    path.write_text("", encoding="utf-8")
    assert TaxaFilter.from_yaml(str(path)).matches({"species": "s__Anything"})


def test_taxa_filter_from_yaml(tmp_path):
    path = tmp_path / "filter.yaml"
    path.write_text(
        'species:\n  - "s__KeepSp"\ngenus:\n  - "g__KeepGen"\n', encoding="utf-8"
    )
    tf = TaxaFilter.from_yaml(str(path))
    assert tf.matches({"species": "s__KeepSp"})
    assert tf.matches({"genus": "g__KeepGen"})
    assert not tf.matches({"species": "s__OtherSp"})


def test_split_info_keeps_null_assembly_rows():
    # `is_in` is null for a null Assembly, which would drop the row from both
    # halves of the split and leave it with null taxonomy in the final output.
    rows = [
        _info_row("accKeep", "asmKeep", "s__KeepSp"),
        _info_row("accSkip", "asmSkip", "s__SkipSp"),
        _info_row("accNull", None, "s__KeepSp"),
    ]
    info_df = pl.DataFrame(rows, schema_overrides={"Segment": pl.Utf8})
    included, excluded = _split_info_by_taxa_filter(
        info_df, TaxaFilter({"species": ["s__KeepSp"]})
    )

    assert included.height + excluded.height == info_df.height
    assert included["Accession"].to_list() == ["accKeep", "accNull"]
    assert excluded["Accession"].to_list() == ["accSkip"]


def test_run_sample_taxa_filter_includes_only_matching_assembly(tmp_path):
    prefix = "S1"
    rows = [
        _info_row("accKeep", "asmKeep", "s__KeepSp", genus="g__KeepGen"),
        _info_row("accSkip", "asmSkip", "s__SkipSp", genus="g__SkipGen"),
    ]
    root = _make_sample_dir(
        tmp_path,
        prefix,
        rows,
        {"accKeep": "ACGT" * 100, "accSkip": "TGCA" * 100},
    )

    hits = make_hits(
        [
            {
                "query": "accKeep",
                "target": "tgt",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 30000,
                "query_length": 30000,
                "query_coverage": 0.95,
                "evalue": 1e-50,
                "bitscore": 5000.0,
            }
        ]
    )
    taxonomy = FakeTaxonomy(
        rank_table={
            "100": {
                "superkingdom": "Viruses",
                "family": "Keepidae",
                "genus": "KeepGen",
                "species": "ReassignedKeepSp",
            }
        }
    )
    aligner = FakeAligner(hits)
    taxa_filter = TaxaFilter({"species": ["s__KeepSp"]})
    config = RunConfig(mode="scratch", taxa_filter=taxa_filter)

    sample = SamplePaths(prefix=prefix, directory=root)
    outdir = str(tmp_path / "out")
    paths = run_sample(sample, aligner, taxonomy, outdir, config)

    new_info = read_tsv(paths["info"])
    keep_row = new_info.filter(pl.col("Assembly") == "asmKeep").row(0, named=True)
    skip_row = new_info.filter(pl.col("Assembly") == "asmSkip").row(0, named=True)

    assert keep_row["species"] == "s__ReassignedKeepSp"
    assert keep_row["postviritu_decision"] == "assigned"
    assert skip_row["species"] == "s__SkipSp"
    assert skip_row["postviritu_decision"] == "taxa_filtered"
    assert skip_row["postviritu_hit_taxid"] is None


def test_run_sample_taxa_filter_excludes_all_keeps_original(tmp_path, caplog):
    caplog.set_level("INFO", logger="postviritu")
    prefix = "S1"
    rows = [
        _info_row("accA", "asmA", "s__SpA"),
        _info_row("accB", "asmB", "s__SpB"),
    ]
    for row in rows:
        row["avg_read_identity"] = 0.5
    root = _make_sample_dir(
        tmp_path, prefix, rows, {"accA": "ACGT" * 50, "accB": "TGCA" * 50}
    )

    class RaisingAligner:
        def search(self, *args, **kwargs):
            raise AssertionError("aligner.search should not be called when all taxa are excluded")

    taxonomy = FakeTaxonomy()
    aligner = RaisingAligner()
    taxa_filter = TaxaFilter({"species": ["s__NotPresent"]})
    config = RunConfig(mode="scratch", taxa_filter=taxa_filter)

    sample = SamplePaths(prefix=prefix, directory=root)
    outdir = str(tmp_path / "out")
    paths = run_sample(sample, aligner, taxonomy, outdir, config)

    new_info = read_tsv(paths["info"])
    species = set(new_info["species"].to_list())
    assert species == {"s__SpA", "s__SpB"}
    assert all(
        d == "taxa_filtered" for d in new_info["postviritu_decision"].to_list()
    )
    # Filtered assemblies are a pass-through: the tax_profile applies the same
    # avg_read_identity thresholding EsViritu itself would have applied, so an
    # identity of 0.5 (< spthresh) collapses the species to unclassified.
    tax_profile = read_tsv(paths["tax_profile"])
    assert set(tax_profile["species"].to_list()) == {"s__unclassified_OrigGen"}
    assert not os.path.exists(os.path.join(outdir, f"{prefix}_tmp"))
    assert "S1: 0/2 assemblies pass taxa filter" in caplog.text
    assert any(
        r.levelname == "WARNING" and "S1: no assemblies matched the taxa filter" in r.getMessage()
        for r in caplog.records
    )


def test_taxa_filter_default_is_high_concern_list():
    tf = TaxaFilter.from_spec(None)
    assert tf.matches({"species": "s__Orthopoxvirus monkeypox"})
    assert tf.matches({"species": "s__Betacoronavirus pandemicum"})
    assert tf.matches({"genus": "g__Ebolavirus"})
    assert tf.matches({"species": "s__Enterovirus coxsackiepol", "subspecies": "t__Poliovirus 2"})
    assert not tf.matches({"species": "s__Human mastadenovirus F", "genus": "g__Mastadenovirus"})


@pytest.mark.parametrize("spec", ["all", "ALL", " All "])
def test_taxa_filter_all_disables_filtering(spec):
    assert TaxaFilter.from_spec(spec) is None


def test_taxa_filter_spec_path_loads_yaml(tmp_path):
    path = tmp_path / "taxa.yaml"
    path.write_text(yaml.safe_dump({"species": ["s__KeepSp"]}))
    tf = TaxaFilter.from_spec(str(path))
    assert tf.matches({"species": "s__KeepSp"})
    assert not tf.matches({"species": "s__Orthopoxvirus monkeypox"})


def test_cli_defaults_to_high_concern_filter():
    from postviritu.cli import _build_parser, _taxa_filter

    parser = _build_parser()
    base = ["blastn", "--input-dir", "in", "--outdir", "out"]
    default = _taxa_filter(parser.parse_args(base))
    assert default is not None and default.matches({"genus": "g__Henipavirus"})
    assert _taxa_filter(parser.parse_args(base + ["--taxa-filter", "all"])) is None
