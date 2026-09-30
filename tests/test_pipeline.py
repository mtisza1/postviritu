import os

import polars as pl

from conftest import FakeAligner, FakeTaxonomy, make_hits
from postviritu.io_esviritu import SamplePaths, read_tsv
from postviritu.run import RunConfig, run_sample


def test_run_sample_scratch_no_hits_unclassified(
    example_data_dir, example_prefix, tmp_path
):
    sample = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    aligner = FakeAligner(make_hits([]))
    taxonomy = FakeTaxonomy()
    outdir = str(tmp_path / "out")
    paths = run_sample(sample, aligner, taxonomy, outdir, RunConfig(mode="scratch"))

    assert os.path.isfile(paths["info"])
    assert os.path.isfile(paths["assembly_summary"])
    assert os.path.isfile(paths["tax_profile"])
    assert os.path.isfile(paths["report"])
    with open(paths["report"]) as report:
        assert report.read().count('class="query-page"') == 2
    # Coverage windows passed through unchanged.
    assert os.path.isfile(
        os.path.join(outdir, example_prefix + ".virus_coverage_windows.tsv")
    )

    tax = read_tsv(paths["tax_profile"])
    # With no hits, scratch mode marks everything unclassified.
    assert all(s.startswith("s__unclassified_") for s in tax["species"].to_list())


def test_run_sample_scratch_with_hit_updates_assembly(
    example_data_dir, example_prefix, tmp_path
):
    sample = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    info = read_tsv(sample.info)
    # Pick a real accession/assembly present in the info table.
    target_acc = "OR777233.1"
    assembly = info.filter(pl.col("Accession") == target_acc)["Assembly"][0]

    hits = make_hits(
        [
            {
                "query": target_acc, "target": "dbhit", "taxid": "999",
                "pct_identity": 0.995, "aln_length": 33000,
                "query_length": 33806, "query_coverage": 0.98,
                "evalue": 1e-99, "bitscore": 6000.0,
            }
        ]
    )
    taxonomy = FakeTaxonomy(
        rank_table={
            "999": {
                "superkingdom": "Viruses", "phylum": "Preplasmiviricota",
                "class": "Tectiliviricetes", "order": "Rowavirales",
                "family": "Adenoviridae", "genus": "Mastadenovirus",
                "species": "Reassigned virus sp.",
            }
        }
    )
    outdir = str(tmp_path / "out")
    paths = run_sample(sample, aligner=FakeAligner(hits), taxonomy=taxonomy,
                       outdir=outdir, config=RunConfig(mode="scratch"))

    new_info = read_tsv(paths["info"])
    row = new_info.filter(pl.col("Accession") == target_acc).row(0, named=True)
    assert row["species"] == "s__Reassigned virus sp."
    assert row["postviritu_decision"] == "assigned"
    assert str(row["postviritu_hit_taxid"]) == "999"
    # The assembly that got the hit is recorded; others are unclassified.
    assert row["Assembly"] == assembly


def test_report_proposed_taxonomy_matches_thresholded_tax_profile(
    example_data_dir, example_prefix, tmp_path
):
    """Below subspthresh the report must not name a subspecies either."""
    import html
    import re

    sample = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    target_acc = "OR777233.1"
    hits = make_hits(
        [
            {
                "query": target_acc, "target": "dbhit", "taxid": "999",
                "pct_identity": 0.93, "aln_length": 33000,
                "query_length": 33806, "query_coverage": 0.98,
                "evalue": 1e-99, "bitscore": 6000.0,
            }
        ]
    )
    taxonomy = FakeTaxonomy(
        rank_table={
            "999": {
                "superkingdom": "Viruses", "family": "Adenoviridae",
                "genus": "Mastadenovirus", "species": "Reassigned virus sp.",
                "subspecies": "Reassigned strain X",
            }
        }
    )
    paths = run_sample(sample, aligner=FakeAligner(hits), taxonomy=taxonomy,
                       outdir=str(tmp_path / "out"), config=RunConfig(mode="scratch"))

    tax = read_tsv(paths["tax_profile"]).filter(pl.col("species") == "s__Reassigned virus sp.")
    assert tax["subspecies"].to_list() == ["t__unclassified_Reassigned virus sp."]

    with open(paths["report"]) as report:
        page = re.search(
            rf'<section class="query-page"[^>]*data-query="{re.escape(target_acc)}".*?</section>',
            report.read(),
            re.S,
        ).group(0)
    proposed = html.unescape(page.split("Proposed taxonomy", 1)[1].split("</div>", 1)[0])
    assert "subspecies</b> unclassified_Reassigned virus sp." in proposed
    assert "Reassigned strain X" not in proposed


def _reassigning_run(data_dir, prefix, tmp_path, pct_identity=0.995, mode="scratch"):
    sample = SamplePaths(prefix=prefix, directory=data_dir)
    hits = make_hits(
        [
            {
                "query": "OR777233.1", "target": "dbhit", "taxid": "999",
                "pct_identity": pct_identity, "aln_length": 33000,
                "query_length": 33806, "query_coverage": 0.98,
                "evalue": 1e-99, "bitscore": 6000.0,
            }
        ]
    )
    taxonomy = FakeTaxonomy(
        rank_table={
            "999": {
                "superkingdom": "Viruses", "genus": "Mastadenovirus",
                "species": "Reassigned virus sp.", "subspecies": "Reassigned strain X",
            }
        }
    )
    return run_sample(sample, aligner=FakeAligner(hits), taxonomy=taxonomy,
                      outdir=str(tmp_path / "out"), config=RunConfig(mode=mode))


def test_run_sample_handles_esviritu_v13_outputs(example_v13_data_dir, example_prefix, tmp_path):
    paths = _reassigning_run(example_v13_data_dir, example_prefix, tmp_path)

    info = read_tsv(paths["info"])
    row = info.filter(pl.col("Accession") == "OR777233.1").row(0, named=True)
    # The new '{Accession}_{sample}_consensus' header was matched to the info row.
    assert row["species"] == "s__Reassigned virus sp."
    assert row["postviritu_decision"] == "assigned"
    assert {"adj_taxonomy", "consensus_ref_identity"} <= set(info.columns)

    assem = read_tsv(paths["assembly_summary"])
    cols = assem.columns
    assert cols.index("adj_taxonomy") == cols.index("subspecies") + 1
    assert cols.index("consensus_ref_identity") == cols.index("avg_read_identity") + 1
    assert assem["consensus_ref_identity"].to_list() == [0.93]

    tax = read_tsv(paths["tax_profile"])
    assert tax.columns[-2:] == ["consensus_ref_identity", "assembly_list"]
    # Hit identity (99.5%) still drives thresholding when there is a hit.
    assert tax["subspecies"].to_list() == ["t__Reassigned strain X"]


def test_old_esviritu_outputs_keep_old_schema(example_data_dir, example_prefix, tmp_path):
    paths = _reassigning_run(example_data_dir, example_prefix, tmp_path)
    for key in ("assembly_summary", "tax_profile"):
        cols = read_tsv(paths[key]).columns
        assert "consensus_ref_identity" not in cols and "adj_taxonomy" not in cols


def test_no_hit_threshold_falls_back_to_consensus_ref_identity(
    example_v13_data_dir, example_prefix, tmp_path
):
    """Like EsViritu >= 1.3: consensus_ref_identity (0.93) before avg_read_identity (0.97)."""
    sample = SamplePaths(prefix=example_prefix, directory=example_v13_data_dir)
    paths = run_sample(sample, FakeAligner(make_hits([])), FakeTaxonomy(),
                       str(tmp_path / "out"), RunConfig(mode="disagree"))

    tax = read_tsv(paths["tax_profile"])
    assert tax["subspecies"].to_list() == ["t__unclassified_OrigSp1"]
