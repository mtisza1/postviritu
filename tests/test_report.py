import polars as pl

from conftest import FakeTaxonomy
from postviritu.aligner import HIT_SCHEMA
from postviritu.io_esviritu import TAX_RANKS
from postviritu.report import write_html_report
from postviritu.reassign import AssemblyResolution


def _info_df():
    row = {
        "sample_ID": "S1",
        "Accession": "query<script>",
        "Assembly": "asm1",
        **{rank: f"{rank[0]}__Original & taxon" for rank in TAX_RANKS},
    }
    return pl.DataFrame([row])


def _hits():
    rows = []
    for taxon in range(7):
        for reference in range(7):
            rows.append(
                {
                    "query": "query<script>",
                    "target": f"reference-{taxon}-{reference}",
                    "taxid": str(100 + taxon),
                    "pct_identity": 0.99 - taxon / 100,
                    "aln_length": 8,
                    "query_length": 8,
                    "query_coverage": 1.0,
                    "evalue": 1e-20,
                    "bitscore": 1000.0 - taxon * 10 - reference,
                    "qaln": "ACGTACGT",
                    "taln": "ACGTTCGT",
                }
            )
    return pl.DataFrame(rows, schema=HIT_SCHEMA)


def _taxonomy():
    return FakeTaxonomy(
        rank_table={
            str(100 + taxon): {"species": f"Database species {taxon}"}
            for taxon in range(7)
        }
    )


def _resolutions():
    return {
        "asm1": AssemblyResolution(
            assembly="asm1",
            lineage={rank: f"{rank[0]}__Proposed taxon" for rank in TAX_RANKS},
            decision="assigned",
        )
    }


def test_write_html_report_paginates_caps_cards_and_escapes_content(tmp_path):
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), _hits(), _resolutions(), _taxonomy())

    report = path.read_text()
    assert report.count('class="query-page"') == 1
    assert report.count('class="taxon-card"') == 6
    assert report.count('class="reference-alignment"') == 36
    assert report.count('class="reference-taxonomy"') == 36
    assert "Original taxonomy" in report
    assert "Proposed taxonomy" in report
    assert "Database species 6" not in report
    assert "reference-0-6" not in report
    assert "query&lt;script&gt;" in report
    assert "query<script>" not in report
    assert "|||| |||" in report
    assert "99.00% ANI" in report


def test_write_html_report_includes_consensus_sequence_dropdown(tmp_path):
    path = tmp_path / "report.html"
    consensus_seqs = {"query<script>": "ACGTACGTACGT"}
    write_html_report(
        str(path),
        "S1",
        _info_df(),
        _hits(),
        _resolutions(),
        _taxonomy(),
        consensus_seqs=consensus_seqs,
    )

    report = path.read_text()
    assert "Consensus sequence (FASTA)" in report
    assert "&gt;query&lt;script&gt;_consensus" in report
    assert "ACGTACGTACGT" in report


def test_write_html_report_adds_search_and_datalist(tmp_path):
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), _hits(), _resolutions(), _taxonomy())

    report = path.read_text()
    assert 'id="page-search"' in report
    assert 'list="page-options"' in report
    assert 'id="page-options"' in report
    assert 'value="query&lt;script&gt;"' in report
    assert 'id="jump"' in report


def test_write_html_report_shows_reference_species_and_subspecies(tmp_path):
    taxonomy = FakeTaxonomy(
        rank_table={
            "100": {
                "superkingdom": "Viruses",
                "species": "Human mastadenovirus F",
                "subspecies": "Human mastadenovirus F1",
            },
        }
    )
    hits = pl.DataFrame(
        [
            {
                "query": "query<script>",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 8,
                "query_coverage": 1.0,
                "evalue": 1e-20,
                "bitscore": 1000.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            }
        ],
        schema=HIT_SCHEMA,
    )
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), hits, _resolutions(), taxonomy)

    report = path.read_text()
    assert (
        '<span class="reference-taxonomy">Human mastadenovirus F · '
        "Human mastadenovirus F1</span>"
    ) in report
    assert 'class="ref-species"' not in report
    assert "1st" in report


def test_write_html_report_strips_gi_prefix_from_reference_accession(tmp_path):
    hits = pl.DataFrame(
        [
            {
                "query": "query<script>",
                "target": "gi|123456789|ref|NC_001405.1|",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 8,
                "query_coverage": 1.0,
                "evalue": 1e-20,
                "bitscore": 1000.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            }
        ],
        schema=HIT_SCHEMA,
    )
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), hits, _resolutions(), _taxonomy())

    report = path.read_text()
    assert ">NC_001405.1</span>" in report
    assert "gi|123456789" not in report


def test_write_html_report_filters_taxa_filtered_records(tmp_path):
    info = _info_df().vstack(
        pl.DataFrame(
            [
                {
                    "sample_ID": "S1",
                    "Accession": "filtered_acc",
                    "Assembly": "asm2",
                    **{rank: f"{rank[0]}__Original & taxon" for rank in TAX_RANKS},
                }
            ]
        )
    )
    resolutions = {
        **_resolutions(),
        "asm2": AssemblyResolution(
            assembly="asm2",
            lineage={rank: f"{rank[0]}__Original & taxon" for rank in TAX_RANKS},
            decision="taxa_filtered",
        ),
    }
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", info, _hits(), resolutions, _taxonomy())

    report = path.read_text()
    assert report.count('class="query-page"') == 1
    assert "filtered_acc" not in report


def test_write_html_report_adds_rank_labels_and_allows_ties(tmp_path):
    hits = pl.DataFrame(
        [
            {
                "query": "query<script>",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 8,
                "query_coverage": 1.0,
                "evalue": 1e-20,
                "bitscore": 1000.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            },
            {
                "query": "query<script>",
                "target": "ref2",
                "taxid": "200",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 8,
                "query_coverage": 1.0,
                "evalue": 1e-20,
                "bitscore": 999.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            },
            {
                "query": "query<script>",
                "target": "ref3",
                "taxid": "300",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 8,
                "query_coverage": 1.0,
                "evalue": 1e-20,
                "bitscore": 900.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            },
        ],
        schema=HIT_SCHEMA,
    )
    taxonomy = FakeTaxonomy(
        rank_table={
            "100": {"species": "Species A"},
            "200": {"species": "Species B"},
            "300": {"species": "Species C"},
        }
    )
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), hits, _resolutions(), taxonomy)

    report = path.read_text()
    assert report.count("1st (tied)") == 2
    assert report.count("2nd") == 1


def test_write_html_report_groups_segments_under_one_reference(tmp_path):
    hits = pl.DataFrame(
        [
            {
                "query": "query<script>",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 20,
                "query_coverage": 0.8,
                "evalue": 1e-20,
                "bitscore": 180.0,
                "qaln": "ACGTACGT",
                "taln": "ACGTACGT",
            },
            {
                "query": "query<script>",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 0.99,
                "aln_length": 8,
                "query_length": 20,
                "query_coverage": 0.8,
                "evalue": 1e-18,
                "bitscore": 180.0,
                "qaln": "TGCATGCA",
                "taln": "TGCATGCA",
            },
        ],
        schema=HIT_SCHEMA,
    )
    path = tmp_path / "report.html"
    write_html_report(str(path), "S1", _info_df(), hits, _resolutions(), _taxonomy())

    report = path.read_text()
    assert report.count('class="reference-alignment"') == 1
    assert report.count('<pre class="alignment">') == 2
    assert report.count(">ref1</span>") == 1
