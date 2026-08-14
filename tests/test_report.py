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


def test_write_html_report_paginates_caps_cards_and_escapes_content(tmp_path):
    taxonomy = FakeTaxonomy(
        rank_table={
            str(100 + taxon): {"species": f"Database species {taxon}"}
            for taxon in range(7)
        }
    )
    resolutions = {
        "asm1": AssemblyResolution(
            assembly="asm1",
            lineage={rank: f"{rank[0]}__Proposed taxon" for rank in TAX_RANKS},
            decision="assigned",
        )
    }
    path = tmp_path / "report.html"

    write_html_report(
        str(path), "S1", _info_df(), _hits(), resolutions, taxonomy
    )

    report = path.read_text()
    assert report.count('class="query-page"') == 1
    assert report.count('class="taxon-card"') == 6
    assert report.count('class="reference-alignment"') == 36
    assert "Original taxonomy" in report
    assert "Proposed taxonomy" in report
    assert "Database species 6" not in report
    assert "reference-0-6" not in report
    assert "query&lt;script&gt;" in report
    assert "query<script>" not in report
    assert "|||| |||" in report
    assert "99.00% ANI" in report
