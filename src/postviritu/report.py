"""Standalone HTML report generation for postviritu results."""

from __future__ import annotations

import html
import os
from collections import defaultdict
from typing import Dict, Iterable, Optional

import polars as pl

from .io_esviritu import TAX_RANKS
from .reassign import AssemblyResolution
from .taxonomy import Taxonomy

_MAX_TAXA = 6
_MAX_REFERENCES = 6


def _escape(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _display_taxon(value: Optional[str]) -> str:
    if not value:
        return "unclassified"
    return value.split("__", 1)[-1] or "unclassified"


def _display_reference(value: Optional[str]) -> str:
    text = "" if value is None else str(value)
    parts = text.split("|")
    if len(parts) >= 4 and parts[0].lower() == "gi" and parts[1].isdigit():
        return parts[3] or text
    return text


def _lineage(lineage: Dict[str, Optional[str]]) -> str:
    parts = [
        f'<span><b>{_escape(rank)}</b> {_escape(_display_taxon(lineage.get(rank)))}</span>'
        for rank in TAX_RANKS
        if lineage.get(rank)
    ]
    return "".join(parts) or "<span>unclassified</span>"


def _format_fasta(header: str, seq: str, width: int = 60) -> str:
    lines = [f">{header}"]
    for start in range(0, len(seq), width):
        lines.append(seq[start : start + width])
    return "\n".join(lines)


def _rank_label(n: int) -> str:
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return f"{n}th"


def _alignment(qaln: Optional[str], taln: Optional[str], width: int = 60) -> str:
    if not qaln or not taln:
        return '<div class="alignment-unavailable">Aligned sequences unavailable</div>'
    blocks = []
    for start in range(0, min(len(qaln), len(taln)), width):
        query = qaln[start : start + width]
        target = taln[start : start + width]
        matches = "".join(
            "|" if q.upper() == t.upper() and q != "-" else " "
            for q, t in zip(query, target)
        )
        blocks.append(
            f"Query  {_escape(query)}\n"
            f"       {_escape(matches)}\n"
            f"Ref    {_escape(target)}"
        )
    content = "\n\n".join(blocks)
    return f'<pre class="alignment">{content}</pre>'


def _taxon_name(taxonomy: Taxonomy, taxid: str) -> str:
    lineage = taxonomy.esviritu_lineage(taxid)
    for rank in reversed(TAX_RANKS):
        value = lineage.get(rank)
        if value and _display_taxon(value) != "unclassified":
            return _display_taxon(value)
    return f"Taxid {taxid}"


def _segment(hit: dict, index: int) -> str:
    coordinates = []
    if hit.get("qstart") is not None and hit.get("qend") is not None:
        coordinates.append(f'Query {hit["qstart"]}–{hit["qend"]}')
    if hit.get("tstart") is not None and hit.get("tend") is not None:
        coordinates.append(f'Ref {hit["tstart"]}–{hit["tend"]}')
    bitscore = hit.get("segment_bitscore")
    if bitscore is None:
        bitscore = hit.get("bitscore")
    details = [f"Alignment {index}", *coordinates]
    if bitscore is not None:
        details.append(f"{float(bitscore):.1f} bits")
    if hit.get("segment_selected") is False:
        details.append("excluded from combined score")
    return (
        '<div class="alignment-segment">'
        f'<div class="segment-metrics">{_escape(" · ".join(details))}</div>'
        f'{_alignment(hit.get("qaln"), hit.get("taln"))}'
        "</div>"
    )


def _reference(reference_hits: Iterable[dict], lineage: Dict[str, str]) -> str:
    segments = list(reference_hits)
    hit = segments[0]
    identity = float(hit.get("pct_identity") or 0) * 100
    coverage = float(hit.get("query_coverage") or 0) * 100
    bitscore = float(hit.get("bitscore") or 0)
    evalue = hit.get("evalue")
    evalue_text = f"{float(evalue):.2g}" if evalue is not None else "n/a"
    species = _display_taxon(lineage.get("species"))
    subspecies = _display_taxon(lineage.get("subspecies"))
    taxonomy_line = " · ".join(
        part for part in (species, subspecies) if part and part != "unclassified"
    ) or "unclassified"
    segments.sort(
        key=lambda segment: (
            min(segment.get("qstart"), segment.get("qend"))
            if segment.get("qstart") is not None and segment.get("qend") is not None
            else float("inf")
        )
    )
    return (
        '<details class="reference-alignment">'
        f'<summary><span><span>{_escape(_display_reference(hit.get("target")))}</span>'
        f'<span class="reference-taxonomy">{_escape(taxonomy_line)}</span></span>'
        f'<span class="metrics">{identity:.2f}% ANI · {coverage:.2f}% query · '
        f"{bitscore:.1f} bits · E {evalue_text}</span></summary>"
        f'{"".join(_segment(segment, index) for index, segment in enumerate(segments, 1))}'
        "</details>"
    )


def _taxon_cards(
    query_hits: Iterable[dict],
    taxonomy: Taxonomy,
    tie_frac: float,
) -> str:
    grouped = defaultdict(list)
    for hit in query_hits:
        grouped[str(hit.get("taxid") or "unclassified")].append(hit)

    # Compute max bitscore per taxon and keep the top taxa.
    scored = [
        (taxid, max(float(hit.get("bitscore") or 0) for hit in hits))
        for taxid, hits in grouped.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    scored = scored[:_MAX_TAXA]

    # Assign dense ordinal ranks, allowing ties within ``tie_frac`` of the
    # previous (better) taxon's bitscore.
    ranks: list[int] = []
    prev_bits: Optional[float] = None
    for i, (_, bits) in enumerate(scored):
        if i == 0:
            ranks.append(1)
        elif prev_bits is not None and bits >= tie_frac * prev_bits:
            ranks.append(ranks[-1])
        else:
            ranks.append(ranks[-1] + 1)
        prev_bits = bits

    lineages = {taxid: taxonomy.esviritu_lineage(taxid) for taxid, _ in scored}

    cards = []
    for (taxid, bits), rank in zip(scored, ranks):
        taxon_hits = grouped[taxid]
        reference_groups = defaultdict(list)
        for hit in taxon_hits:
            reference_groups[str(hit.get("target") or "unclassified")].append(hit)
        references = sorted(
            reference_groups.values(),
            key=lambda hits: float(hits[0].get("bitscore") or 0),
            reverse=True,
        )[:_MAX_REFERENCES]
        lineage = lineages[taxid]
        rank_label = _rank_label(rank)
        if ranks.count(rank) > 1:
            rank_label += " (tied)"

        cards.append(
            '<article class="taxon-card">'
            '<header><div>'
            f'<h2>{_escape(_taxon_name(taxonomy, taxid))}</h2>'
            f'<span class="taxon-rank">{_escape(rank_label)}</span>'
            "</div>"
            f'<code>taxid:{_escape(taxid)}</code></header>'
            f'{"".join(_reference(hits, lineage) for hits in references)}'
            "</article>"
        )
    return "".join(cards) or '<div class="no-hits">No acceptable database hits</div>'


def _query_page(
    index: int,
    row: dict,
    hits: pl.DataFrame,
    resolutions: Dict[str, AssemblyResolution],
    taxonomy: Taxonomy,
    consensus_seqs: Optional[Dict[str, str]],
    tie_frac: float,
) -> str:
    query = str(row.get("Accession") or "")
    assembly = row.get("Assembly")
    original = {rank: row.get(rank) for rank in TAX_RANKS}
    resolution = resolutions.get(assembly)
    proposed = resolution.lineage if resolution else original
    decision = resolution.decision if resolution else "unchanged"
    query_hits = (
        hits.filter(pl.col("query") == query).sort("bitscore", descending=True)
        if not hits.is_empty()
        else hits
    )

    consensus = consensus_seqs.get(query) if consensus_seqs else None
    if consensus:
        fasta_content = _format_fasta(f"{query}_consensus", consensus)
        consensus_block = (
            '<details class="consensus-dropdown">'
            '<summary>Consensus sequence (FASTA)</summary>'
            f'<pre class="consensus-fasta">{_escape(fasta_content)}</pre>'
            "</details>"
        )
    else:
        consensus_block = (
            '<details class="consensus-dropdown">'
            '<summary>Consensus sequence (FASTA)</summary>'
            '<div class="consensus-unavailable">Consensus sequence unavailable</div>'
            "</details>"
        )

    return (
        f'<section class="query-page" data-page="{index}" data-query="{_escape(query)}">'
        f'{consensus_block}'
        '<div class="query-heading">'
        f'<div><span class="eyebrow">Query {index + 1}</span><h1>{_escape(query)}</h1></div>'
        f'<div><span class="eyebrow">Assembly</span><code>{_escape(assembly)}</code></div>'
        "</div>"
        '<div class="taxonomy-change">'
        f'<div><h2>Original taxonomy</h2><div class="lineage">{_lineage(original)}</div></div>'
        '<div class="arrow" aria-hidden="true">→</div>'
        f'<div><h2>Proposed taxonomy</h2><div class="lineage">{_lineage(proposed)}</div>'
        f'<div class="decision">{_escape(decision)}</div></div>'
        "</div>"
        f'<div class="taxa-grid">{_taxon_cards(query_hits.iter_rows(named=True), taxonomy, tie_frac)}</div>'
        "</section>"
    )


def write_html_report(
    path: str,
    sample_prefix: str,
    info_df: pl.DataFrame,
    hits: pl.DataFrame,
    resolutions: Dict[str, AssemblyResolution],
    taxonomy: Taxonomy,
    consensus_seqs: Optional[Dict[str, str]] = None,
    tie_frac: float = 0.99,
) -> str:
    """Write a self-contained, paginated report and return its path."""
    filtered_assemblies = {
        asm for asm, res in resolutions.items() if res.decision == "taxa_filtered"
    }
    query_rows = info_df.unique(subset=["Accession"], keep="first", maintain_order=True)
    if filtered_assemblies:
        query_rows = query_rows.filter(
            ~pl.col("Assembly").is_in(list(filtered_assemblies))
        )

    pages = "".join(
        _query_page(index, row, hits, resolutions, taxonomy, consensus_seqs, tie_frac)
        for index, row in enumerate(query_rows.iter_rows(named=True))
    )

    queries = [str(row.get("Accession") or "") for row in query_rows.iter_rows(named=True)]
    datalist_options = "".join(
        f'<option value="{_escape(q)}"></option>' for q in queries
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>postviritu report · {_escape(sample_prefix)}</title>
<style>
:root{{--ink:#101820;--muted:#5b6570;--line:#aab3bb;--paper:#f4f6f7;--panel:#fff;--accent:#005a70}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
nav{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;padding:12px 24px;border-bottom:2px solid var(--ink);background:var(--panel);flex-wrap:wrap}}
nav strong{{margin-right:auto}}.nav-search{{display:flex;align-items:center;gap:8px}}.nav-search input[type=search]{{border:1px solid var(--ink);padding:6px 8px;font:inherit;width:220px}}button{{border:1px solid var(--ink);border-radius:0;background:var(--panel);padding:7px 12px;font:inherit;cursor:pointer}}button:disabled{{color:var(--line);border-color:var(--line);cursor:default}}
main{{max-width:1440px;margin:0 auto;padding:24px}}.query-page{{display:none}}.query-page.active{{display:block}}
.consensus-dropdown{{border:1px solid var(--ink);background:var(--panel);margin-bottom:12px}}.consensus-dropdown summary{{cursor:pointer;padding:12px;font-weight:bold}}.consensus-fasta{{margin:0;padding:12px;border-top:1px solid var(--line);background:#111820;color:#e8f0f2;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:auto}}.consensus-unavailable{{padding:12px;color:var(--muted)}}
.query-heading,.taxonomy-change,.taxon-card{{border:1px solid var(--ink);background:var(--panel)}}.query-heading{{display:flex;justify-content:space-between;align-items:end;padding:16px;margin-bottom:12px}}h1,h2{{margin:0}}h1{{font-size:20px}}h2{{font-size:14px}}.eyebrow{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
.taxonomy-change{{display:grid;grid-template-columns:1fr 42px 1fr;margin-bottom:16px}}.taxonomy-change>div:not(.arrow){{padding:16px}}.taxonomy-change h2{{margin-bottom:10px;text-transform:uppercase}}.arrow{{display:grid;place-items:center;border-left:1px solid var(--line);border-right:1px solid var(--line);font-size:22px}}
.lineage{{display:flex;flex-wrap:wrap;gap:5px}}.lineage span{{border:1px solid var(--line);padding:3px 6px}}.decision{{display:inline-block;margin-top:10px;border:1px solid var(--accent);color:var(--accent);padding:2px 6px}}
.taxa-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.taxon-card header{{display:flex;justify-content:space-between;align-items:center;padding:12px;border-bottom:1px solid var(--ink);background:#e9eef0}}.taxon-card header div{{min-width:0}}.taxon-rank{{display:inline-block;margin-top:4px;margin-right:8px;border:1px solid var(--accent);color:var(--accent);padding:1px 5px;font-size:11px;text-transform:uppercase}}
.reference-alignment{{border-top:1px solid var(--line)}}.reference-alignment:first-of-type{{border-top:0}}summary{{display:flex;justify-content:space-between;gap:12px;padding:10px;cursor:pointer}}.reference-taxonomy{{display:block;color:var(--muted);font-size:12px;margin-top:2px}}.metrics{{color:var(--muted);text-align:right}}.alignment-segment{{border-top:1px solid var(--line)}}.segment-metrics{{padding:6px 12px;color:var(--muted);font-size:11px;background:#f3f5f6}}
.alignment{{overflow:auto;margin:0;padding:12px;border-top:1px solid var(--line);background:#111820;color:#e8f0f2;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}}.alignment-unavailable,.no-hits{{padding:18px;color:var(--muted)}}
@media(max-width:850px){{.taxa-grid{{grid-template-columns:1fr}}.taxonomy-change{{grid-template-columns:1fr}}.arrow{{border:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:6px;transform:rotate(90deg)}}summary{{display:block}}.metrics{{display:block;text-align:left;margin-top:4px}}.nav-search input[type=search]{{width:160px}}}}
</style>
</head>
<body>
<nav><strong>postviritu · {_escape(sample_prefix)}</strong><div class="nav-search"><input type="search" id="page-search" list="page-options" placeholder="Find query..." autocomplete="off"><datalist id="page-options">{datalist_options}</datalist><button id="jump">Go</button></div><button id="prev">Previous</button><span id="counter"></span><button id="next">Next</button></nav>
<main>{pages}</main>
<script>
const pages=[...document.querySelectorAll('.query-page')];let page=0;
function show(next){{page=Math.max(0,Math.min(next,pages.length-1));pages.forEach((node,i)=>node.classList.toggle('active',i===page));document.getElementById('counter').textContent=pages.length?`${{page+1}} / ${{pages.length}}`:'0 / 0';document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=page>=pages.length-1}}
function jumpToQuery(){{const term=document.getElementById('page-search').value.trim();if(!term)return;const exact=pages.findIndex(node=>node.dataset.query===term);const idx=exact>=0?exact:pages.findIndex(node=>node.dataset.query.toLowerCase().includes(term.toLowerCase()));if(idx>=0)show(idx);}}
document.getElementById('prev').addEventListener('click',()=>show(page-1));document.getElementById('next').addEventListener('click',()=>show(page+1));document.getElementById('jump').addEventListener('click',jumpToQuery);document.getElementById('page-search').addEventListener('keydown',event=>{{if(event.key==='Enter')jumpToQuery();}});document.addEventListener('keydown',event=>{{if(event.key==='ArrowLeft')show(page-1);if(event.key==='ArrowRight')show(page+1)}});show(0);
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as report:
        report.write(document)
    return path
