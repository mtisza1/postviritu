"""Rebuild EsViritu-format output tables with reassigned taxonomy.

Quantitative metrics (read_count, RPKMF, covered_bases, mean_coverage, Pi,
avg_read_identity) are reused unchanged; only taxonomy columns are replaced,
and provenance columns are appended. Tables are re-aggregated by the new
taxonomy to mirror EsViritu's ``assembly_table_maker`` / ``tax_profile`` logic.
"""

from __future__ import annotations

from typing import Dict, List

import polars as pl

from .io_esviritu import TAX_RANKS
from .reassign import AssemblyResolution, apply_identity_thresholds

# Provenance columns appended to the info and assembly_summary tables.
PROVENANCE_COLUMNS = [
    "esviritu_species",
    "esviritu_subspecies",
    "postviritu_hit_accession",
    "postviritu_hit_taxid",
    "postviritu_bitscore",
    "postviritu_pct_identity",
    "postviritu_ambiguous",
    "postviritu_decision",
    "postviritu_genotypes",
    "postviritu_genotype_ambiguous",
]


# Optional EsViritu >= 1.3 per-Assembly columns carried into assembly_summary:
# ``adj_taxonomy`` (EsViritu adjusted its own call via consensus LCA) and
# ``consensus_ref_identity`` (consensus vs. EsViritu reference identity).
ASSEMBLY_CONSTANT_COLUMNS = ["adj_taxonomy", "consensus_ref_identity"]


def resolutions_to_df(resolutions: Dict[str, AssemblyResolution]) -> pl.DataFrame:
    """Build a per-Assembly DataFrame of new lineage + provenance + identity."""
    rows = []
    for asm, res in resolutions.items():
        row = {"Assembly": asm}
        for rank in TAX_RANKS:
            row[rank] = res.lineage.get(rank)
        row["esviritu_species"] = res.esviritu_species
        row["esviritu_subspecies"] = res.esviritu_subspecies
        row["postviritu_hit_accession"] = res.hit_accession
        row["postviritu_hit_taxid"] = res.hit_taxid
        row["postviritu_bitscore"] = res.bitscore
        row["postviritu_pct_identity"] = res.pct_identity
        row["postviritu_ambiguous"] = res.ambiguous
        row["postviritu_decision"] = res.decision
        row["postviritu_genotypes"] = res.genotypes
        row["postviritu_genotype_ambiguous"] = res.genotype_ambiguous
        rows.append(row)
    schema = {"Assembly": pl.Utf8}
    schema.update({r: pl.Utf8 for r in TAX_RANKS})
    schema.update(
        {
            "esviritu_species": pl.Utf8,
            "esviritu_subspecies": pl.Utf8,
            "postviritu_hit_accession": pl.Utf8,
            "postviritu_hit_taxid": pl.Utf8,
            "postviritu_bitscore": pl.Float64,
            "postviritu_pct_identity": pl.Float64,
            "postviritu_ambiguous": pl.Boolean,
            "postviritu_decision": pl.Utf8,
            "postviritu_genotypes": pl.Utf8,
            "postviritu_genotype_ambiguous": pl.Boolean,
        }
    )
    return pl.DataFrame(rows, schema=schema)


def rebuild_info(
    info_df: pl.DataFrame, resolutions: Dict[str, AssemblyResolution]
) -> pl.DataFrame:
    """Replace per-Assembly taxonomy in the info table and append provenance."""
    res_df = resolutions_to_df(resolutions)
    original_cols = info_df.columns

    stripped = info_df.drop([c for c in TAX_RANKS if c in info_df.columns])
    merged = stripped.join(res_df, on="Assembly", how="left")

    # Restore original column order, then append provenance columns.
    ordered = [c for c in original_cols if c in merged.columns]
    provenance = [c for c in PROVENANCE_COLUMNS if c in merged.columns]
    return merged.select(ordered + provenance)


def _filtered_reads(info_df: pl.DataFrame) -> int:
    if "filtered_reads_in_sample" in info_df.columns and info_df.height > 0:
        return int(info_df["filtered_reads_in_sample"][0])
    return 0


def _join_unique(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .fill_null("")
        .alias(col)
    )


def build_assembly_summary(new_info: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the rebuilt info table by Assembly (EsViritu format)."""
    filtered_reads = _filtered_reads(new_info)
    df = new_info.filter(pl.col("read_count") >= 1)

    group_keys = [
        "sample_ID",
        "filtered_reads_in_sample",
        "Assembly",
        "Asm_length",
        *TAX_RANKS,
    ]
    # EsViritu >= 1.3 columns, constant within an Assembly.
    optional = [c for c in ASSEMBLY_CONSTANT_COLUMNS if c in df.columns]
    agg = df.group_by(group_keys).agg(
        pl.col("read_count").sum().alias("read_count"),
        pl.col("covered_bases").sum().alias("covered_bases"),
        pl.col("avg_read_identity").mean().alias("avg_read_identity"),
        *[pl.col(c).first().alias(c) for c in optional],
        pl.col("Accession").cast(pl.Utf8).alias("Accession"),
        pl.col("Segment").cast(pl.Utf8).alias("Segment"),
    )
    agg = agg.with_columns(
        pl.col("Accession").list.join(","),
        pl.col("Segment").list.eval(pl.element().fill_null("")).list.join(","),
        (
            pl.col("read_count")
            / (pl.col("Asm_length") / 1000)
            / (filtered_reads / 1e6)
        ).alias("RPKMF"),
    )

    column_order = [
        "sample_ID",
        "filtered_reads_in_sample",
        "Assembly",
        "Asm_length",
        *TAX_RANKS,
        *[c for c in ["adj_taxonomy"] if c in optional],
        "read_count",
        "covered_bases",
        "avg_read_identity",
        *[c for c in ["consensus_ref_identity"] if c in optional],
        "Accession",
        "Segment",
        "RPKMF",
    ]
    return agg.select(column_order).sort(
        ["family", "genus", "species", "Assembly"]
    )


def thresholded_lineages(
    assembly_summary: pl.DataFrame,
    resolutions: Dict[str, AssemblyResolution],
    spthresh: float = 0.90,
    subspthresh: float = 0.95,
) -> Dict[str, Dict[str, str]]:
    """Return {Assembly: lineage} after species/subspecies identity thresholding.

    Thresholding uses the new consensus->hit identity (resolution.pct_identity).
    When no hit identity is available it falls back, like EsViritu >= 1.3, to
    EsViritu's consensus_ref_identity and then to the assembly's mean
    avg_read_identity.
    """
    out: Dict[str, Dict[str, str]] = {}
    for row in assembly_summary.iter_rows(named=True):
        res = resolutions.get(row["Assembly"])
        identity = None
        if res is not None and res.pct_identity is not None:
            identity = res.pct_identity
        elif row.get("consensus_ref_identity") is not None:
            identity = row["consensus_ref_identity"]
        elif row.get("avg_read_identity") is not None:
            identity = row["avg_read_identity"]
        lineage = {r: row[r] for r in TAX_RANKS}
        out[row["Assembly"]] = apply_identity_thresholds(
            lineage, identity, spthresh, subspthresh
        )
    return out


def build_tax_profile(
    new_info: pl.DataFrame,
    resolutions: Dict[str, AssemblyResolution],
    spthresh: float = 0.90,
    subspthresh: float = 0.95,
) -> pl.DataFrame:
    """Build the tax_profile, applying identity thresholding per assembly.

    See :func:`thresholded_lineages` for the identity used.
    """
    filtered_reads = _filtered_reads(new_info)
    assem = build_assembly_summary(new_info)

    # Apply per-assembly species/subspecies thresholding to the lineage.
    lineages = thresholded_lineages(assem, resolutions, spthresh, subspthresh)
    thresholded_rows: List[dict] = []
    for row in assem.iter_rows(named=True):
        new_row = dict(row)
        new_row.update(lineages[row["Assembly"]])
        thresholded_rows.append(new_row)

    assem_t = pl.DataFrame(thresholded_rows, schema=assem.schema)

    group_keys = ["sample_ID", "filtered_reads_in_sample", *TAX_RANKS]
    has_cons_id = "consensus_ref_identity" in assem_t.columns
    tax = assem_t.group_by(group_keys).agg(
        pl.col("read_count").sum().alias("read_count"),
        pl.col("RPKMF").sum().alias("RPKMF"),
        pl.col("avg_read_identity").mean().alias("avg_read_identity"),
        *(
            [pl.col("consensus_ref_identity").mean().alias("consensus_ref_identity")]
            if has_cons_id
            else []
        ),
        pl.col("Assembly").unique().alias("assembly_list"),
    )
    tax = tax.with_columns(pl.col("assembly_list").list.join(","))

    column_order = [
        "sample_ID",
        "filtered_reads_in_sample",
        *TAX_RANKS,
        "read_count",
        "RPKMF",
        "avg_read_identity",
        *(["consensus_ref_identity"] if has_cons_id else []),
        "assembly_list",
    ]
    return tax.select(column_order).sort(
        ["family", "genus", "species", "subspecies"]
    )
