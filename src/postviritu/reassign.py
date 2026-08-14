"""Per-assembly taxonomy reassignment from database hits.

Hits are compiled per EsViritu ``Assembly`` (all member Accessions pooled),
the best-scoring taxon (or LCA of tied taxa) is resolved, and an EsViritu-style
8-rank lineage is produced (with species/subspecies ``unclassified_``
thresholding driven by the new consensus->hit identity).

Two modes:
- ``scratch``  : re-derive every assembly's taxonomy from the hits; no-hit
  assemblies become fully unclassified.
- ``disagree`` : keep EsViritu's call unless the top hit clearly disagrees
  (override) or best hits tie across taxa (LCA + ambiguity flag), with an
  optional second-round re-alignment excluding the round-1 taxid.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import polars as pl

from .io_esviritu import RANK_PREFIXES, TAX_RANKS, write_fasta
from .taxonomy import Taxonomy, unclassified_lineage

MODE_SCRATCH = "scratch"
MODE_DISAGREE = "disagree"


@dataclass
class AssemblyResolution:
    """Resolved taxonomy + provenance for a single Assembly."""

    assembly: str
    lineage: Dict[str, str]
    hit_accession: Optional[str] = None
    hit_taxid: Optional[str] = None
    bitscore: Optional[float] = None
    pct_identity: Optional[float] = None
    ambiguous: bool = False
    decision: str = ""
    esviritu_species: Optional[str] = None
    esviritu_subspecies: Optional[str] = None
    tied_taxids: List[str] = field(default_factory=list)


def apply_identity_thresholds(
    lineage: Dict[str, str],
    identity: Optional[float],
    spthresh: float,
    subspthresh: float,
) -> Dict[str, str]:
    """Replicate EsViritu's species/subspecies ``unclassified_`` thresholding.

    If ``identity`` < ``spthresh`` the species becomes
    ``s__unclassified_<genus>``; if ``identity`` < ``subspthresh`` the
    subspecies becomes ``t__unclassified_<species>``. ``None`` identity skips.
    """
    out = dict(lineage)
    if identity is None:
        return out

    if identity < spthresh:
        genus_core = (
            out.get("genus", "g__")
            .removeprefix(RANK_PREFIXES["genus"])
            .removeprefix("unclassified_")
        )
        out["species"] = RANK_PREFIXES["species"] + "unclassified_" + genus_core

    if identity < subspthresh:
        species_core = (
            out.get("species", "s__")
            .removeprefix(RANK_PREFIXES["species"])
            .removeprefix("unclassified_")
        )
        out["subspecies"] = (
            RANK_PREFIXES["subspecies"] + "unclassified_" + species_core
        )
    return out


def _best_hits(assembly_hits: pl.DataFrame, tie_frac: float) -> pl.DataFrame:
    """Return hits whose bitscore is within ``tie_frac`` of the max bitscore."""
    max_bits = assembly_hits["bitscore"].max()
    return assembly_hits.filter(pl.col("bitscore") >= tie_frac * max_bits)


def _accession_to_assembly(info_df: pl.DataFrame) -> Dict[str, str]:
    sub = info_df.select(["Accession", "Assembly"]).unique()
    return dict(zip(sub["Accession"].to_list(), sub["Assembly"].to_list()))


def original_taxonomy(info_df: pl.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return {Assembly: original 8-rank lineage} from the EsViritu info table."""
    cols = ["Assembly"] + TAX_RANKS
    sub = info_df.select([c for c in cols if c in info_df.columns]).unique(
        subset=["Assembly"], keep="first"
    )
    out: Dict[str, Dict[str, str]] = {}
    for row in sub.iter_rows(named=True):
        out[row["Assembly"]] = {r: row.get(r) for r in TAX_RANKS}
    return out


def resolve_assemblies(
    hits: pl.DataFrame,
    info_df: pl.DataFrame,
    taxonomy: Taxonomy,
    mode: str = MODE_SCRATCH,
    spthresh: float = 0.90,
    subspthresh: float = 0.95,
    bitscore_tie_frac: float = 0.99,
    aligner=None,
    consensus_seqs: Optional[Dict[str, str]] = None,
    threads: int = 1,
    realign_ties: bool = False,
) -> Dict[str, AssemblyResolution]:
    """Resolve taxonomy for every Assembly present in ``info_df``."""
    acc2asm = _accession_to_assembly(info_df)
    orig_tax = original_taxonomy(info_df)
    assemblies = list(orig_tax.keys())

    # Attach Assembly to each hit via its query Accession.
    if not hits.is_empty():
        hits = hits.with_columns(
            pl.col("query")
            .replace_strict(acc2asm, default=None)
            .alias("Assembly")
        )

    resolutions: Dict[str, AssemblyResolution] = {}
    for assembly in assemblies:
        orig = orig_tax.get(assembly, {})
        orig_species = orig.get("species")
        orig_subspecies = orig.get("subspecies")

        asm_hits = (
            hits.filter(pl.col("Assembly") == assembly)
            if not hits.is_empty()
            else hits
        )

        if asm_hits.is_empty():
            resolutions[assembly] = _resolve_no_hit(
                assembly, mode, orig, orig_species, orig_subspecies
            )
            continue

        resolutions[assembly] = _resolve_with_hits(
            assembly=assembly,
            asm_hits=asm_hits,
            taxonomy=taxonomy,
            mode=mode,
            spthresh=spthresh,
            subspthresh=subspthresh,
            tie_frac=bitscore_tie_frac,
            orig=orig,
            orig_species=orig_species,
            orig_subspecies=orig_subspecies,
            aligner=aligner,
            consensus_seqs=consensus_seqs,
            threads=threads,
            realign_ties=realign_ties,
        )
    return resolutions


def _resolve_no_hit(assembly, mode, orig, orig_species, orig_subspecies):
    if mode == MODE_DISAGREE:
        # Keep EsViritu's original call when there is nothing to disagree with.
        lineage = {r: orig.get(r) for r in TAX_RANKS}
        return AssemblyResolution(
            assembly=assembly,
            lineage=lineage,
            decision="no_hit_kept_original",
            esviritu_species=orig_species,
            esviritu_subspecies=orig_subspecies,
        )
    # scratch mode: no acceptable hit -> fully unclassified
    return AssemblyResolution(
        assembly=assembly,
        lineage=unclassified_lineage(),
        decision="no_hit_unclassified",
        esviritu_species=orig_species,
        esviritu_subspecies=orig_subspecies,
    )


def _resolve_with_hits(
    assembly,
    asm_hits,
    taxonomy,
    mode,
    spthresh,
    subspthresh,
    tie_frac,
    orig,
    orig_species,
    orig_subspecies,
    aligner,
    consensus_seqs,
    threads,
    realign_ties,
):
    best = _best_hits(asm_hits, tie_frac)
    best_sorted = best.sort("bitscore", descending=True)
    top = best_sorted.row(0, named=True)
    tied_taxids = [t for t in best["taxid"].unique().to_list() if t]
    best_identity = float(best["pct_identity"].max())

    ambiguous = False
    decision = ""
    assigned_taxid: Optional[str] = top["taxid"]

    if len(tied_taxids) >= 2:
        # Tie across multiple taxa -> LCA.
        lca_taxid = taxonomy.lca(tied_taxids)
        assigned_taxid = lca_taxid or top["taxid"]
        ambiguous = True
        decision = "lca_ambiguous"
    else:
        # Single top taxon.
        if mode == MODE_DISAGREE and realign_ties and aligner and consensus_seqs:
            second = _second_round_tie_check(
                assembly=assembly,
                asm_hits=asm_hits,
                round1_taxid=top["taxid"],
                round1_bits=float(top["bitscore"]),
                aligner=aligner,
                consensus_seqs=consensus_seqs,
                tie_frac=tie_frac,
                threads=threads,
            )
            if second is not None:
                combined = [top["taxid"], second]
                lca_taxid = taxonomy.lca(combined)
                assigned_taxid = lca_taxid or top["taxid"]
                ambiguous = True
                decision = "lca_ambiguous_realigned"

        if not decision:
            decision = "assigned" if mode == MODE_SCRATCH else "single_taxon"

    lineage = taxonomy.esviritu_lineage(assigned_taxid) if assigned_taxid else unclassified_lineage()

    # Decide whether to override EsViritu in disagree mode.
    if mode == MODE_DISAGREE and not ambiguous:
        new_species = lineage.get("species")
        if new_species and orig_species and new_species != orig_species:
            decision = "overridden"
        else:
            # Agrees (or can't tell) -> keep EsViritu's lineage unchanged.
            lineage = {r: orig.get(r) for r in TAX_RANKS}
            decision = "kept_original"

    # NOTE: ``lineage`` here is the *raw* lineage used for info/assembly_summary.
    # Species/subspecies identity thresholding is applied later (tax_profile).
    return AssemblyResolution(
        assembly=assembly,
        lineage=lineage,
        hit_accession=top["target"],
        hit_taxid=top["taxid"],
        bitscore=float(top["bitscore"]),
        pct_identity=best_identity,
        ambiguous=ambiguous,
        decision=decision,
        esviritu_species=orig_species,
        esviritu_subspecies=orig_subspecies,
        tied_taxids=tied_taxids,
    )


def _second_round_tie_check(
    assembly,
    asm_hits,
    round1_taxid,
    round1_bits,
    aligner,
    consensus_seqs,
    tie_frac,
    threads,
) -> Optional[str]:
    """Re-align this assembly's consensus excluding round-1 taxid.

    Returns the taxid of a second-round hit whose bitscore ties round 1
    (>= tie_frac * round1_bits), else None.
    """
    accessions = asm_hits["query"].unique().to_list()
    subset = {a: consensus_seqs[a] for a in accessions if a in consensus_seqs}
    if not subset:
        return None
    tmp_dir = tempfile.mkdtemp(prefix="postviritu_realign_")
    try:
        fasta_path = os.path.join(tmp_dir, "subset.fasta")
        write_fasta(subset, fasta_path)
        query_nonN_len = {
            acc: sum(1 for base in seq if base not in "Nn")
            for acc, seq in subset.items()
        }
        round2 = aligner.search(
            fasta_path,
            threads=threads,
            exclude_taxids=[round1_taxid],
            query_nonN_len=query_nonN_len,
        )
        if round2.is_empty():
            return None
        r2_top = round2.sort("bitscore", descending=True).row(0, named=True)
        if float(r2_top["bitscore"]) >= tie_frac * round1_bits:
            return r2_top["taxid"]
        return None
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
