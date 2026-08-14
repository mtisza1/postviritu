"""Per-sample and batch orchestration for the ``postviritu run`` command."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl

from .aligner import Aligner, empty_hits, filter_hits
from .io_esviritu import (
    SamplePaths,
    TAX_RANKS,
    get_thresholds,
    iter_samples,
    load_params,
    parse_consensus_fasta,
    read_tsv,
    write_fasta,
    write_tsv,
)
from .outputs import build_assembly_summary, build_tax_profile, rebuild_info
from .reassign import AssemblyResolution, original_taxonomy, resolve_assemblies
from .taxonomy import TaxaFilter, Taxonomy


@dataclass
class RunConfig:
    mode: str = "scratch"
    min_identity: float = 0.9
    min_aln_fraction: float = 0.5
    max_evalue: float = 1e-10
    bitscore_tie_frac: float = 0.99
    threads: int = 1
    realign_ties: bool = False
    tmp_dir: Optional[str] = None
    keep_alignments: bool = False
    taxa_filter: Optional[TaxaFilter] = None


def _split_info_by_taxa_filter(
    info_df: pl.DataFrame, taxa_filter: Optional[TaxaFilter]
) -> tuple:
    """Return (included_info, excluded_info) based on the original taxonomy.

    Inclusion is decided at the Assembly level: if any Accession row for an
    assembly matches the filter, the whole assembly is processed. Rows with a
    null Assembly are always kept on the included side, so the filtered path
    hands them to the rest of the pipeline exactly as the unfiltered path does.
    """
    if taxa_filter is None or not taxa_filter.include:
        return info_df, info_df.filter(pl.lit(False))

    rows = list(info_df.iter_rows(named=True))
    included_assemblies = set()
    for row in rows:
        if row["Assembly"] is not None and taxa_filter.matches(row):
            included_assemblies.add(row["Assembly"])

    # `is_in` yields null for a null Assembly, which would drop the row from
    # both halves of the split; treat those rows as included.
    matched = pl.col("Assembly").is_in(list(included_assemblies)).fill_null(True)
    included = info_df.filter(matched)
    excluded = info_df.filter(~matched)
    return included, excluded


def _excluded_resolution(assembly: str, orig: Dict[str, Optional[str]]) -> AssemblyResolution:
    """Resolution for assemblies skipped by the taxonomy filter.

    The original EsViritu lineage is kept unchanged and provenance columns are
    left blank.
    """
    return AssemblyResolution(
        assembly=assembly,
        lineage={r: orig.get(r) for r in TAX_RANKS},
        decision="taxa_filtered",
        esviritu_species=orig.get("species"),
        esviritu_subspecies=orig.get("subspecies"),
    )


def run_sample(
    sample: SamplePaths,
    aligner: Aligner,
    taxonomy: Taxonomy,
    outdir: str,
    config: RunConfig,
) -> dict:
    """Process a single EsViritu sample. Returns the output file paths."""
    missing = sample.missing_required()
    if missing:
        raise FileNotFoundError(
            f"Sample '{sample.prefix}' is missing required files: {missing}"
        )

    info_df = read_tsv(sample.info)
    params = load_params(sample.params)
    spthresh, subspthresh = get_thresholds(params)
    consensus_seqs = parse_consensus_fasta(sample.consensus)

    included_info, excluded_info = _split_info_by_taxa_filter(
        info_df, config.taxa_filter
    )
    filter_active = config.taxa_filter is not None and bool(config.taxa_filter.include)
    if filter_active:
        included_count = included_info["Assembly"].n_unique()
        total_count = info_df["Assembly"].n_unique()
        print(
            f"[postviritu] {sample.prefix}: {included_count}/{total_count} "
            "assemblies pass taxa filter"
        )
        if included_count == 0:
            print(
                f"[postviritu] warning: {sample.prefix}: no assemblies matched "
                "the taxa filter"
            )

    included_accessions = set(included_info["Accession"].to_list())
    included_seqs = {
        acc: seq for acc, seq in consensus_seqs.items() if acc in included_accessions
    }

    if included_seqs:
        tmp_dir = config.tmp_dir or os.path.join(outdir, f"{sample.prefix}_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        query_fasta = sample.consensus
        if filter_active:
            query_fasta = os.path.join(tmp_dir, f"{sample.prefix}_query.fasta")
            write_fasta(included_seqs, query_fasta)
        query_nonN_len = {
            acc: sum(1 for base in seq if base not in "Nn")
            for acc, seq in included_seqs.items()
        }
        hits = aligner.search(
            query_fasta,
            threads=config.threads,
            tmp_dir=tmp_dir,
            keep=config.keep_alignments,
            result_name=f"{sample.prefix}_alignment.m8",
            query_nonN_len=query_nonN_len,
        )
        hits = filter_hits(
            hits,
            min_identity=config.min_identity,
            min_aln_fraction=config.min_aln_fraction,
            max_evalue=config.max_evalue,
        )
    else:
        hits = empty_hits()

    resolutions: Dict[str, AssemblyResolution] = {}
    if not included_info.is_empty():
        inc_res = resolve_assemblies(
            hits=hits,
            info_df=included_info,
            taxonomy=taxonomy,
            mode=config.mode,
            spthresh=spthresh,
            subspthresh=subspthresh,
            bitscore_tie_frac=config.bitscore_tie_frac,
            aligner=aligner,
            consensus_seqs=included_seqs,
            threads=config.threads,
            realign_ties=config.realign_ties,
        )
        resolutions.update(inc_res)

    if not excluded_info.is_empty():
        for asm, orig in original_taxonomy(excluded_info).items():
            resolutions[asm] = _excluded_resolution(asm, orig)

    new_info = rebuild_info(info_df, resolutions)
    assembly_summary = build_assembly_summary(new_info)
    tax_profile = build_tax_profile(new_info, resolutions, spthresh, subspthresh)

    os.makedirs(outdir, exist_ok=True)
    out = SamplePaths(prefix=sample.prefix, directory=outdir)
    write_tsv(new_info, out.info)
    write_tsv(assembly_summary, out.assembly_summary)
    write_tsv(tax_profile, out.tax_profile)

    # Pass-through files (unchanged) when present.
    for src in [sample.coverage_windows, sample.params, sample.readstats]:
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(outdir, os.path.basename(src)))

    return {
        "info": out.info,
        "assembly_summary": out.assembly_summary,
        "tax_profile": out.tax_profile,
    }


def run_batch(
    input_dir: str,
    aligner: Aligner,
    taxonomy: Taxonomy,
    outdir: str,
    config: RunConfig,
    sample_id: Optional[str] = None,
) -> List[str]:
    """Process all samples (or a single sample_id) in ``input_dir``."""
    processed: List[str] = []
    samples = list(iter_samples(input_dir, sample_id=sample_id))
    if not samples:
        print(f"No EsViritu samples found in {input_dir}")
        return processed
    for sample in samples:
        print(f"[postviritu] processing sample: {sample.prefix}")
        try:
            run_sample(sample, aligner, taxonomy, outdir, config)
            processed.append(sample.prefix)
        except FileNotFoundError as e:
            print(f"[postviritu] skipping {sample.prefix}: {e}")
    return processed
