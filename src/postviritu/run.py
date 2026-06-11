"""Per-sample and batch orchestration for the ``postviritu run`` command."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from .aligner import Aligner, filter_hits
from .io_esviritu import (
    SamplePaths,
    get_thresholds,
    iter_samples,
    load_params,
    parse_consensus_fasta,
    read_tsv,
    write_tsv,
)
from .outputs import build_assembly_summary, build_tax_profile, rebuild_info
from .reassign import resolve_assemblies
from .taxonomy import Taxonomy


@dataclass
class RunConfig:
    mode: str = "scratch"
    min_identity: float = 0.9
    min_aln_fraction: float = 0.5
    max_evalue: float = 1e-10
    bitscore_tie_frac: float = 0.99
    threads: int = 1
    realign_ties: bool = False


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

    hits = aligner.search(sample.consensus, threads=config.threads)
    hits = filter_hits(
        hits,
        min_identity=config.min_identity,
        min_aln_fraction=config.min_aln_fraction,
        max_evalue=config.max_evalue,
    )

    resolutions = resolve_assemblies(
        hits=hits,
        info_df=info_df,
        taxonomy=taxonomy,
        mode=config.mode,
        spthresh=spthresh,
        subspthresh=subspthresh,
        bitscore_tie_frac=config.bitscore_tie_frac,
        aligner=aligner,
        consensus_seqs=consensus_seqs,
        threads=config.threads,
        realign_ties=config.realign_ties,
    )

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
