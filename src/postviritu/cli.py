"""Command-line interface for postviritu.

Subcommands:
  setup-db   build the mmseqs2 target DB + taxonomy (run once)
  run        re-align EsViritu consensus genomes and rewrite outputs
  blastn     re-align using Biopython's remote NCBI BLAST against nt
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from . import __version__

logger = logging.getLogger("postviritu")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postviritu",
        description=(
            "Post-process EsViritu output: re-align consensus genomes against a "
            "large NCBI DB with mmseqs2, re-derive taxonomy, and rewrite "
            "EsViritu-format tables."
        ),
    )
    parser.add_argument("--version", action="version", version=f"postviritu {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug-level logging (default: timestamped progress at info level).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-db
    p_db = sub.add_parser(
        "setup-db", help="Build the mmseqs2 target DB with taxonomy (run once)."
    )
    p_db.add_argument("--fasta", required=True, help="Reference sequence FASTA (e.g. core_nt).")
    p_db.add_argument("--taxdump", required=True, help="NCBI taxdump directory (nodes/names.dmp).")
    p_db.add_argument(
        "--acc2taxid", required=True, help="NCBI accession2taxid mapping file."
    )
    p_db.add_argument("--out", required=True, help="Output DB directory.")
    p_db.add_argument("--threads", type=int, default=1)
    p_db.add_argument("--mmseqs-bin", default="mmseqs")
    p_db.set_defaults(func=_cmd_setup_db)

    # run
    p_run = sub.add_parser(
        "run", help="Re-align consensus genomes and rewrite EsViritu outputs."
    )
    p_run.add_argument("--input-dir", required=True, help="Directory of EsViritu outputs.")
    p_run.add_argument("--db", required=True, help="Prepared DB directory from setup-db.")
    p_run.add_argument("--outdir", required=True, help="Output directory.")
    p_run.add_argument(
        "--sample_id", default=None, help="Process only this sample prefix (default: all)."
    )
    p_run.add_argument(
        "--mode",
        choices=["scratch", "disagree"],
        default="scratch",
        help="Reassignment mode (default: scratch).",
    )
    p_run.add_argument("--min-identity", type=float, default=0.9)
    p_run.add_argument("--min-aln-fraction", type=float, default=0.5)
    p_run.add_argument("--max-evalue", type=float, default=1e-10)
    p_run.add_argument("--bitscore-tie-frac", type=float, default=0.99)
    p_run.add_argument(
        "--realign-ties",
        action="store_true",
        help="(disagree mode) second-round re-align excluding the round-1 taxid.",
    )
    p_run.add_argument("--threads", type=int, default=1)
    p_run.add_argument("--mmseqs-bin", default="mmseqs")
    p_run.add_argument(
        "--tmp-dir",
        default=None,
        help="Working directory for alignment intermediates "
        "(default: {outdir}/{prefix}_tmp).",
    )
    p_run.add_argument(
        "--keep-alignments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the tabular alignment output files for inspection "
        "(default: on; use --no-keep-alignments to delete them after each sample).",
    )
    p_run.add_argument(
        "--taxdump",
        default=None,
        help="NCBI taxdump dir for pytaxonkit (default: from DB manifest, "
        "else taxonkit's ~/.taxonkit).",
    )
    p_run.add_argument(
        "--taxa-filter",
        default=None,
        help="YAML file of taxa to include (default: process all).",
    )
    p_run.add_argument(
        "--vvsearch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Supplement viral subspecies with NCBI Virus Variation genotypes "
        "(default: on). Use --no-vvsearch for a fully offline, deterministic "
        "run.",
    )
    p_run.add_argument(
        "--vvsearch-email",
        default=None,
        help="Contact email sent with vvsearch2 requests so NCBI can reach you "
        "about a misbehaving client (recommended for large batches).",
    )
    p_run.add_argument(
        "--vvsearch-timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds for vvsearch2 (default: 10).",
    )
    p_run.set_defaults(func=_cmd_run)

    # blastn
    p_blastn = sub.add_parser(
        "blastn",
        help="Re-align consensus genomes using Biopython's remote NCBI BLAST against nt.",
    )
    p_blastn.add_argument(
        "--input-dir", required=True, help="Directory of EsViritu outputs."
    )
    p_blastn.add_argument(
        "--outdir", required=True, help="Output directory."
    )
    p_blastn.add_argument(
        "--sample_id", default=None, help="Process only this sample prefix (default: all)."
    )
    p_blastn.add_argument(
        "--mode",
        choices=["scratch", "disagree"],
        default="scratch",
        help="Reassignment mode (default: scratch).",
    )
    p_blastn.add_argument("--min-identity", type=float, default=0.9)
    p_blastn.add_argument("--min-aln-fraction", type=float, default=0.5)
    p_blastn.add_argument("--max-evalue", type=float, default=1e-10)
    p_blastn.add_argument("--bitscore-tie-frac", type=float, default=0.99)
    p_blastn.add_argument(
        "--realign-ties",
        action="store_true",
        help="(disagree mode) second-round re-align excluding the round-1 taxid.",
    )
    p_blastn.add_argument("--threads", type=int, default=1)
    p_blastn.add_argument(
        "--db", default="nt", help="NCBI database name (default: nt)."
    )
    p_blastn.add_argument(
        "--taxdump",
        default=None,
        help="NCBI taxdump dir for pytaxonkit (default: taxonkit's ~/.taxonkit).",
    )
    p_blastn.add_argument(
        "--taxa-filter",
        default=None,
        help="YAML file of taxa to include (default: process all).",
    )
    p_blastn.add_argument(
        "--tmp-dir",
        default=None,
        help="Working directory for alignment intermediates "
        "(default: {outdir}/{prefix}_tmp).",
    )
    p_blastn.add_argument(
        "--keep-alignments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the tabular alignment output files for inspection "
        "(default: on; use --no-keep-alignments to delete them after each sample).",
    )
    p_blastn.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of query sequences per remote BLASTN call (default: 3).",
    )
    p_blastn.add_argument(
        "--max-target-seqs",
        type=int,
        default=300,
        help="Maximum target sequences reported per query (default: 300).",
    )
    p_blastn.add_argument(
        "--blast-retries",
        type=int,
        default=3,
        help="Times to re-submit a remote BLASTN batch that is still running "
        "after 10 minutes (default: 3).",
    )
    p_blastn.add_argument(
        "--vvsearch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Supplement viral subspecies with NCBI Virus Variation genotypes "
        "(default: on). Use --no-vvsearch for a fully offline, deterministic "
        "run.",
    )
    p_blastn.add_argument(
        "--vvsearch-email",
        default=None,
        help="Contact email sent with vvsearch2 requests so NCBI can reach you "
        "about a misbehaving client (recommended for large batches).",
    )
    p_blastn.add_argument(
        "--vvsearch-timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds for vvsearch2 (default: 10).",
    )
    p_blastn.set_defaults(func=_cmd_blastn)

    return parser


def _vvsearch_config(args: argparse.Namespace):
    """Build the Virus Variation client policy from parsed CLI arguments."""
    from .taxonomy import VVSearchConfig

    return VVSearchConfig(
        enabled=args.vvsearch,
        email=args.vvsearch_email,
        timeout=args.vvsearch_timeout,
    )


def _cmd_setup_db(args: argparse.Namespace) -> int:
    from .setup_db import setup_db

    setup_db(
        fasta=args.fasta,
        taxdump=args.taxdump,
        acc2taxid=args.acc2taxid,
        out=args.out,
        threads=args.threads,
        mmseqs_bin=args.mmseqs_bin,
    )
    logger.info("database ready at: %s", args.out)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .aligner import Mmseqs2Aligner
    from .run import RunConfig, run_batch
    from .setup_db import load_manifest, target_db_path
    from .taxonomy import TaxaFilter, Taxonomy

    manifest = load_manifest(args.db)
    target_db = target_db_path(args.db)
    # pytaxonkit falls back to ~/.taxonkit when data_dir is None.
    taxdump = args.taxdump or manifest.get("taxdump")

    aligner = Mmseqs2Aligner(target_db=target_db, mmseqs_bin=args.mmseqs_bin)
    taxonomy = Taxonomy(
        data_dir=taxdump, threads=args.threads, vvsearch=_vvsearch_config(args)
    )
    taxa_filter = TaxaFilter.from_yaml(args.taxa_filter) if args.taxa_filter else None
    config = RunConfig(
        mode=args.mode,
        min_identity=args.min_identity,
        min_aln_fraction=args.min_aln_fraction,
        max_evalue=args.max_evalue,
        bitscore_tie_frac=args.bitscore_tie_frac,
        threads=args.threads,
        realign_ties=args.realign_ties,
        tmp_dir=args.tmp_dir,
        keep_alignments=args.keep_alignments,
        taxa_filter=taxa_filter,
    )
    processed = run_batch(
        input_dir=args.input_dir,
        aligner=aligner,
        taxonomy=taxonomy,
        outdir=args.outdir,
        config=config,
        sample_id=args.sample_id,
    )
    logger.info("processed %d sample(s): %s", len(processed), ", ".join(processed))
    return 0


def _cmd_blastn(args: argparse.Namespace) -> int:
    from .aligner import BlastnAligner
    from .run import RunConfig, run_batch
    from .taxonomy import TaxaFilter, Taxonomy

    aligner = BlastnAligner(
        db=args.db,
        max_target_seqs=args.max_target_seqs,
        batch_size=args.batch_size,
        max_retries=args.blast_retries,
    )
    taxonomy = Taxonomy(
        data_dir=args.taxdump, threads=args.threads, vvsearch=_vvsearch_config(args)
    )
    taxa_filter = TaxaFilter.from_yaml(args.taxa_filter) if args.taxa_filter else None
    config = RunConfig(
        mode=args.mode,
        min_identity=args.min_identity,
        min_aln_fraction=args.min_aln_fraction,
        max_evalue=args.max_evalue,
        bitscore_tie_frac=args.bitscore_tie_frac,
        threads=args.threads,
        realign_ties=args.realign_ties,
        tmp_dir=args.tmp_dir,
        keep_alignments=args.keep_alignments,
        taxa_filter=taxa_filter,
    )
    processed = run_batch(
        input_dir=args.input_dir,
        aligner=aligner,
        taxonomy=taxonomy,
        outdir=args.outdir,
        config=config,
        sample_id=args.sample_id,
    )
    logger.info("processed %d sample(s): %s", len(processed), ", ".join(processed))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [postviritu] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Keep third-party chatter (HTTP connection pools) out of the progress log
    # unless explicitly debugging.
    if not args.verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
