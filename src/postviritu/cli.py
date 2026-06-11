"""Command-line interface for postviritu.

Subcommands:
  setup-db   build the mmseqs2 target DB + taxonomy (run once)
  run        re-align EsViritu consensus genomes and rewrite outputs
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__


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
        "--taxdump",
        default=None,
        help="NCBI taxdump dir for pytaxonkit (default: from DB manifest, "
        "else taxonkit's ~/.taxonkit).",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


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
    print(f"[postviritu] database ready at: {args.out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .aligner import Mmseqs2Aligner
    from .run import RunConfig, run_batch
    from .setup_db import load_manifest, target_db_path
    from .taxonomy import Taxonomy

    manifest = load_manifest(args.db)
    target_db = target_db_path(args.db)
    # pytaxonkit falls back to ~/.taxonkit when data_dir is None.
    taxdump = args.taxdump or manifest.get("taxdump")

    aligner = Mmseqs2Aligner(target_db=target_db, mmseqs_bin=args.mmseqs_bin)
    taxonomy = Taxonomy(data_dir=taxdump, threads=args.threads)
    config = RunConfig(
        mode=args.mode,
        min_identity=args.min_identity,
        min_aln_fraction=args.min_aln_fraction,
        max_evalue=args.max_evalue,
        bitscore_tie_frac=args.bitscore_tie_frac,
        threads=args.threads,
        realign_ties=args.realign_ties,
    )
    processed = run_batch(
        input_dir=args.input_dir,
        aligner=aligner,
        taxonomy=taxonomy,
        outdir=args.outdir,
        config=config,
        sample_id=args.sample_id,
    )
    print(f"[postviritu] processed {len(processed)} sample(s): {', '.join(processed)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
