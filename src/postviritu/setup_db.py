"""Build and index the mmseqs2 target database with NCBI taxonomy.

``postviritu setup-db`` runs ``mmseqs createdb`` then ``mmseqs createtaxdb`` so
the resulting database can emit taxids during search. A small JSON manifest is
written so ``postviritu run`` can locate the prepared database.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
from typing import Dict, Optional

MANIFEST_NAME = "postviritu_db.json"
TARGET_DB_NAME = "targetDB"


def manifest_path(db_dir: str) -> str:
    return os.path.join(db_dir, MANIFEST_NAME)


def load_manifest(db_dir: str) -> Dict:
    """Load the DB manifest; raise a helpful error if absent."""
    path = manifest_path(db_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No '{MANIFEST_NAME}' found in {db_dir}. Run `postviritu setup-db` "
            "first to build the database."
        )
    with open(path) as fh:
        return json.load(fh)


def target_db_path(db_dir: str) -> str:
    """Return the mmseqs target DB path recorded in the manifest."""
    return load_manifest(db_dir)["target_db"]


def setup_db(
    fasta: str,
    taxdump: str,
    acc2taxid: str,
    out: str,
    threads: int = 1,
    mmseqs_bin: str = "mmseqs",
    extra_createtaxdb_args: Optional[list] = None,
) -> str:
    """Build the mmseqs DB + taxonomy and write the manifest. Returns db_dir."""
    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(
            f"'{mmseqs_bin}' not found on PATH. Install mmseqs2 "
            "(e.g. `conda install -c bioconda mmseqs2`)."
        )
    for label, path in [("fasta", fasta), ("taxdump", taxdump), ("acc2taxid", acc2taxid)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"--{label} path does not exist: {path}")

    os.makedirs(out, exist_ok=True)
    target_db = os.path.join(out, TARGET_DB_NAME)
    tmp_dir = os.path.join(out, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1) sequence database
    subprocess.run(
        [mmseqs_bin, "createdb", fasta, target_db],
        check=True,
    )

    # 2) taxonomy database (enables taxid output during search)
    createtaxdb_cmd = [
        mmseqs_bin,
        "createtaxdb",
        target_db,
        tmp_dir,
        "--ncbi-tax-dump",
        taxdump,
        "--tax-mapping-file",
        acc2taxid,
        "--threads",
        str(threads),
        *(extra_createtaxdb_args or []),
    ]
    subprocess.run(createtaxdb_cmd, check=True)

    manifest = {
        "target_db": target_db,
        "fasta": os.path.abspath(fasta),
        "taxdump": os.path.abspath(taxdump),
        "acc2taxid": os.path.abspath(acc2taxid),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "mmseqs_bin": mmseqs_bin,
    }
    with open(manifest_path(out), "w") as fh:
        json.dump(manifest, fh, indent=2)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out
