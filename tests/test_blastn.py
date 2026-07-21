import stat

import pytest

from postviritu.aligner import (
    BlastnAligner,
    HIT_COLUMNS,
    filter_hits,
    parse_blastn,
)
from postviritu.io_esviritu import write_fasta


def _write_blastn_tsv(path, fields, lines):
    """Write BLASTN -outfmt 6 lines with the given column order."""
    path.write_text("\n".join("\t".join(str(v) for v in line) for line in lines) + "\n")


def test_parse_blastn_strips_consensus_and_scales(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    # query target taxid pident length qlen qcovs evalue bitscore qseq sseq
    lines = [
        ["accA_consensus", "tgt1", "100", "99.0", "1000", "1000", "95.0", "1e-50", "500.0", "ACGT", "ACGT"],
        ["accB_consensus", "tgt2", "200", "80.0", "900", "1000", "90.0", "1e-30", "300.0", "ACGA", "ACGT"],
    ]
    _write_blastn_tsv(out, fields, lines)
    df = parse_blastn(str(out), fields)
    assert df.columns == HIT_COLUMNS
    assert set(df["query"].to_list()) == {"accA", "accB"}
    assert df["taxid"].to_list() == ["100", "200"]

    filtered = filter_hits(df, min_identity=0.9, min_aln_fraction=0.5, max_evalue=1e-10)
    assert filtered.height == 1
    row = filtered.row(0, named=True)
    assert row["query"] == "accA"
    assert row["taxid"] == "100"
    assert row["pct_identity"] == 1.0
    assert abs(row["query_coverage"] - 0.95) < 1e-9


def test_parse_blastn_empty(tmp_path):
    out = tmp_path / "empty.tsv"
    out.write_text("")
    df = parse_blastn(str(out), BlastnAligner._FORMAT_FIELDS)
    assert df.height == 0
    assert df.columns == HIT_COLUMNS


def test_parse_blastn_nonN_metrics_exclude_query_N(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    lines = [
        ["accC_consensus", "tgt", "100", "50.0", "5", "20", "25.0", "1e-50", "500.0", "ACNGT", "ACAGT"],
    ]
    _write_blastn_tsv(out, fields, lines)
    df = parse_blastn(str(out), fields, query_nonN_len={"accC": 10})
    row = df.row(0, named=True)
    # 4 non-N aligned query residues all match -> identity 1.0
    assert row["pct_identity"] == 1.0
    # coverage = 4 aligned non-N / 10 total non-N = 0.4
    assert abs(row["query_coverage"] - 0.4) < 1e-9


def test_parse_blastn_multiple_staxids_takes_first(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    lines = [
        ["accD", "tgt", "100;200;300", "100.0", "4", "4", "100.0", "1e-50", "500.0", "ACGT", "ACGT"],
    ]
    _write_blastn_tsv(out, fields, lines)
    df = parse_blastn(str(out), fields)
    assert df.row(0, named=True)["taxid"] == "100"


@pytest.fixture
def fake_blastn(tmp_path):
    """Return a path to a fake blastn executable for testing batching/filtering."""
    script = tmp_path / "fake_blastn.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-query")
parser.add_argument("-out")
parser.add_argument("-db")
parser.add_argument("-outfmt")
parser.add_argument("-max_target_seqs")
parser.add_argument("-remote", action="store_true")
args, _ = parser.parse_known_args()

log = os.environ.get("POSTVIRITU_BLASTN_LOG")
if log:
    Path(log).parent.mkdir(parents=True, exist_ok=True)

records = []
with open(args.query) as fh:
    name = None
    for line in fh:
        line = line.rstrip("\\n")
        if line.startswith(">"):
            name = line[1:].split()[0]
            records.append(name)
        # sequence lines ignored

with open(args.out, "w") as fh:
    for name in records:
        fh.write(f"{name}\\ttgt_{name}\\t100\\t100.0\\t4\\t4\\t100.0\\t1e-50\\t500.0\\tACGT\\tACGT\\n")

if log:
    with open(log, "a") as fh:
        fh.write(f"{len(records)}\\n")
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_blastn_aligner_batches_and_awaits_each(tmp_path, fake_blastn, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("POSTVIRITU_BLASTN_LOG", str(log))

    query_fasta = tmp_path / "queries.fasta"
    seqs = {f"seq{i}": "ACGT" * 10 for i in range(5)}
    write_fasta(seqs, str(query_fasta))

    aligner = BlastnAligner(blastn_bin=fake_blastn, batch_size=2)
    hits = aligner.search(str(query_fasta), tmp_dir=str(tmp_path / "work"))

    assert hits.height == 5
    assert sorted(hits["query"].to_list()) == sorted(seqs.keys())
    assert log.read_text().strip().split("\n") == ["2", "2", "1"]


def test_blastn_aligner_filters_excluded_taxids(tmp_path, fake_blastn, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("POSTVIRITU_BLASTN_LOG", str(log))

    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq1": "ACGT" * 10}, str(query_fasta))

    script = tmp_path / "fake_blastn_multi.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-query")
parser.add_argument("-out")
parser.add_argument("-db")
parser.add_argument("-outfmt")
parser.add_argument("-max_target_seqs")
parser.add_argument("-remote", action="store_true")
args, _ = parser.parse_known_args()

with open(args.query) as fh:
    name = None
    for line in fh:
        if line.startswith(">"):
            name = line[1:].split()[0]

with open(args.out, "w") as fh:
    fh.write(f"{name}\\ttgtA\\t100\\t100.0\\t4\\t4\\t100.0\\t1e-50\\t500.0\\tACGT\\tACGT\\n")
    fh.write(f"{name}\\ttgtB\\t200\\t100.0\\t4\\t4\\t100.0\\t1e-50\\t500.0\\tACGT\\tACGT\\n")
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    aligner = BlastnAligner(blastn_bin=str(script), batch_size=3)
    hits = aligner.search(str(query_fasta), exclude_taxids=["100"], tmp_dir=str(tmp_path / "work"))

    assert hits.height == 1
    assert hits.row(0, named=True)["taxid"] == "200"


def test_blastn_subcommand_cli_args():
    from postviritu.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["blastn", "--input-dir", "in", "--outdir", "out", "--batch-size", "5"]
    )
    assert args.command == "blastn"
    assert args.db == "nt"
    assert args.batch_size == 5
    assert args.max_target_seqs == 300
    assert args.blastn_bin == "blastn"
