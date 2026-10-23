import random
import shutil
import subprocess

import pytest

from postviritu.aligner import (
    BLASTN_TABULAR_FIELDS,
    HIT_COLUMNS,
    LocalBlastnAligner,
    filter_hits,
    parse_blastn_tabular,
)
from postviritu.io_esviritu import write_fasta


def _row(query, target, taxid, qseq, sseq, qstart, qend, sstart, send, qlen,
         evalue=1e-50, bits=500.0):
    pident = 100 * sum(q == s for q, s in zip(qseq, sseq)) / len(qseq)
    values = [query, target, taxid, pident, len(qseq), qlen, evalue, bits,
              qseq, sseq, qstart, qend, sstart, send]
    return "\t".join(str(v) for v in values)


def _write_tsv(path, rows):
    path.write_text("".join(row + "\n" for row in rows))


def test_parse_blastn_tabular_scales_and_strips(tmp_path):
    out = tmp_path / "result.m8"
    _write_tsv(
        out,
        [_row("seq1_consensus", "NC_000001.1", "100", "ACGT" * 25, "ACGT" * 25,
              1, 100, 1, 100, qlen=200)],
    )
    hits = parse_blastn_tabular(str(out))

    assert hits.columns == HIT_COLUMNS
    row = hits.row(0, named=True)
    assert row["query"] == "seq1"
    assert row["target"] == "NC_000001.1"
    assert row["taxid"] == "100"
    assert row["pct_identity"] == pytest.approx(1.0)
    assert row["query_coverage"] == pytest.approx(0.5)


def test_parse_blastn_tabular_nonN_metrics(tmp_path):
    out = tmp_path / "result.m8"
    qseq = "ACGT" * 20 + "N" * 20
    sseq = "ACGT" * 20 + "ACGT" * 5
    _write_tsv(out, [_row("seq1", "tgt", "100", qseq, sseq, 1, 100, 1, 100, qlen=100)])

    hits = parse_blastn_tabular(str(out), {"seq1": 80})

    row = hits.row(0, named=True)
    assert row["pct_identity"] == pytest.approx(1.0)
    assert row["query_coverage"] == pytest.approx(1.0)


def test_parse_blastn_tabular_missing_taxid_is_dropped(tmp_path, caplog):
    out = tmp_path / "result.m8"
    _write_tsv(
        out,
        [
            _row("seq1", "tgtA", "N/A", "ACGT" * 25, "ACGT" * 25, 1, 100, 1, 100, qlen=100),
            _row("seq1", "tgtB", "0", "ACGT" * 25, "ACGT" * 25, 1, 100, 1, 100, qlen=100),
        ],
    )
    with caplog.at_level("WARNING"):
        hits = parse_blastn_tabular(str(out))

    assert hits["taxid"].null_count() == hits.height == 2
    assert filter_hits(hits).is_empty()
    assert "taxid_map" in caplog.text


def test_parse_blastn_tabular_empty(tmp_path):
    out = tmp_path / "result.m8"
    out.write_text("")
    assert parse_blastn_tabular(str(out)).columns == HIT_COLUMNS
    assert parse_blastn_tabular(str(tmp_path / "missing.m8")).is_empty()


def _fake_blastn(rows, calls):
    def fake_run(cmd, check):
        calls.append(cmd)
        out = cmd[cmd.index("-out") + 1]
        with open(out, "w") as handle:
            handle.write("".join(row + "\n" for row in rows))
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def _identical_rows():
    return [
        _row("seq1", "tgtA", "100", "ACGT" * 10, "ACGT" * 10, 1, 40, 1, 40, qlen=40),
        _row("seq1", "tgtB", "200", "ACGT" * 10, "ACGT" * 10, 1, 40, 1, 40, qlen=40),
    ]


def test_local_blastn_command_and_exclusion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("subprocess.run", _fake_blastn(_identical_rows(), calls))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq1": "ACGT" * 10}, str(query_fasta))

    aligner = LocalBlastnAligner(db="/dbs/core_nt", max_target_seqs=50)
    hits = aligner.search(
        str(query_fasta), threads=4, exclude_taxids=["100"], tmp_dir=str(tmp_path / "w")
    )

    (cmd,) = calls
    assert cmd[0] == "blastn"
    assert cmd[cmd.index("-db") + 1] == "/dbs/core_nt"
    assert cmd[cmd.index("-task") + 1] == "megablast"
    assert cmd[cmd.index("-num_threads") + 1] == "4"
    assert cmd[cmd.index("-max_target_seqs") + 1] == "50"
    assert cmd[cmd.index("-outfmt") + 1] == "6 " + " ".join(BLASTN_TABULAR_FIELDS)
    assert cmd[cmd.index("-negative_taxids") + 1] == "100"
    assert hits["taxid"].to_list() == ["200"]


def test_local_blastn_keep_controls_result_file(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("subprocess.run", _fake_blastn(_identical_rows(), []))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq1": "ACGT" * 10}, str(query_fasta))
    aligner = LocalBlastnAligner(db="db")

    kept = tmp_path / "kept"
    aligner.search(str(query_fasta), tmp_dir=str(kept), keep=True, result_name="s.m8")
    assert (kept / "s.m8").is_file()

    dropped = tmp_path / "dropped"
    hits = aligner.search(str(query_fasta), tmp_dir=str(dropped))
    assert not dropped.exists()
    assert hits.height == 2


def test_local_blastn_missing_binary(tmp_path):
    aligner = LocalBlastnAligner(db="db", blastn_bin="no-such-blastn")
    with pytest.raises(RuntimeError, match="BLAST\\+"):
        aligner.search(str(tmp_path / "q.fasta"))


def test_blastn_subcommand_local_args():
    from postviritu.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["blastn", "--input-dir", "in", "--outdir", "out"])
    assert args.local is False
    assert args.task == "megablast"

    args = parser.parse_args(
        ["blastn", "--input-dir", "in", "--outdir", "out", "--local",
         "--db", "/dbs/core_nt", "--task", "blastn", "--blastn-bin", "/opt/blastn"]
    )
    assert args.local is True
    assert args.db == "/dbs/core_nt"
    assert args.task == "blastn"
    assert args.blastn_bin == "/opt/blastn"


@pytest.mark.skipif(
    shutil.which("blastn") is None or shutil.which("makeblastdb") is None,
    reason="BLAST+ not installed",
)
def test_local_blastn_end_to_end(tmp_path):
    rng = random.Random(0)
    genome_a = "".join(rng.choice("ACGT") for _ in range(2000))
    genome_b = "".join(rng.choice("ACGT") for _ in range(2000))
    refs = tmp_path / "refs.fasta"
    write_fasta({"REF_A.1": genome_a, "REF_B.1": genome_b}, str(refs))
    taxid_map = tmp_path / "taxid_map.txt"
    taxid_map.write_text("REF_A.1\t111\nREF_B.1\t222\n")
    db = tmp_path / "db" / "refs"
    subprocess.run(
        ["makeblastdb", "-in", str(refs), "-dbtype", "nucl", "-parse_seqids",
         "-taxid_map", str(taxid_map), "-out", str(db)],
        check=True,
        capture_output=True,
    )

    # Query: a stretch of genome A with a run of Ns and a few mismatches.
    query = list(genome_a[100:1100])
    query[400:420] = "N" * 20
    query = "".join(query)
    queries = tmp_path / "queries.fasta"
    write_fasta({"acc1_consensus": query, "acc2_consensus": genome_b[500:1500]}, str(queries))

    aligner = LocalBlastnAligner(db=str(db))
    hits = aligner.search(
        str(queries), tmp_dir=str(tmp_path / "w"), query_nonN_len={"acc1": 980, "acc2": 1000}
    )
    by_query = {row["query"]: row for row in hits.iter_rows(named=True)}
    assert by_query["acc1"]["target"] == "REF_A.1"
    assert by_query["acc1"]["taxid"] == "111"
    assert by_query["acc1"]["pct_identity"] == pytest.approx(1.0)
    assert by_query["acc1"]["query_coverage"] == pytest.approx(1.0)
    assert by_query["acc2"]["taxid"] == "222"

    excluded = aligner.search(str(queries), exclude_taxids=["222"])
    assert set(excluded["taxid"].to_list()) == {"111"}
