from io import StringIO

from postviritu.aligner import (
    BlastnAligner,
    HIT_COLUMNS,
    filter_hits,
    parse_blastn,
)
from postviritu.io_esviritu import write_fasta


def _write_blastn_tsv(path, fields, lines):
    """Write BLASTN tabular lines with the given column order."""
    path.write_text("\n".join("\t".join(str(v) for v in line) for line in lines) + "\n")


def test_parse_blastn_strips_consensus_and_scales(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    lines = [
        ["accA_consensus", "tgt1", "100", "99.0", "1000", "1000", "95.0", "1e-50", "500.0", "ACGT", "ACGT", "1", "4", "1", "4"],
        ["accB_consensus", "tgt2", "200", "80.0", "900", "1000", "90.0", "1e-30", "300.0", "ACGA", "ACGT", "1", "4", "1", "4"],
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
        ["accC_consensus", "tgt", "100", "50.0", "5", "20", "25.0", "1e-50", "500.0", "ACNGT", "ACAGT", "1", "5", "1", "5"],
    ]
    _write_blastn_tsv(out, fields, lines)
    df = parse_blastn(str(out), fields, query_nonN_len={"accC": 10})
    row = df.row(0, named=True)
    assert row["pct_identity"] == 1.0
    assert abs(row["query_coverage"] - 0.4) < 1e-9


def test_parse_blastn_multiple_staxids_takes_first(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    lines = [
        ["accD", "tgt", "100;200;300", "100.0", "4", "4", "100.0", "1e-50", "500.0", "ACGT", "ACGT", "1", "4", "1", "4"],
    ]
    _write_blastn_tsv(out, fields, lines)
    df = parse_blastn(str(out), fields)
    assert df.row(0, named=True)["taxid"] == "100"


def test_parse_blastn_comment_only_result_is_empty(tmp_path):
    out = tmp_path / "blastn.tsv"
    out.write_text("# BLASTN 2.17.0+\n# 0 hits found\n")
    df = parse_blastn(str(out), BlastnAligner._FORMAT_FIELDS)
    assert df.height == 0
    assert df.columns == HIT_COLUMNS


def test_parse_blastn_ignores_qblast_comments(tmp_path):
    out = tmp_path / "blastn.tsv"
    out.write_text(
        "# BLASTN 2.17.0+\n"
        "accD\ttgt\t100\t100.0\t4\t4\t100.0\t1e-50\t500.0\tACGT\tACGT\t1\t4\t1\t4\n"
        "# BLAST processed 1 queries\n"
    )
    df = parse_blastn(str(out), BlastnAligner._FORMAT_FIELDS)
    assert df.height == 1
    assert df.row(0, named=True)["taxid"] == "100"


class FakeResult(StringIO):
    def __init__(self, value):
        super().__init__(value)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


def test_blastn_aligner_batches_and_awaits_each(tmp_path, monkeypatch):
    calls = []
    streams = []

    def fake_qblast(program, database, query, **kwargs):
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        calls.append((program, database, names, kwargs))
        rows = [
            f"{name}\ttgt_{name}\t100\t100.0\t4\t4\t100.0\t1e-50\t500.0\tACGT\tACGT\t1\t4\t1\t4"
            for name in names
        ]
        stream = FakeResult("# BLASTN\n" + "\n".join(rows) + "\n")
        streams.append(stream)
        return stream

    monkeypatch.setattr("Bio.Blast.qblast", fake_qblast)
    query_fasta = tmp_path / "queries.fasta"
    seqs = {f"seq{i}": "ACGT" * 10 for i in range(5)}
    write_fasta(seqs, str(query_fasta))

    aligner = BlastnAligner(batch_size=2, max_target_seqs=25)
    hits = aligner.search(str(query_fasta), tmp_dir=str(tmp_path / "work"))

    assert hits.height == 5
    assert sorted(hits["query"].to_list()) == sorted(seqs.keys())
    assert [len(call[2]) for call in calls] == [2, 2, 1]
    assert all(call[:2] == ("blastn", "nt") for call in calls)
    assert all(
        call[3]
        == {
            "format_type": "Tabular",
            "hitlist_size": 25,
            "alignments": 25,
            "descriptions": 25,
        }
        for call in calls
    )
    assert all(stream.was_closed for stream in streams)


def test_blastn_aligner_filters_excluded_taxids(tmp_path, monkeypatch):
    def fake_qblast(*args, **kwargs):
        return StringIO(
            "seq1\ttgtA\t100\t100.0\t4\t4\t100.0\t1e-50\t500.0\tACGT\tACGT\t1\t4\t1\t4\n"
            "seq1\ttgtB\t200\t100.0\t4\t4\t100.0\t1e-50\t500.0\tACGT\tACGT\t1\t4\t1\t4\n"
        )

    monkeypatch.setattr("Bio.Blast.qblast", fake_qblast)
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq1": "ACGT" * 10}, str(query_fasta))

    aligner = BlastnAligner(batch_size=3)
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


def test_blastn_minimum_ani_is_enforced(tmp_path):
    out = tmp_path / "blastn.tsv"
    fields = BlastnAligner._FORMAT_FIELDS
    lines = [
        ["query", "below", "100", "89.9", "100", "100", "100.0", "1e-50", "500.0", "A" * 899 + "C" * 101, "A" * 1000, "1", "1000", "1", "1000"],
        ["query", "at", "200", "90.0", "100", "100", "100.0", "1e-50", "499.0", "A" * 900 + "C" * 100, "A" * 1000, "1", "1000", "1", "1000"],
    ]
    _write_blastn_tsv(out, fields, lines)
    hits = parse_blastn(str(out), fields)

    filtered = filter_hits(hits, min_identity=0.9, min_aln_fraction=0.5, max_evalue=1e-10)

    assert filtered["target"].unique().to_list() == ["at"]
