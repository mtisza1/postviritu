import io
import zipfile
from xml.sax.saxutils import escape

from postviritu.aligner import (
    BlastnAligner,
    HIT_COLUMNS,
    filter_hits,
    parse_blastn,
)
from postviritu.io_esviritu import write_fasta


def _hsp_xml(qseq, hseq, qfrom, hfrom, hto, evalue=1e-50, bits=500.0, strand="Plus"):
    identity = sum(q == h and q != "-" for q, h in zip(qseq, hseq))
    qto = qfrom + len(qseq.replace("-", "")) - 1
    return f"""
<Hsp>
  <num>1</num>
  <bit-score>{bits}</bit-score>
  <score>{int(bits)}</score>
  <evalue>{evalue}</evalue>
  <identity>{identity}</identity>
  <query-from>{qfrom}</query-from>
  <query-to>{qto}</query-to>
  <query-strand>Plus</query-strand>
  <hit-from>{hfrom}</hit-from>
  <hit-to>{hto}</hit-to>
  <hit-strand>{strand}</hit-strand>
  <align-len>{len(qseq)}</align-len>
  <gaps>{qseq.count("-") + hseq.count("-")}</gaps>
  <qseq>{qseq}</qseq>
  <hseq>{hseq}</hseq>
  <midline>{"".join("|" if q == h else " " for q, h in zip(qseq, hseq))}</midline>
</Hsp>"""


def _hit_xml(target, taxid, length, hsps, num=1):
    return f"""
<Hit>
  <num>{num}</num>
  <description>
    <HitDescr>
      <id>{escape(target)}</id>
      <accession>{escape(target)}</accession>
      <title>{escape(target)} virus</title>
      <taxid>{taxid}</taxid>
      <sciname>Some virus</sciname>
    </HitDescr>
  </description>
  <len>{length}</len>
  <hsps>{"".join(hsps)}</hsps>
</Hit>"""


def _search_xml(query, query_len, hits, num=1, message=None):
    message_xml = f"<message>{escape(message)}</message>" if message else ""
    return f"""<?xml version="1.0"?>
<BlastXML2 xmlns="http://www.ncbi.nlm.nih.gov" xmlns:xs="http://www.w3.org/2001/XMLSchema-instance" xs:schemaLocation="http://www.ncbi.nlm.nih.gov http://www.ncbi.nlm.nih.gov/data_specs/schema_alt/NCBI_BlastOutput2.xsd">
<BlastOutput2>
<report>
<Report>
  <program>blastn</program>
  <version>BLASTN 2.17.0+</version>
  <reference>ref</reference>
  <search-target><Target><db>nt</db></Target></search-target>
  <params><Parameters><expect>10</expect><sc-match>2</sc-match><sc-mismatch>-3</sc-mismatch><gap-open>5</gap-open><gap-extend>2</gap-extend><filter>L;m;</filter></Parameters></params>
  <results>
  <Results>
  <search>
  <Search>
    <query-id>Query_{num}</query-id>
    <query-title>{escape(query)}</query-title>
    <query-len>{query_len}</query-len>
    <hits>{"".join(hits)}</hits>
    {message_xml}
  </Search>
  </search>
  </Results>
  </results>
</Report>
</report>
</BlastOutput2>
</BlastXML2>
"""


def _qblast_zip(searches):
    """Build a QBLAST-style XML2 ZIP: a master file plus one file per query."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        includes = "".join(
            f'<xi:include href="RID_{i}.xml"/>' for i in range(1, len(searches) + 1)
        )
        archive.writestr(
            "RID.xml",
            '<?xml version="1.0"?>\n<BlastXML2 xmlns="http://www.ncbi.nlm.nih.gov" '
            f'xmlns:xi="http://www.w3.org/2003/XInclude">{includes}</BlastXML2>\n',
        )
        for i, search in enumerate(searches, start=1):
            archive.writestr(f"RID_{i}.xml", search)
    return buffer.getvalue()


def _write_zip(path, searches):
    path.write_bytes(_qblast_zip(searches))


def test_parse_blastn_strips_consensus_and_scales(tmp_path):
    out = tmp_path / "blastn.zip"
    _write_zip(
        out,
        [
            _search_xml(
                "accA_consensus",
                20,
                [_hit_xml("tgt1", 100, 100, [_hsp_xml("ACGTACGTACGTACGTACG", "ACGTACGTACGTACGTACG", 1, 11, 29)])],
                num=1,
            ),
            _search_xml(
                "accB_consensus",
                20,
                [_hit_xml("tgt2", 200, 100, [_hsp_xml("ACGAACGAACGAACGAACGA", "ACGTACGTACGTACGTACGT", 1, 1, 20, evalue=1e-30, bits=300.0)])],
                num=2,
            ),
        ],
    )
    df = parse_blastn(str(out))
    assert df.columns == HIT_COLUMNS
    assert set(df["query"].to_list()) == {"accA", "accB"}
    assert sorted(df["taxid"].to_list()) == ["100", "200"]

    filtered = filter_hits(df, min_identity=0.9, min_aln_fraction=0.5, max_evalue=1e-10)
    assert filtered.height == 1
    row = filtered.row(0, named=True)
    assert row["query"] == "accA"
    assert row["target"] == "tgt1"
    assert row["taxid"] == "100"
    assert row["pct_identity"] == 1.0
    assert abs(row["query_coverage"] - 0.95) < 1e-9
    assert (row["qstart"], row["qend"], row["tstart"], row["tend"]) == (1, 19, 11, 29)


def test_parse_blastn_empty(tmp_path):
    out = tmp_path / "empty.zip"
    out.write_text("")
    df = parse_blastn(str(out))
    assert df.height == 0
    assert df.columns == HIT_COLUMNS


def test_parse_blastn_no_hits_is_empty(tmp_path):
    out = tmp_path / "blastn.zip"
    _write_zip(out, [_search_xml("accA", 20, [])])
    df = parse_blastn(str(out))
    assert df.height == 0
    assert df.columns == HIT_COLUMNS


def test_parse_blastn_nonN_metrics_exclude_query_N(tmp_path):
    out = tmp_path / "blastn.zip"
    _write_zip(
        out,
        [_search_xml("accC_consensus", 20, [_hit_xml("tgt", 100, 100, [_hsp_xml("ACNGT", "ACAGT", 1, 1, 5)])])],
    )
    df = parse_blastn(str(out), query_nonN_len={"accC": 10})
    row = df.row(0, named=True)
    assert row["qaln"] == "ACNGT"
    assert row["taln"] == "ACAGT"
    assert row["pct_identity"] == 1.0
    assert abs(row["query_coverage"] - 0.4) < 1e-9


def test_parse_blastn_minus_strand_and_gaps(tmp_path):
    out = tmp_path / "blastn.zip"
    _write_zip(
        out,
        [_search_xml("accD", 10, [_hit_xml("tgt", 100, 100, [_hsp_xml("ACG-TACGT", "ACGGTACGT", 2, 60, 52, strand="Minus")])])],
    )
    row = parse_blastn(str(out)).row(0, named=True)
    assert row["qaln"] == "ACG-TACGT"
    assert row["taln"] == "ACGGTACGT"
    assert (row["qstart"], row["qend"], row["tstart"], row["tend"]) == (2, 9, 60, 52)
    assert row["segment_aln_length"] == 9
    assert abs(row["segment_query_coverage"] - 0.8) < 1e-9


def test_parse_blastn_plain_xml_file(tmp_path):
    out = tmp_path / "blastn.xml"
    out.write_text(_search_xml("accE", 4, [_hit_xml("tgt", 300, 100, [_hsp_xml("ACGT", "ACGT", 1, 1, 4)])]))
    df = parse_blastn(str(out))
    assert df.height == 1
    assert df.row(0, named=True)["taxid"] == "300"


class FakeResult(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


def _identical_hit_zip(names, targets=None):
    targets = targets or {name: [(f"tgt_{name}", 100)] for name in names}
    return _qblast_zip(
        [
            _search_xml(
                name,
                4,
                [
                    _hit_xml(target, taxid, 100, [_hsp_xml("ACGT", "ACGT", 1, 1, 4)], num=j)
                    for j, (target, taxid) in enumerate(targets[name], start=1)
                ],
                num=i,
            )
            for i, name in enumerate(names, start=1)
        ]
    )


def test_blastn_aligner_batches_and_awaits_each(tmp_path, monkeypatch):
    calls = []
    streams = []

    def fake_qblast(program, database, query, **kwargs):
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        calls.append((program, database, names, kwargs))
        stream = FakeResult(_identical_hit_zip(names))
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
            "format_type": "XML2",
            "hitlist_size": 25,
            "alignments": 25,
            "descriptions": 25,
        }
        for call in calls
    )
    assert all(stream.was_closed for stream in streams)


def test_blastn_aligner_logs_batch_progress(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="postviritu")

    def fake_qblast(program, database, query, **kwargs):
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        return FakeResult(_identical_hit_zip(names))

    monkeypatch.setattr("Bio.Blast.qblast", fake_qblast)
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq0": "ACGT" * 30, "seq1": "ACGT" * 10}, str(query_fasta))

    BlastnAligner(batch_size=1).search(str(query_fasta), tmp_dir=str(tmp_path / "work"))

    messages = [r.getMessage() for r in caplog.records]
    assert "BLASTN: 2 queries (160 bp) against 'nt' in 2 remote batch(es) of <= 1" in messages
    assert any(
        m.startswith("BLASTN batch 1/2: submitting 1 query (120 bp; 50.0% of queries, 75.0% of bp): seq0")
        for m in messages
    )
    assert any("Progress: 2/2 queries (100.0%), 100.0% of bp" in m for m in messages)


def test_blastn_aligner_logs_failed_batch(tmp_path, monkeypatch, caplog):
    def failing_qblast(*args, **kwargs):
        raise ValueError("NCBI said no")

    monkeypatch.setattr("Bio.Blast.qblast", failing_qblast)
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq0": "ACGT" * 10}, str(query_fasta))

    try:
        BlastnAligner().search(str(query_fasta), tmp_dir=str(tmp_path / "work"))
    except ValueError:
        pass
    else:
        raise AssertionError("the BLAST failure must propagate")

    assert any(
        r.levelname == "ERROR" and "BLASTN batch 1/1 failed" in r.getMessage()
        and "ValueError: NCBI said no" in r.getMessage()
        for r in caplog.records
    )


def _slow_then_ok_qblast(n_slow, calls):
    import warnings

    from Bio import BiopythonWarning

    def fake_qblast(program, database, query, **kwargs):
        calls.append(query)
        if len(calls) <= n_slow:
            warnings.warn(
                f"BLAST request RID{len(calls)} is taking longer than 10 minutes, "
                "consider re-issuing it",
                BiopythonWarning,
            )
            raise AssertionError("the timeout warning must abort the poll loop")
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        return FakeResult(_identical_hit_zip(names))

    return fake_qblast


def test_blastn_aligner_retries_slow_request(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr("Bio.Blast.qblast", _slow_then_ok_qblast(2, calls))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq0": "ACGT" * 10}, str(query_fasta))

    hits = BlastnAligner(max_retries=2).search(
        str(query_fasta), tmp_dir=str(tmp_path / "work")
    )

    assert hits.height == 1
    assert len(calls) == 3
    assert len(set(calls)) == 1
    retries = [r for r in caplog.records if "timed out" in r.getMessage()]
    assert [r.levelname for r in retries] == ["WARNING", "WARNING"]
    assert "attempt 1/3" in retries[0].getMessage()


def test_blastn_aligner_gives_up_after_retries(tmp_path, monkeypatch, caplog):
    from Bio import BiopythonWarning

    calls = []
    monkeypatch.setattr("Bio.Blast.qblast", _slow_then_ok_qblast(10, calls))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq0": "ACGT" * 10}, str(query_fasta))

    try:
        BlastnAligner(max_retries=1).search(str(query_fasta), tmp_dir=str(tmp_path / "work"))
    except BiopythonWarning:
        pass
    else:
        raise AssertionError("the timeout must propagate once retries are exhausted")

    assert len(calls) == 2
    assert any(
        r.levelname == "ERROR" and "BLASTN batch 1/1 failed" in r.getMessage()
        for r in caplog.records
    )


_CPU_LIMIT = (
    "Informational Message: [blastsrv4.REAL]: Error: CPU usage limit was exceeded, "
    "resulting in SIGXCPU (24).\nNo hits found"
)


def _cpu_limited_qblast(always_fail, calls):
    """Fake QBLAST: queries in ``always_fail`` always hit the CPU limit; any
    other query hits it only when searched together with 3+ queries."""

    def fake_qblast(program, database, query, **kwargs):
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        calls.append(names)
        searches = []
        for i, name in enumerate(names, start=1):
            if name in always_fail or len(names) >= 3:
                searches.append(_search_xml(name, 40, [], num=i, message=_CPU_LIMIT))
            else:
                hit = _hit_xml(f"tgt_{name}", 100, 100, [_hsp_xml("ACGT", "ACGT", 1, 1, 4)])
                searches.append(_search_xml(name, 4, [hit], num=i))
        return FakeResult(_qblast_zip(searches))

    return fake_qblast


def test_parse_blastn_tolerates_search_messages(tmp_path):
    path = tmp_path / "msg.xml2.zip"
    hit = _hit_xml("tgtA", 100, 100, [_hsp_xml("ACGT", "ACGT", 1, 1, 4)])
    _write_zip(
        path,
        [
            _search_xml("q1", 4, [hit], num=1),
            _search_xml("q2", 40, [], num=2, message=_CPU_LIMIT),
        ],
    )

    hits = parse_blastn(str(path))

    assert hits["query"].to_list() == ["q1"]


def test_blastn_aligner_splits_cpu_limited_batches(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr("Bio.Blast.qblast", _cpu_limited_qblast(set(), calls))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({f"seq{i}": "ACGT" * 10 for i in range(4)}, str(query_fasta))

    hits = BlastnAligner(batch_size=4).search(str(query_fasta), tmp_dir=str(tmp_path / "work"))

    assert sorted(hits["query"].to_list()) == ["seq0", "seq1", "seq2", "seq3"]
    assert calls == [["seq0", "seq1", "seq2", "seq3"], ["seq0", "seq1"], ["seq2", "seq3"]]
    assert any(
        r.levelname == "WARNING" and "CPU usage limit was exceeded for 4/4 queries" in r.getMessage()
        for r in caplog.records
    )


def test_blastn_aligner_reports_query_that_fails_alone(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr("Bio.Blast.qblast", _cpu_limited_qblast({"seq1"}, calls))
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq0": "ACGT" * 10, "seq1": "ACGT" * 10}, str(query_fasta))

    hits = BlastnAligner(batch_size=2).search(str(query_fasta), tmp_dir=str(tmp_path / "work"))

    assert hits["query"].to_list() == ["seq0"]
    assert calls == [["seq0", "seq1"], ["seq1"]]
    assert any(
        r.levelname == "WARNING"
        and "seq1 exceeded NCBI's CPU usage limit even when searched alone" in r.getMessage()
        for r in caplog.records
    )


def test_blastn_aligner_keep_writes_result_table(tmp_path, monkeypatch):
    def fake_qblast(program, database, query, **kwargs):
        names = [line[1:] for line in query.splitlines() if line.startswith(">")]
        return FakeResult(_identical_hit_zip(names))

    monkeypatch.setattr("Bio.Blast.qblast", fake_qblast)
    query_fasta = tmp_path / "queries.fasta"
    write_fasta({"seq1": "ACGT" * 10}, str(query_fasta))

    work = tmp_path / "work"
    BlastnAligner().search(
        str(query_fasta), tmp_dir=str(work), keep=True, result_name="s_alignment.m8"
    )
    assert (work / "s_alignment.m8").is_file()
    assert (work / "batch_0.xml2.zip").is_file()
    assert not (work / "batch_0.fasta").exists()


def test_blastn_aligner_filters_excluded_taxids(tmp_path, monkeypatch):
    def fake_qblast(*args, **kwargs):
        return io.BytesIO(
            _identical_hit_zip(["seq1"], {"seq1": [("tgtA", 100), ("tgtB", 200)]})
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
    assert args.blast_retries == 3


def test_blastn_minimum_ani_is_enforced(tmp_path):
    out = tmp_path / "blastn.zip"
    _write_zip(
        out,
        [
            _search_xml(
                "query",
                1000,
                [
                    _hit_xml("below", 100, 1000, [_hsp_xml("A" * 899 + "C" * 101, "A" * 1000, 1, 1, 1000)], num=1),
                    _hit_xml("at", 200, 1000, [_hsp_xml("A" * 900 + "C" * 100, "A" * 1000, 1, 1, 1000, bits=499.0)], num=2),
                ],
            )
        ],
    )
    hits = parse_blastn(str(out))

    filtered = filter_hits(hits, min_identity=0.9, min_aln_fraction=0.5, max_evalue=1e-10)

    assert filtered["target"].unique().to_list() == ["at"]
