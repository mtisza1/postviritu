from postviritu.aligner import (
    HIT_COLUMNS,
    Mmseqs2Aligner,
    filter_hits,
    parse_m8,
)


def _write_m8(path):
    # fields: query target taxid fident alnlen qlen qcov evalue bits
    lines = [
        "accA_consensus\ttgt1\t100\t0.99\t1000\t1000\t0.95\t1e-50\t500",
        "accA_consensus\ttgt2\t200\t0.80\t900\t1000\t0.90\t1e-30\t300",
        "accB_consensus\ttgt3\t0\t0.99\t800\t800\t0.99\t1e-40\t400",
    ]
    path.write_text("\n".join(lines) + "\n")


def test_parse_m8_strips_consensus_and_schema(tmp_path):
    m8 = tmp_path / "result.m8"
    _write_m8(m8)
    df = parse_m8(str(m8), Mmseqs2Aligner._FORMAT_FIELDS)
    assert df.columns == HIT_COLUMNS
    assert set(df["query"].to_list()) == {"accA", "accB"}
    assert df["pct_identity"].dtype.is_float()


def test_parse_m8_empty(tmp_path):
    m8 = tmp_path / "empty.m8"
    m8.write_text("")
    df = parse_m8(str(m8), Mmseqs2Aligner._FORMAT_FIELDS)
    assert df.height == 0
    assert df.columns == HIT_COLUMNS


def test_filter_hits_drops_low_quality_and_no_taxid(tmp_path):
    m8 = tmp_path / "result.m8"
    _write_m8(m8)
    df = parse_m8(str(m8), Mmseqs2Aligner._FORMAT_FIELDS)
    filtered = filter_hits(df, min_identity=0.9, min_aln_fraction=0.5, max_evalue=1e-10)
    # Only the first row passes: pident 0.99, taxid 100, good evalue.
    assert filtered.height == 1
    assert filtered["taxid"].to_list() == ["100"]
