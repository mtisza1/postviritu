import polars as pl

from postviritu.aligner import (
    HIT_COLUMNS,
    Mmseqs2Aligner,
    aggregate_hits,
    filter_hits,
    parse_m8,
)


def _write_m8(path):
    # fields: query target taxid fident alnlen qlen qcov evalue bits qaln taln qstart qend tstart tend
    # pct_identity/query_coverage are recomputed from qaln/taln (N-aware), so
    # identity here is driven by the aligned strings rather than fident.
    lines = [
        "accA_consensus\ttgt1\t100\t0.99\t1000\t1000\t0.95\t1e-50\t500\tACGT\tACGT\t1\t4\t1\t4",
        "accA_consensus\ttgt2\t200\t0.80\t900\t1000\t0.90\t1e-30\t300\tACGT\tACGA\t1\t4\t1\t4",
        "accB_consensus\ttgt3\t0\t0.99\t800\t800\t0.99\t1e-40\t400\tACGT\tACGT\t1\t4\t1\t4",
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
    # Only the first row passes: recomputed identity 1.0 (ACGT/ACGT), taxid 100,
    # good evalue. Row 2 fails identity (3/4), row 3 has taxid 0.
    assert filtered.height == 1
    assert filtered["taxid"].to_list() == ["100"]


def test_parse_m8_nonN_metrics_exclude_query_N(tmp_path):
    m8 = tmp_path / "result.m8"
    # qaln has an N (column 3); the 4 non-N query residues all match the target.
    line = "accC_consensus\ttgt\t100\t0.5\t5\t20\t0.25\t1e-50\t500\tACNGT\tACAGT\t1\t5\t1\t5"
    m8.write_text(line + "\n")
    df = parse_m8(
        str(m8), Mmseqs2Aligner._FORMAT_FIELDS, query_nonN_len={"accC": 10}
    )
    row = df.row(0, named=True)
    # Identity = matches / non-N aligned columns = 4 / 4 = 1.0.
    assert row["pct_identity"] == 1.0
    # Coverage = aligned non-N query residues / total non-N query residues
    #          = 4 / 10 = 0.4.
    assert abs(row["query_coverage"] - 0.4) < 1e-9


def test_parse_m8_nonN_metrics_exclude_target_N(tmp_path):
    m8 = tmp_path / "result.m8"
    # taln has an N (column 2): excluded from the identity denominator, but the
    # opposing non-N query residue still counts toward coverage.
    line = "accD_consensus\ttgt\t100\t0.5\t4\t4\t1.0\t1e-50\t500\tACGT\tANGT\t1\t4\t1\t4"
    m8.write_text(line + "\n")
    df = parse_m8(
        str(m8), Mmseqs2Aligner._FORMAT_FIELDS, query_nonN_len={"accD": 4}
    )
    row = df.row(0, named=True)
    # Remaining 3 aligned columns all match -> identity 1.0.
    assert row["pct_identity"] == 1.0
    # All 4 query residues are non-N and aligned -> coverage 1.0.
    assert row["query_coverage"] == 1.0


def test_parse_m8_coverage_fallback_without_map(tmp_path):
    m8 = tmp_path / "result.m8"
    # No query_nonN_len provided -> coverage falls back to the mmseqs qcov (0.42).
    line = "accE_consensus\ttgt\t100\t0.5\t4\t4\t0.42\t1e-50\t500\tACGT\tACGT\t1\t4\t1\t4"
    m8.write_text(line + "\n")
    df = parse_m8(str(m8), Mmseqs2Aligner._FORMAT_FIELDS)
    row = df.row(0, named=True)
    assert row["pct_identity"] == 1.0
    assert abs(row["query_coverage"] - 0.42) < 1e-9


def test_parse_m8_coverage_counts_only_canonical_bases(tmp_path):
    m8 = tmp_path / "result.m8"
    # The aligned query contains an N at position 3. Only the four canonical
    # (A, T, C, G) aligned positions count toward coverage; the full query has
    # five canonical bases.
    line = "accF_consensus\ttgt\t100\t0.5\t5\t10\t0.5\t1e-50\t500\tACNGT\tACAGT\t1\t5\t1\t5"
    m8.write_text(line + "\n")
    df = parse_m8(
        str(m8), Mmseqs2Aligner._FORMAT_FIELDS, query_nonN_len={"accF": 5}
    )
    row = df.row(0, named=True)
    # Identity denominator excludes the N column -> 4 canonical matches / 4.
    assert row["pct_identity"] == 1.0
    # Coverage = aligned canonical query bases / total canonical query bases.
    assert abs(row["query_coverage"] - 0.8) < 1e-9


def test_aggregate_hits_chains_non_overlapping_segments_and_rejects_overlap():
    hits = pl.DataFrame(
        [
            {
                "query": "query1",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 40,
                "query_length": 100,
                "query_coverage": 0.4,
                "evalue": 1e-20,
                "bitscore": 50.0,
                "qaln": "A" * 40,
                "taln": "A" * 40,
                "qstart": 1,
                "qend": 40,
                "tstart": 101,
                "tend": 140,
            },
            {
                "query": "query1",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 0.95,
                "aln_length": 40,
                "query_length": 100,
                "query_coverage": 0.4,
                "evalue": 1e-18,
                "bitscore": 60.0,
                "qaln": "C" * 40,
                "taln": "C" * 38 + "AA",
                "qstart": 61,
                "qend": 100,
                "tstart": 161,
                "tend": 200,
            },
            {
                "query": "query1",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 61,
                "query_length": 100,
                "query_coverage": 0.61,
                "evalue": 1e-25,
                "bitscore": 80.0,
                "qaln": "G" * 61,
                "taln": "G" * 61,
                "qstart": 20,
                "qend": 80,
                "tstart": 120,
                "tend": 180,
            },
        ]
    )

    aggregated = aggregate_hits(hits)

    assert aggregated.height == 3
    assert aggregated["segment_selected"].to_list() == [True, True, False]
    assert aggregated["segment_bitscore"].to_list() == [50.0, 60.0, 80.0]
    assert aggregated["bitscore"].to_list() == [110.0, 110.0, 110.0]
    assert aggregated["query_coverage"].to_list() == [0.8, 0.8, 0.8]
    assert aggregated["pct_identity"].to_list() == [0.975, 0.975, 0.975]


def test_combined_partial_segments_pass_reference_level_filter():
    hits = pl.DataFrame(
        [
            {
                "query": "query1",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 30,
                "query_length": 100,
                "query_coverage": 0.3,
                "evalue": 1e-20,
                "bitscore": 40.0,
                "qaln": "A" * 30,
                "taln": "A" * 30,
                "qstart": 1,
                "qend": 30,
                "tstart": 1,
                "tend": 30,
            },
            {
                "query": "query1",
                "target": "ref1",
                "taxid": "100",
                "pct_identity": 1.0,
                "aln_length": 30,
                "query_length": 100,
                "query_coverage": 0.3,
                "evalue": 1e-20,
                "bitscore": 50.0,
                "qaln": "C" * 30,
                "taln": "C" * 30,
                "qstart": 71,
                "qend": 100,
                "tstart": 71,
                "tend": 100,
            },
        ]
    )

    filtered = filter_hits(
        aggregate_hits(hits),
        min_identity=0.9,
        min_aln_fraction=0.5,
        max_evalue=1e-10,
    )

    assert filtered.height == 2
    assert filtered["bitscore"].to_list() == [90.0, 90.0]
