import os

from postviritu.io_esviritu import (
    SamplePaths,
    discover_sample_prefixes,
    get_thresholds,
    load_params,
    parse_consensus_fasta,
    read_tsv,
    write_fasta,
)


def test_discover_sample_prefixes(example_data_dir, example_prefix):
    prefixes = discover_sample_prefixes(example_data_dir)
    assert example_prefix in prefixes


def test_sample_paths_resolution(example_data_dir, example_prefix):
    sp = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    assert os.path.isfile(sp.info)
    assert os.path.isfile(sp.consensus)
    assert sp.missing_required() == []


def test_parse_consensus_strips_suffix(example_data_dir, example_prefix):
    sp = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    seqs = parse_consensus_fasta(sp.consensus)
    # Headers like '>Y15173.1_consensus' should map to accession 'Y15173.1'.
    assert "Y15173.1" in seqs
    assert all(not k.endswith("_consensus") for k in seqs)
    assert all(len(v) > 0 for v in seqs.values())


def test_thresholds_from_params(example_data_dir, example_prefix):
    sp = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    params = load_params(sp.params)
    sp_t, subsp_t = get_thresholds(params)
    assert sp_t == 0.9
    assert subsp_t == 0.95


def test_write_fasta_roundtrip(tmp_path):
    seqs = {"acc1": "ACGT" * 30, "acc2": "TTTT"}
    out = tmp_path / "out.fasta"
    write_fasta(seqs, str(out))
    back = parse_consensus_fasta(str(out))
    assert back == seqs


def test_read_info_has_expected_columns(example_data_dir, example_prefix):
    sp = SamplePaths(prefix=example_prefix, directory=example_data_dir)
    df = read_tsv(sp.info)
    for col in ["Accession", "Assembly", "species", "subspecies", "read_count"]:
        assert col in df.columns
