"""Tests for CLI wiring of the Virus Variation client policy."""

import pytest

from postviritu.cli import _build_parser, _vvsearch_config

_MMSEQS = ["mmseqs", "--input-dir", "in", "--db", "db", "--outdir", "out"]
_BLASTN = ["blastn", "--input-dir", "in", "--outdir", "out"]


def test_mmseqs_subcommands_replace_old_names(capsys):
    parser = _build_parser()
    assert parser.parse_args(_MMSEQS).command == "mmseqs"
    assert parser.parse_args(
        ["setup-mmseqs", "--fasta", "f", "--taxdump", "t", "--acc2taxid", "a", "--out", "o"]
    ).command == "setup-mmseqs"
    for old_name in ("run", "setup-db"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([old_name])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "postviritu 0.4.0"


@pytest.mark.parametrize("base", [_MMSEQS, _BLASTN], ids=["mmseqs", "blastn"])
def test_vvsearch_enabled_by_default(base):
    config = _vvsearch_config(_build_parser().parse_args(base))
    assert config.enabled is True
    assert config.tool == "postviritu"
    assert config.min_interval > 0  # NCBI rate-limit guidance


@pytest.mark.parametrize("base", [_MMSEQS, _BLASTN], ids=["mmseqs", "blastn"])
def test_no_vvsearch_makes_the_run_offline(base):
    config = _vvsearch_config(_build_parser().parse_args(base + ["--no-vvsearch"]))
    assert config.enabled is False


@pytest.mark.parametrize("base", [_MMSEQS, _BLASTN], ids=["mmseqs", "blastn"])
def test_vvsearch_contact_and_timeout_are_configurable(base):
    args = _build_parser().parse_args(
        base + ["--vvsearch-email", "lab@example.org", "--vvsearch-timeout", "2.5"]
    )
    config = _vvsearch_config(args)
    assert config.email == "lab@example.org"
    assert config.timeout == 2.5
