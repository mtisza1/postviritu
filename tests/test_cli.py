"""Tests for CLI wiring of the Virus Variation client policy."""

import pytest

from postviritu.cli import _build_parser, _vvsearch_config

_RUN = ["run", "--input-dir", "in", "--db", "db", "--outdir", "out"]
_BLASTN = ["blastn", "--input-dir", "in", "--outdir", "out"]


@pytest.mark.parametrize("base", [_RUN, _BLASTN], ids=["run", "blastn"])
def test_vvsearch_enabled_by_default(base):
    config = _vvsearch_config(_build_parser().parse_args(base))
    assert config.enabled is True
    assert config.tool == "postviritu"
    assert config.min_interval > 0  # NCBI rate-limit guidance


@pytest.mark.parametrize("base", [_RUN, _BLASTN], ids=["run", "blastn"])
def test_no_vvsearch_makes_the_run_offline(base):
    config = _vvsearch_config(_build_parser().parse_args(base + ["--no-vvsearch"]))
    assert config.enabled is False


@pytest.mark.parametrize("base", [_RUN, _BLASTN], ids=["run", "blastn"])
def test_vvsearch_contact_and_timeout_are_configurable(base):
    args = _build_parser().parse_args(
        base + ["--vvsearch-email", "lab@example.org", "--vvsearch-timeout", "2.5"]
    )
    config = _vvsearch_config(args)
    assert config.email == "lab@example.org"
    assert config.timeout == 2.5
