# tests/test_cli.py
from ml import cli

def test_cli_unknown_stage_returns_nonzero():
    assert cli.main(["bogus", "play1"]) != 0

def test_cli_help_lists_stages(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    for stage in ("detect", "track", "positions", "all"):
        assert stage in out
