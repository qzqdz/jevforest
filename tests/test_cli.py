"""CLI smoke."""

from __future__ import annotations

import pytest

from jevforest.cli.main import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_eval_requires_config() -> None:
    assert main(["eval-afa"]) == 2
