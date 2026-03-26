r"""
Tests for the :mod:`scglue.check` module
"""

import pytest

import scglue.check


def test_module_checker():
    module_checker = scglue.check.ModuleChecker(
        "numpy", vmin=None, install_hint="You may install via..."
    )
    module_checker.check()

    module_checker = scglue.check.ModuleChecker(
        "numpy", vmin="99.99.99", install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        module_checker.check()

    module_checker = scglue.check.ModuleChecker(
        "xxx", vmin=None, install_hint="You may install via..."
    )
    with pytest.raises(RuntimeError):
        module_checker.check()


def test_cmd_checker():
    cmd_checker = scglue.check.CmdChecker(
        "ls",
        "ls --version",
        r"([0-9\.]+)$",
        vmin=None,
        install_hint="You may install via...",
    )
    cmd_checker.check()

    cmd_checker = scglue.check.CmdChecker(
        "ls",
        "ls --version",
        r"([0-9\.]+)$",
        vmin="99.99.99",
        install_hint="You may install via...",
    )
    with pytest.raises(RuntimeError):
        cmd_checker.check()

    cmd_checker = scglue.check.CmdChecker(
        "xxx",
        "xxx --version",
        r"([0-9\.]+)$",
        vmin=None,
        install_hint="You may install via...",
    )
    with pytest.raises(RuntimeError):
        cmd_checker.check()


def test_bedtools_checker_honors_runtime_config(monkeypatch):
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        return ["bedtools v2.31.1"]

    monkeypatch.setattr(scglue.check, "run_command", fake_run_command)
    old_bedtools_path = scglue.check.config.BEDTOOLS_PATH
    try:
        scglue.check.config.BEDTOOLS_PATH = "/tmp/custom-bedtools"
        scglue.check.CHECKERS["bedtools"].check()
    finally:
        scglue.check.config.BEDTOOLS_PATH = old_bedtools_path

    assert commands == ["/tmp/custom-bedtools --version"]

