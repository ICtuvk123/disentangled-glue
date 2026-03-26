r"""
Tests for the :mod:`scglue.utils` module
"""

import logging

import scglue


def test_log_manager(tmp_path):
    file1 = tmp_path / "log1.log"
    scglue.log.log_file = file1
    scglue.log.console_log_level = logging.WARNING
    scglue.log.file_log_level = logging.INFO
    logger1 = scglue.log.get_logger("logger1")

    file2 = tmp_path / "log2.log"
    scglue.log.log_file = file2
    scglue.log.console_log_level = logging.DEBUG
    scglue.log.file_log_level = logging.DEBUG
    logger2 = scglue.log.get_logger("logger1")
    assert logger1 is logger2

    scglue.log.log_file = None
    file1.unlink()
    file2.unlink()


def test_get_rs():
    rs1 = scglue.utils.get_rs(0)
    rs2 = scglue.utils.get_rs(rs1)
    assert rs1 is rs2
    rs1 = scglue.utils.get_rs()
    rs2 = scglue.utils.get_rs()
    assert rs1 is rs2


def test_bedtools_path_accepts_executable_path(monkeypatch):
    calls = []

    monkeypatch.setattr(scglue.utils, "set_bedtools_path", calls.append)
    old_bedtools_path = scglue.config.BEDTOOLS_PATH
    try:
        scglue.config.BEDTOOLS_PATH = "/tmp/custom-bin/bedtools"
        assert scglue.config.BEDTOOLS_PATH == "/tmp/custom-bin/bedtools"
        assert calls[-1] == "/tmp/custom-bin"
    finally:
        scglue.config.BEDTOOLS_PATH = old_bedtools_path


def test_bedtools_path_accepts_command_name(monkeypatch):
    calls = []

    monkeypatch.setattr(scglue.utils, "set_bedtools_path", calls.append)
    old_bedtools_path = scglue.config.BEDTOOLS_PATH
    try:
        scglue.config.BEDTOOLS_PATH = "bedtools"
        assert scglue.config.BEDTOOLS_PATH == "bedtools"
        assert calls[-1] == ""
    finally:
        scglue.config.BEDTOOLS_PATH = old_bedtools_path

