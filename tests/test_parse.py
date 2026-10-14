"""Plain-text documents must be read as UTF-8 / Arabic, never with the OS default encoding (cp1252 on Windows)."""
from DB.parse import Parser

TEXT = "المادة 7: يحق للمراجع التظلم خلال 30 يوماً."


def test_read_plain_utf8_bom_utf16_cp1256(tmp_path):
    cases = {"utf8.txt": TEXT.encode("utf-8"), "bom.txt": TEXT.encode("utf-8-sig"),
             "utf16.txt": TEXT.encode("utf-16"), "cp1256.txt": TEXT.encode("cp1256")}
    for name, data in cases.items():
        (tmp_path / name).write_bytes(data)
        md, tool, status = Parser(tmp_path, tmp_path / "out").convert(tmp_path / name)
        assert (md, tool, status) == (TEXT, "plain", "ok"), name
