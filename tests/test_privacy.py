from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_forbidden_org_string_in_src():
    forbidden = "s" + "entra"
    for p in (ROOT / "src").rglob("*.py"):
        assert forbidden not in p.read_text().lower(), p
