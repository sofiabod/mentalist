import json

from eval import table


def test_load_table_prefers_image_path_when_present(tmp_path, monkeypatch):
    img = tmp_path / "data" / "tables" / "global.json"
    img.parent.mkdir(parents=True)
    img.write_text(json.dumps({"from": "image"}))
    monkeypatch.setattr(table, "IMAGE_PATH", img)
    assert table.resolve_table_path() == img
    assert table.load_table() == {"from": "image"}


def test_load_table_falls_back_to_host_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "IMAGE_PATH", tmp_path / "data" / "tables" / "global.json")
    assert table.resolve_table_path() == table.HOST_PATH
    assert table.HOST_PATH.exists()
    table.load_table()


def test_image_path_is_on_node_absolute():
    assert str(table.IMAGE_PATH) == "/root/data/tables/global.json"


def test_sfx_table_env_selects_benchmark(monkeypatch):
    import importlib
    monkeypatch.setenv("SFX_TABLE", "benchmark")
    importlib.reload(table)
    try:
        assert table.HOST_PATH.name == "benchmark.json"
        t = table.load_table()
        assert t["provenance"] == "benchmark"
        assert table.TABLE_SOURCE == "benchmark"
    finally:
        monkeypatch.delenv("SFX_TABLE", raising=False)
        importlib.reload(table)
