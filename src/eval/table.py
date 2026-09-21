import json
import os
from pathlib import Path

TABLE_FILE = os.environ.get("SFX_TABLE", "global").removesuffix(".json") + ".json"
IMAGE_PATH = Path("/root/data/tables") / TABLE_FILE
HOST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tables" / TABLE_FILE
TABLE_SOURCE = "benchmark" if TABLE_FILE == "benchmark.json" else "TraceLab syfi_coding_trace"


def resolve_table_path():
    return IMAGE_PATH if IMAGE_PATH.exists() else HOST_PATH


def load_table(path=None):
    return json.loads(Path(path or resolve_table_path()).read_bytes())
