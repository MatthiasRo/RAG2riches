from pathlib import Path

import pytest

pytest.importorskip("lancedb")

import lancedb

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "lanceDB_PR"
TABLE_NAME = "chunk_registry"

def supports_filter_and_columns(tbl):
    try:
        _ = tbl.to_pandas(filter="chunk_id = 'dummy_id'", columns=["chunk_id"])
        return True, True
    except TypeError:
        # Try columns-only support
        try:
            _ = tbl.to_pandas(columns=["chunk_id"])
            return False, True
        except TypeError:
            return False, False


def test_lancedb_filtering_capabilities():
    if not DATA_PATH.exists():
        pytest.skip(f"LanceDB test skipped: {DATA_PATH} not found")

    db = lancedb.connect(str(DATA_PATH))
    tables = db.list_tables()
    if TABLE_NAME not in tables:
        pytest.skip(f"LanceDB test skipped: table '{TABLE_NAME}' not found")

    table = db.open_table(TABLE_NAME)
    print(table.schema.names)

    filter_ok, columns_ok = supports_filter_and_columns(table)
    print("filter supported:", filter_ok)
    print("columns supported:", columns_ok)