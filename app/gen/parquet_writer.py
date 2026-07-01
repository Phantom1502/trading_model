import pyarrow as pa
import pyarrow.parquet as pq
from typing import Iterator, Optional

def _build_schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])

def write_to_parquet(iter: Iterator, output_parquet: str):
    schema = _build_schema()
    writer: Optional[pq.ParquetWriter] = None
    
    try:
        for row_idx, rows in enumerate(iter):
            table = pa.Table.from_pylist(rows, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(output_parquet, schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()