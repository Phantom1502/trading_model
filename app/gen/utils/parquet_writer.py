import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Iterator, List

class ParquetWriterUtil:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.schema = pa.schema([
            ("text",         pa.string()),
            ("source",       pa.string()),
            ("token_length", pa.int64()),
            ("meta",         pa.string()),
        ])

    def run(self, generator: Iterator[List[Dict]]):
        """Pipeline bây giờ chỉ việc nhận batch từ generator và ghi xuống."""
        print(f"Bắt đầu ghi dữ liệu vào: {self.output_path}")
        writer = None
        total = 0

        try:
            for batch in generator:
                table = pa.Table.from_pylist(batch, schema=self.schema)
                if writer is None:
                    writer = pq.ParquetWriter(self.output_path, self.schema, compression='snappy')
                writer.write_table(table)
                total += len(batch)
                print(f"Đã ghi batch {len(batch)} dòng. Tổng cộng: {total}")

            print(f"Thành công! Tổng file có {total} dòng.")
        finally:
            if writer:
                writer.close()