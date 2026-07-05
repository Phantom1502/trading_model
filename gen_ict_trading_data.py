import json
from typing import List

from app.utils.chart.data_pipeline import ChartCodec, ChartDatasetBuilder, ActionDataGen, M1_SCALE, D1_SCALE
from app.utils.chart.chart_pretrain_pipeline import ChartPretrainPipeline
from app.memlm.tokenizer import VietnameseTokenizer
from app.utils.chart.book_and_output_pipeline import BookPipeline, OutputPipeline
from app.ict.render import render_all_samples
import random
import pandas as pd

def gen_action_data(
    df_path     : str,
    output_path : str,
    tokenizer   : VietnameseTokenizer,
    scale       : float     = M1_SCALE,
    window_size : int       = 20,
    forward_size: int       = 60,
    stride      : int       = 10,
    sl_bins     : list      = None,
    tp_bins     : list      = None,
    seed        : int       = 42,
) -> list[str]:
    df    = pd.read_csv(df_path)
    codec = ChartCodec(scale=scale)
    gen   = ActionDataGen(
        codec,
        window_size  = window_size,
        forward_size = forward_size,
        sl_bins      = sl_bins or [-20, -40, -60, -80],
        tp_bins      = tp_bins or [+40, +80, +120, +160],
    )

    gen.build_to_parquet(
        df          = df,
        tokenizer   = tokenizer,
        output_path = output_path,
        source_name = "action_data",
        stride      = stride,
    )

def gen_base_trading_data(
    tokenizer   : VietnameseTokenizer,
    source_name : str,
    input_parquet_path  : str,
    output_path : str,
):
    pipeline = ChartPretrainPipeline(
        tokenizer, 
        source_name=source_name
    )
    pipeline.build_from_parquet(
        input_path  = input_parquet_path,
        output_path = output_path,
    )
    
def build_raw_dataset(
    input_ohlc: str,
    output_path: str,
    scale: float = M1_SCALE,
    window_size: int = 100,
    atr_period: int = 100,
    stride: int = 10,
) -> None:
    codec = ChartCodec(scale=scale)
    
    builder = ChartDatasetBuilder(
        codec, 
        window_size=window_size, 
        stride=stride, 
        atr_period=atr_period
    )
    builder.build_from_file(input_ohlc, output_path)

from app.ict.candle import Candle

def _c(o, h, l, c):
    return Candle(open=o, high=h, low=l, close=c)
    
from app.ict.facts import build_facts
    
def _build_facts_and_raw(candles, initial_trend="BULL"):
    parser = CandleParser.from_candles(candles, swing_window=2)
    facts = build_facts(parser, initial_trend=initial_trend, lookback=10)
    raw_chart_text = parser.raw_text
    return facts, raw_chart_text

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

def _parquet_schema():
    return pa.schema([
        ("text",         pa.string()),
        ("source",       pa.string()),
        ("token_length", pa.int64()),
        ("meta",         pa.string()),
    ])
    
from app.ict.parser import CandleParser

def gen_records(
    raw_charts          : List[str],
    tokenizer           : VietnameseTokenizer,
    chart_index_offset  : int = 0,
) -> List[dict]:
    records = []

    for local_idx, raw in enumerate(raw_charts):
        chart_idx   = chart_index_offset + local_idx
        base_parser = CandleParser(raw, swing_window=2) # TODO: swing_window=2 có thể thay đổi nếu muốn, nhưng hiện tại giữ nguyên để consistent với các test và render sample.
        n           = len(base_parser)

        # Cắt đoạn con
        if n < 20: # TODO: có thể thay đổi nếu muốn, nhưng hiện tại giữ nguyên để consistent với các test và render sample.
            sub_ranges = [(0, n)]
        else:
            sub_ranges = []
            for _ in range(4): # TODO: có thể thay đổi nиф muốn, nhưng hiện tại giữ nguyên để consistent với các test và render sample.
                #sl  = random.randint(20, min(30, n)) # TODO: có thể thay đổi nếu muốn, nhưng hiện tại giữ nguyên để consistent với các test và render sample.
                #st  = random.randint(0, n - sl)
                #sub_ranges.append((st, st + sl))
                # fix cố định window size 20 để consistent với các test và render sample.
                st  = random.randint(0, n - 20)
                sub_ranges.append((st, st + 20))
                
        for start, end in sub_ranges:
            sub_parser = base_parser.slice(start, end)

            for type in ["BULL", "BEAR"]:
                facts, raw_chart_text = _build_facts_and_raw(sub_parser.candles, initial_trend=type)
                
                samples = render_all_samples(facts, raw_chart_text, rng=random.Random(42))
                if len(samples) == 0:
                    continue
                
                for sample in samples:
                    text         = sample["text"]
                    token_length = len(tokenizer.encode(text))
                    meta         = {
                        "chart_index" : chart_idx,
                        "sub_range"   : [start, end],
                        "sample_type" : sample["request"],
                    }
                    
                    records.append({
                        "text"        : text,
                        "source"      : sample["request"],
                        "token_length": token_length,
                        "meta"        : json.dumps(meta, ensure_ascii=False),
                    })

    return records
    
def build_from_parquet(
        input_path : str,
        output_path: str,
        tokenizer  : VietnameseTokenizer,
        text_column: str = "text",
        batch_size : int = 2000,
    ) -> int:
        """
        Đọc parquet chart token theo batch → sinh text curriculum → ghi parquet.
        Input thường là output của ChartDatasetBuilder (nhánh 2).
        """
        if not _HAS_PYARROW:
            raise ImportError("Cần cài pyarrow: pip install pyarrow")

        schema = _parquet_schema()
        writer = None
        total  = 0

        try:
            pf = pq.ParquetFile(input_path)
            for batch_idx, batch in enumerate(
                pf.iter_batches(batch_size=batch_size, columns=[text_column])
            ):
                raw_charts = batch.column(text_column).to_pylist()
                records    = gen_records(raw_charts, tokenizer,chart_index_offset=batch_idx * batch_size)

                if not records:
                    continue

                table = pa.Table.from_pylist(records, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                writer.write_table(table)

                total += len(records)
                print(f"[batch {batch_idx}] +{len(records)} samples | tổng: {total}")

        finally:
            if writer:
                writer.close()

        print(f"\n✅ Hoàn tất. {total} samples -> {output_path}")
        return total
    
if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "app", "memlm", "custom_tokenizer")
    print(f"Testing tokenizer: {tok_path}\n")

    tokenizer = VietnameseTokenizer(pretrained_name=tok_path)
    
    raw_text_path = "data/chart_XAUUSD_dataset_1Min.parquet"
    build_from_parquet(
        input_path  = raw_text_path,
        output_path = "data/chart_XAUUSD_dataset_1Min_W20_samples.parquet",
        tokenizer   = tokenizer,
        text_column = "text",
        batch_size  = 2000,
    )
    
    # read output parquet to verify
    import pyarrow.parquet as pq
    table = pq.read_table("data/chart_XAUUSD_dataset_1Min_W20_samples.parquet")
    df = table.to_pandas()

    print(df.iloc[100])