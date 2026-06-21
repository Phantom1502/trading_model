import pandas as pd
import numpy as np

def calculate_atr(df, period=100):
    """Tính chỉ báo ATR 100 chuẩn kỹ thuật"""
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    
    tr = np.max(np.vstack((high_low, high_cp, low_cp)), axis=0)
    # Dùng trung bình động lũy thừa (EMA) để tính ATR cho mượt
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr

if __name__ == "__main__":
    df_history = pd.read_csv("data\\XAUUSD_5Min.csv") 
    
    # 1. Tính ATR 100 cho toàn bộ tập dữ liệu
    df_history['ATR_100'] = calculate_atr(df_history, period=100)
    df_history = df_history.dropna().reset_index(drop=True)

    window_size = 100
    all_max_ratios = []

    # 2. Quét cửa sổ trượt cuốn chiếu toàn bộ lịch sử
    # Vòng lặp này chạy mất vài giây trên tập dữ liệu lớn
    for t in range(window_size - 1, len(df_history)):
        # Trích xuất 100 nến quá khứ tính đến thời điểm t
        window = df_history.iloc[t - window_size + 1 : t + 1]
        
        open_t = df_history.loc[t, 'Open']    # Giá neo số 0 hiện tại
        atr_t = df_history.loc[t, 'ATR_100']  # ATR tại thời điểm t làm thước đo
        
        # Tính khoảng cách thô tuyệt đối của tất cả 400 điểm trong vùng so với open_t
        ohlc_raw = window[['Open', 'High', 'Low', 'Close']].values
        max_absolute_distance = np.max(np.abs(ohlc_raw - open_t))
        
        # Tính tỷ lệ: Cửa sổ này dạt xa gấp mấy lần ATR_t?
        ratio = max_absolute_distance / atr_t
        all_max_ratios.append(ratio)

    # 3. Tìm hằng số SCALE bao trùm tuyệt đối
    # Lấy bách phân vị 99.9% để loại bỏ nhiễu cực đoan của lỗi dữ liệu nếu có
    FINAL_SCALE_FACTOR = np.percentile(all_max_ratios, 99.9)

    print(f"--- KẾT QUẢ PHÂN TÍCH THỐNG KÊ ĐỒ THỊ ---")
    print(f"Tỉ số dạt xa ATR lớn nhất từng xuất hiện trong lịch sử: {max(all_max_ratios):.2f}")
    print(f"=> Cấu hình hằng số SCALE bảo hiểm được chọn: {FINAL_SCALE_FACTOR:.2f}")
