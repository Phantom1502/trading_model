import os
import glob
import pyarrow.parquet as pq
import pyarrow as pa

def merge_parquet_files(input_dir, output_dir, target_size_mb=500):
    """
    Gom các file Parquet nhỏ trong thư mục thành các file lớn ~500MB.
    """
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tìm tất cả các file parquet trong thư mục đầu vào
    # (Hỗ trợ quét cả thư mục con nếu bạn đổi thành '**/*.parquet' và recursive=True)
    search_path = os.path.join(input_dir, "*.parquet")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"Không tìm thấy file Parquet nào trong: {input_dir}")
        return

    print(f"Tìm thấy {len(file_list)} file cần xử lý.")
    
    target_size_bytes = target_size_mb * 1024 * 1024
    current_writer = None
    current_file_size = 0
    file_idx = 1
    schema = None

    # Lấy schema từ file đầu tiên để đảm bảo tính nhất quán
    first_file = pq.ParquetFile(file_list[0])
    schema = first_file.schema.to_arrow_schema()

    def get_next_output_path():
        nonlocal file_idx
        path = os.path.join(output_dir, f"part_{file_idx:04d}.parquet")
        file_idx += 1
        return path

    try:
        for file_path in file_list:
            # Lấy kích thước vật lý của file hiện tại để ước tính
            file_size = os.path.getsize(file_path)
            
            # Nếu file hiện tại cộng vào làm vượt quá 500MB, đóng file cũ để tạo file mới
            if current_writer and (current_file_size + file_size > target_size_bytes):
                current_writer.close()
                current_writer = None
                current_file_size = 0
                print(f"-> Đã đóng file part_{file_idx-1:04d}.parquet (Đạt kích thước mục tiêu)")

            # Khởi tạo writer mới nếu chưa có
            if current_writer == None:
                out_path = get_next_output_path()
                print(f"Đang tạo file mới: {out_path}...")
                # Sử dụng nén ZSTD (Zstandard) - Tối ưu nhất cho Parquet hiện tại về cả tốc độ lẫn tỷ lệ nén
                current_writer = pq.ParquetWriter(out_path, schema, compression='ZSTD')

            # Đọc file nhỏ và ghi trực tiếp vào file lớn (Stream từng file một để tiết kiệm RAM)
            pf = pq.ParquetFile(file_path)
            for rg_idx in range(pf.num_row_groups):
                # Đọc từng row group của file cũ
                row_group = pf.read_row_group(rg_idx)
                # Ghi vào file mới
                current_writer.write_table(row_group)
            
            # Cộng dồn kích thước ước tính
            current_file_size += file_size

    finally:
        # Đóng writer cuối cùng sau khi chạy xong
        if current_writer:
            current_writer.close()
            print(f"-> Đã đóng file cuối cùng.")

    print("\n[Hoàn thành] Dữ liệu đã được gom thành công!")

# --- CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY ---
if __name__ == "__main__":
    INPUT_DIRECTORY = "E:\LLM Dataset\Python_EduClean"       # Thư mục chứa hàng trăm file 40MB ban đầu
    OUTPUT_DIRECTORY = "E:\LLM Dataset\Python_EduClean_merged"   # Thư mục chứa các file ~500MB sau khi gom
    
    merge_parquet_files(
        input_dir=INPUT_DIRECTORY, 
        output_dir=OUTPUT_DIRECTORY, 
        target_size_mb=700
    )