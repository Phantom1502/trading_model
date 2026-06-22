"""
benchmark.py — Đánh giá pretrained model toàn diện
=====================================================
Đo 5 chiều độc lập, tổng hợp thành TOTAL_SCORE có trọng số:

    TOTAL = semantic * 0.20
          + entity   * 0.20
          + fact     * 0.30
          + language * 0.20
          + ood      * 0.10

Mỗi chiều đo một khía cạnh khác nhau của pretrain:

    Semantic  — phân biệt loại thực thể TRONG CÙNG MIỀN (hard negatives)
    Entity    — phân biệt chi tiết (nhà vật lý vs nhà hóa học)
    Fact      — sự kiện cụ thể, không tranh cãi
    Language  — chất lượng sinh văn bản: Distinct-1/2, tỉ lệ lặp
    OOD       — generalization ra ngoài phân phối train (robot, blockchain...)

Cách dùng:
    from benchmark import run_all
    from generate import load_model_for_inference

    model, tokenizer, cfg = load_model_for_inference("checkpoints/chunk_10.pt")
    results = run_all(model, tokenizer, cfg, verbose=True, step=10000)

So sánh nhiều checkpoint:
    from benchmark import compare_checkpoints
    compare_checkpoints(["checkpoints/chunk_10.pt", "checkpoints/chunk_50.pt"])

────────────────────────────────────────────────────────────────────────────
Ghi chú kỹ thuật:

1. Log-prob normalize theo SỐ TOKEN (không phải số từ) — PhoBERT BPE có
   thể cắt 1 từ thành 3-4 subword, không normalize sẽ phạt oan chuỗi dài.

2. Mỗi BenchItem reset M trước khi chạy — context độc lập nhau.

3. Language benchmark dùng sampling temperature=0.8 + top_k=50, giống
   điều kiện inference thật (không dùng greedy để tránh chỉ đo degenerate).
────────────────────────────────────────────────────────────────────────────
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


# ══════════════════════════════════════════════════════════════════════════
# Cấu trúc dữ liệu
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchItem:
    """1 mục log-prob benchmark."""
    prompt   : str
    positive : List[str]
    negative : List[str]
    note     : str = ""


# ══════════════════════════════════════════════════════════════════════════
# Cấp 1: Semantic — hard negatives (cùng siêu miền, khác loại)
#
# Khác với bản cũ dùng negative "quá dễ" (quốc gia, hành tinh),
# bản này dùng negative GẦN NGHĨA để buộc model phải học phân biệt
# tinh tế hơn trong cùng siêu miền.
# ══════════════════════════════════════════════════════════════════════════

SEMANTIC_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Con mèo là",
        positive = ["động vật có vú", "thú nuôi", "sinh vật"],
        negative = ["loài chim", "loài cá", "thực vật", "côn trùng"],
        note     = "hard negative: cùng là sinh vật nhưng khác lớp",
    ),
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà vật lý", "nhà khoa học", "học giả"],
        negative = ["nhà hóa học", "nhà toán học", "nhà triết học", "nhà văn"],
        note     = "hard negative: cùng là học giả nhưng khác ngành",
    ),
    BenchItem(
        prompt   = "Hà Nội là",
        positive = ["thành phố", "thủ đô", "đô thị"],
        negative = ["thị trấn", "làng quê", "tỉnh lẻ", "vùng nông thôn"],
        note     = "hard negative: cùng là địa danh nhưng khác quy mô",
    ),
    BenchItem(
        prompt   = "Python là",
        positive = ["ngôn ngữ lập trình", "công cụ lập trình"],
        negative = ["ngôn ngữ tự nhiên", "ngôn ngữ đánh dấu", "ngôn ngữ truy vấn"],
        note     = "hard negative: cùng là 'ngôn ngữ' nhưng khác loại",
    ),
    BenchItem(
        prompt   = "Sông Hồng là",
        positive = ["con sông", "dòng sông"],
        negative = ["hồ nước", "vịnh biển", "suối nhỏ", "kênh đào"],
        note     = "hard negative: cùng là thủy vực nhưng khác loại",
    ),
    BenchItem(
        prompt   = "Bóng đá là",
        positive = ["môn thể thao", "trò chơi tập thể"],
        negative = ["môn thể thao cá nhân", "trò chơi điện tử", "bộ môn nghệ thuật"],
        note     = "hard negative: cùng là hoạt động giải trí nhưng khác hình thức",
    ),
    BenchItem(
        prompt   = "Mặt Trời là",
        positive = ["ngôi sao", "thiên thể phát sáng"],
        negative = ["hành tinh", "vệ tinh", "sao lùn trắng", "lỗ đen"],
        note     = "hard negative: cùng là thiên thể nhưng khác loại",
    ),
    BenchItem(
        prompt   = "Bác sĩ là",
        positive = ["chuyên gia y tế", "người làm ngành y"],
        negative = ["y tá", "dược sĩ", "kỹ thuật viên xét nghiệm", "hộ lý"],
        note     = "hard negative: cùng là nhân viên y tế nhưng khác vai trò",
    ),
    BenchItem(
        prompt="Con chó là",
        positive=["động vật", "thú nuôi", "sinh vật"],
        negative=["thực vật", "quốc gia", "hành tinh", "ngôn ngữ"],
    ),
    BenchItem(
        prompt="Đại bàng là",
        positive=["loài chim", "động vật", "sinh vật"],
        negative=["loài cá", "thực vật", "quốc gia", "hành tinh"],
    ),
    BenchItem(
        prompt="Cá voi là",
        positive=["động vật có vú", "động vật", "sinh vật"],
        negative=["loài cá", "thực vật", "quốc gia", "núi lửa"],
    ),
    BenchItem(
        prompt="Bác sĩ là",
        positive=["nhân viên y tế", "người lao động", "chuyên gia"],
        negative=["động vật", "quốc gia", "phần mềm", "hành tinh"],
    ),
    BenchItem(
        prompt="Lập trình viên là",
        positive=["người lao động", "kỹ sư", "chuyên gia"],
        negative=["loài chim", "quốc gia", "thực vật", "núi lửa"],
    ),
    BenchItem(
        prompt="Sao Hỏa là",
        positive=["hành tinh", "thiên thể"],
        negative=["ngôi sao", "quốc gia", "động vật", "thực vật"],
    ),
    BenchItem(
        prompt="Mặt Trăng là",
        positive=["vệ tinh", "thiên thể"],
        negative=["hành tinh", "quốc gia", "động vật", "ngôn ngữ"],
    ),
    BenchItem(
        prompt="Con chó là",
        positive=["động vật", "thú nuôi", "sinh vật"],
        negative=["thực vật", "quốc gia", "hành tinh", "ngôn ngữ"],
    ),
    BenchItem(
        prompt="Hoa hồng là",
        positive=["thực vật", "loài hoa", "sinh vật"],
        negative=["động vật", "quốc gia", "phần mềm", "hành tinh"],
    ),
    BenchItem(
        prompt="Xe hơi là",
        positive=["phương tiện", "công cụ", "máy móc"],
        negative=["động vật", "thực vật", "quốc gia", "thiên thể"],
    ),
    BenchItem(
        prompt="Máy bay là",
        positive=["phương tiện", "máy móc", "công nghệ"],
        negative=["động vật", "thực vật", "quốc gia", "núi lửa"],
    ),
    BenchItem(
        prompt="Bệnh viện là",
        positive=["cơ sở y tế", "tổ chức", "nơi làm việc"],
        negative=["động vật", "hành tinh", "thực vật", "loài chim"],
    ),
    BenchItem(
        prompt="Trường học là",
        positive=["cơ sở giáo dục", "tổ chức", "nơi học tập"],
        negative=["động vật", "hành tinh", "quốc gia", "vũ khí"],
    ),
    BenchItem(
        prompt="Internet là",
        positive=["mạng lưới", "công nghệ", "hệ thống"],
        negative=["động vật", "quốc gia", "ngọn núi", "loài cây"],
    ),
    BenchItem(
        prompt="Âm nhạc là",
        positive=["nghệ thuật", "hoạt động sáng tạo", "hình thức biểu đạt"],
        negative=["quốc gia", "hành tinh", "loài chim", "vũ khí"],
    )
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 2: Entity — phân biệt tinh tế trong cùng miền
# Mở rộng từ 5 → 20 item để score ổn định hơn về mặt thống kê
# ══════════════════════════════════════════════════════════════════════════

ENTITY_BENCH: List[BenchItem] = [
    # ── Người nổi tiếng ───────────────────────────────────────────────────
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà vật lý", "nhà khoa học"],
        negative = ["nhà hóa học", "nhà văn", "ca sĩ", "vận động viên"],
    ),
    BenchItem(
        prompt   = "William Shakespeare là",
        positive = ["nhà văn", "nhà thơ", "kịch tác gia"],
        negative = ["nhà khoa học", "nhà chính trị", "nhà thám hiểm", "nhạc sĩ"],
    ),
    BenchItem(
        prompt   = "Marie Curie là",
        positive = ["nhà khoa học", "nhà vật lý", "nhà hóa học"],
        negative = ["nhà văn", "ca sĩ", "diễn viên", "chính trị gia"],
    ),
    BenchItem(
        prompt   = "Hồ Chí Minh là",
        positive = ["chính trị gia", "lãnh tụ", "nhà cách mạng"],
        negative = ["nhà khoa học", "nhà văn", "nhạc sĩ", "vận động viên"],
    ),
    BenchItem(
        prompt="Isaac Newton là",
        positive=["nhà vật lý", "nhà khoa học"],
        negative=["ca sĩ", "nhà văn", "diễn viên", "vận động viên"],
    ),
    BenchItem(
        prompt="Charles Darwin là",
        positive=["nhà sinh học", "nhà khoa học"],
        negative=["ca sĩ", "nhà thơ", "cầu thủ", "diễn viên"],
    ),
    BenchItem(
        prompt="Nikola Tesla là",
        positive=["nhà phát minh", "kỹ sư"],
        negative=["ca sĩ", "nhà thơ", "cầu thủ", "diễn viên"],
    ),
    BenchItem(
        prompt="Galileo Galilei là",
        positive=["nhà thiên văn", "nhà khoa học"],
        negative=["ca sĩ", "diễn viên", "vận động viên", "nhà văn"],
    ),
    BenchItem(
        prompt="Mozart là",
        positive=["nhà soạn nhạc", "nhạc sĩ"],
        negative=["nhà vật lý", "cầu thủ", "bác sĩ", "nhà hóa học"],
    ),
    BenchItem(
        prompt="Beethoven là",
        positive=["nhạc sĩ", "nhà soạn nhạc"],
        negative=["cầu thủ", "nhà vật lý", "ca sĩ", "nhà sinh học"],
    ),
    BenchItem(
        prompt="Leonardo da Vinci là",
        positive=["họa sĩ", "nhà phát minh"],
        negative=["ca sĩ", "vận động viên", "phi hành gia", "bác sĩ"],
    ),
    # ── Địa lý ────────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Hà Nội là thủ đô của",
        positive = ["Việt Nam", "nước Việt Nam"],
        negative = ["Trung Quốc", "Nhật Bản", "Thái Lan", "Campuchia"],
    ),
    BenchItem(
        prompt   = "Tokyo là thủ đô của",
        positive = ["Nhật Bản", "nước Nhật"],
        negative = ["Trung Quốc", "Hàn Quốc", "Thái Lan", "Việt Nam"],
    ),
    BenchItem(
        prompt   = "Sông Mekong chảy qua",
        positive = ["nhiều quốc gia Đông Nam Á", "Việt Nam"],
        negative = ["châu Âu", "châu Phi", "Bắc Mỹ", "Úc"],
    ),
    BenchItem(
        prompt   = "Núi Phú Sĩ nằm ở",
        positive = ["Nhật Bản", "nước Nhật"],
        negative = ["Trung Quốc", "Hàn Quốc", "Việt Nam", "Thái Lan"],
    ),
    BenchItem(
        prompt="Tokyo là thủ đô của",
        positive=["Nhật Bản"],
        negative=["Hàn Quốc", "Trung Quốc", "Thái Lan", "Singapore"],
    ),
    BenchItem(
        prompt="Berlin là thủ đô của",
        positive=["Đức"],
        negative=["Pháp", "Áo", "Ba Lan", "Bỉ"],
    ),
    BenchItem(
        prompt="Thủ đô của Canada là",
        positive=[" Ottawa"],
        negative=[" Toronto", " Montreal", " Vancouver", " Calgary"],
    ),
    BenchItem(
        prompt="Sông dài nhất Việt Nam là",
        positive=[" sông Mekong"],
        negative=[" sông Hồng", " sông Đồng Nai", " sông Đà"],
    ),
    BenchItem(
        prompt="Sydney nằm ở",
        positive=["Úc", "Australia"],
        negative=["Canada", "Brazil", "Ấn Độ", "Nga"],
    ),
    BenchItem(
        prompt="Sông Nile chảy qua",
        positive=["Ai Cập"],
        negative=["Việt Nam", "Nhật Bản", "Hàn Quốc", "Thái Lan"],
    ),
    BenchItem(
        prompt="Tháp Eiffel nằm ở",
        positive=["Pháp", "Paris"],
        negative=["Đức", "Ý", "Tây Ban Nha", "Anh"],
    ),
    # ── Sinh vật ──────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Hổ là loài động vật",
        positive = ["ăn thịt", "nguy hiểm", "thuộc họ mèo lớn"],
        negative = ["ăn cỏ", "bay được", "sống dưới nước", "sống ở Bắc Cực"],
    ),
    BenchItem(
        prompt   = "Cá heo là loài động vật",
        positive = ["có vú", "thông minh", "sống dưới nước"],
        negative = ["bò sát", "lưỡng cư", "côn trùng", "chim"],
    ),
    BenchItem(
        prompt   = "Đại bàng là loài",
        positive = ["chim", "động vật có cánh", "chim săn mồi"],
        negative = ["thú", "bò sát", "cá", "côn trùng"],
    ),
    # ── Công nghệ ─────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Ngôn ngữ Python thường được dùng để",
        positive = ["lập trình", "phân tích dữ liệu", "xây dựng ứng dụng"],
        negative = ["nấu ăn", "leo núi", "chơi thể thao", "vẽ tranh sơn dầu"],
    ),
    BenchItem(
        prompt   = "Trí tuệ nhân tạo là lĩnh vực thuộc",
        positive = ["khoa học máy tính", "công nghệ thông tin"],
        negative = ["y học", "nông nghiệp", "nghệ thuật", "thể thao"],
    ),
    BenchItem(
        prompt   = "Máy tính được dùng để",
        positive = ["xử lý thông tin", "tính toán", "lưu trữ dữ liệu"],
        negative = ["trồng cây", "chữa bệnh", "xây nhà", "nấu ăn"],
    ),
    # ── Thiên văn ────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Mặt Trăng là",
        positive = ["vệ tinh của Trái Đất", "thiên thể"],
        negative = ["ngôi sao", "hành tinh", "tiểu hành tinh tự do", "sao chổi"],
    ),
    BenchItem(
        prompt   = "Hành tinh Sao Hỏa có màu",
        positive = ["đỏ", "đỏ cam"],
        negative = ["xanh lam", "vàng", "trắng", "đen"],
    ),
    # ── Lịch sử / văn hóa ────────────────────────────────────────────────
    BenchItem(
        prompt   = "Chiến tranh thế giới thứ hai kết thúc vào năm",
        positive = ["1945"],
        negative = ["1918", "1939", "1950", "1975"],
    ),
    BenchItem(
        prompt   = "Kim tự tháp Giza nằm ở",
        positive = ["Ai Cập", "Bắc Phi"],
        negative = ["Hy Lạp", "Iraq", "Iran", "Ấn Độ"],
    ),
    # ── Khoa học ─────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Nước sôi ở nhiệt độ",
        positive = ["100 độ Celsius", "100°C"],
        negative = ["0 độ Celsius", "37 độ Celsius", "200 độ Celsius"],
    ),
    BenchItem(
        prompt   = "Quang hợp là quá trình",
        positive = ["thực vật tổng hợp chất hữu cơ từ ánh sáng", "chuyển hóa năng lượng ánh sáng"],
        negative = ["động vật tiêu hóa thức ăn", "vi khuẩn phân hủy chất hữu cơ",
                    "con người hít thở oxygen"],
    ),
    BenchItem(
        prompt="Ký hiệu hóa học của bạc là",
        positive=[" Ag"],
        negative=[" Au", " Fe", " Cu", " Zn"],
    ),
    BenchItem(
        prompt="Nguyên tố có số hiệu nguyên tử 1 là",
        positive=[" Hydro"],
        negative=[" Oxy", " Heli", " Carbon", " Nitơ"],
    )
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 3: Fact — sự kiện không tranh cãi
# Loại bỏ các fact còn tranh luận (sông dài nhất thế giới: Nile vs Amazon)
# ══════════════════════════════════════════════════════════════════════════

FACT_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Thủ đô của Việt Nam là",
        positive = [" Hà Nội"],
        negative = [" Thành phố Hồ Chí Minh", " Đà Nẵng", " Huế", " Cần Thơ"],
        note     = "khoảng trắng đầu để ghép sát prompt, tránh BPE tokenize khác",
    ),
    BenchItem(
        prompt   = "Thủ đô của Pháp là",
        positive = [" Paris"],
        negative = [" London", " Berlin", " Rome", " Madrid"],
    ),
    BenchItem(
        prompt   = "Hóa học, ký hiệu nguyên tố của vàng là",
        positive = [" Au"],
        negative = [" Ag", " Fe", " Cu", " Pt"],
    ),
    BenchItem(
        prompt   = "Hành tinh thứ ba tính từ Mặt Trời là",
        positive = [" Trái Đất"],
        negative = [" Sao Hỏa", " Sao Kim", " Sao Mộc", " Sao Thủy"],
    ),
    BenchItem(
        prompt   = "Nước láng giềng phía Bắc của Việt Nam là",
        positive = [" Trung Quốc"],
        negative = [" Lào", " Campuchia", " Thái Lan", " Myanmar"],
    ),
    BenchItem(
        prompt   = "Ngôn ngữ lập trình phổ biến nhất để phân tích dữ liệu là",
        positive = [" Python"],
        negative = [" Java", " C++", " JavaScript", " Ruby"],
    ),
    BenchItem(
        prompt   = "Đơn vị đo nhiệt độ trong hệ SI là",
        positive = [" Kelvin"],
        negative = [" Celsius", " Fahrenheit", " Rankine"],
    ),
    BenchItem(
        prompt   = "Hành tinh lớn nhất trong hệ Mặt Trời là",
        positive = [" Sao Mộc"],
        negative = [" Sao Thổ", " Sao Hải Vương", " Trái Đất", " Sao Hỏa"],
    ),
    BenchItem(
        prompt   = "Con người thuộc loài linh trưởng",
        positive = [" đúng", " chính xác"],
        negative = [" sai", " không phải", " nhầm"],
    ),
    BenchItem(
        prompt   = "Tốc độ ánh sáng trong chân không xấp xỉ",
        positive = [" 300.000 km/s", " 3×10⁸ m/s"],
        negative = [" 1.000 km/s", " 30.000 km/s", " 3.000.000 km/s"],
    ),
    BenchItem(
        prompt="Hành tinh gần Mặt Trời nhất là",
        positive=[" Sao Thủy"],
        negative=[" Sao Kim", " Trái Đất", " Sao Hỏa"],
    ),
    BenchItem(
        prompt="DNA viết tắt của",
        positive=[" Deoxyribonucleic Acid"],
        negative=[" Digital Network Access",
                " Dynamic Numeric Array",
                " Distributed Neural Architecture"],
    ),
    BenchItem(
        prompt="Kim loại lỏng ở nhiệt độ phòng là",
        positive=[" Thủy ngân"],
        negative=[" Sắt", " Đồng", " Nhôm", " Bạc"],
    ),
    BenchItem(
        prompt="Trái Đất quay quanh",
        positive=[" Mặt Trời"],
        negative=[" Mặt Trăng", " Sao Hỏa", " Sao Mộc"],
    ),
    BenchItem(
        prompt="Đơn vị cơ bản đo khối lượng trong SI là",
        positive=[" kilogram"],
        negative=[" gram", " pound", " ounce"],
    ),
    BenchItem(
        prompt="Số nguyên tố nhỏ nhất là",
        positive=[" 2"],
        negative=[" 1", " 3", " 5", " 7"],
    )
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 4: Language Quality — Distinct-1/2 và repetition rate
#
# Đo chất lượng sinh văn bản: model có bị mắc kẹt trong vòng lặp cụm từ
# hay không. Đây là vấn đề rất phổ biến ở giai đoạn đầu pretrain (thường
# thấy repeat_ratio cao ở 20-40k step, giảm dần khi train tiếp).
#
# Distinct-1/2 là số lượng unigram/bigram unique chia cho tổng số token —
# giá trị cao = đa dạng từ vựng và cấu trúc, thấp = lặp lại.
# ══════════════════════════════════════════════════════════════════════════

LANGUAGE_PROMPTS = [
    "Con mèo là",
    "Albert Einstein là",
    "Trí tuệ nhân tạo là",
    "Máy tính là",
    "Hà Nội là thành phố",
    "Sông Hồng chảy qua",
    "Bóng đá là môn thể thao",
    "Khoa học máy tính là",
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 5: OOD — generalization ra ngoài phân phối train
#
# Nếu Wikipedia tiếng Việt ít đề cập blockchain, robot, NFT... thì đây
# là thử thách đo khả năng generalize thay vì chỉ memorize.
# Dùng cùng kiểu log-prob nhưng với các concept ít xuất hiện hơn.
# ══════════════════════════════════════════════════════════════════════════

OOD_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Robot là",
        positive = ["máy móc tự động", "thiết bị cơ điện tử", "máy móc thông minh"],
        negative = ["sinh vật sống", "loài động vật", "thực vật", "khoáng vật"],
        note     = "OOD: công nghệ hiện đại, ít xuất hiện trong Wikipedia cũ",
    ),
    BenchItem(
        prompt   = "Blockchain là",
        positive = ["công nghệ lưu trữ dữ liệu", "chuỗi khối dữ liệu", "công nghệ phi tập trung"],
        negative = ["loài động vật", "địa danh", "môn thể thao", "nhân vật lịch sử"],
    ),
    BenchItem(
        prompt   = "Đàn guitar là",
        positive = ["nhạc cụ", "nhạc cụ có dây", "công cụ âm nhạc"],
        negative = ["loài chim", "vũ khí", "phương tiện giao thông", "loại thực phẩm"],
    ),
    BenchItem(
        prompt   = "Vaccine là",
        positive = ["chế phẩm sinh học", "biện pháp phòng bệnh", "thuốc phòng ngừa"],
        negative = ["loại thực phẩm", "loại máy móc", "loại vũ khí", "dụng cụ thể thao"],
    ),
    BenchItem(
        prompt   = "Năng lượng mặt trời là",
        positive = ["nguồn năng lượng tái tạo", "nguồn năng lượng sạch", "năng lượng từ ánh sáng"],
        negative = ["loại nhiên liệu hóa thạch", "loại thực phẩm", "loại khoáng sản", "vũ khí"],
    ),
    BenchItem(
        prompt   = "Mạng xã hội là",
        positive = ["nền tảng kết nối trực tuyến", "công cụ giao tiếp số", "dịch vụ internet"],
        negative = ["mạng lưới điện", "mạng giao thông", "loài sinh vật", "loại thực phẩm"],
    ),
    BenchItem(
        prompt="Zorb là một loài động vật. Mọi động vật đều là sinh vật. Zorb là",
        positive=["sinh vật"],
        negative=["quốc gia", "thành phố", "hành tinh"],
    ),
    BenchItem(
        prompt="Quark là một loại hạt cơ bản. Mọi hạt cơ bản đều là vật chất. Quark là",
        positive=["vật chất"],
        negative=["năng lượng", "sinh vật", "ngôn ngữ"],
    ),
    BenchItem(
        prompt="An là bác sĩ. Mọi bác sĩ đều là nhân viên y tế. Mọi nhân viên y tế đều là người lao động. An là",
        positive=["người lao động"],
        negative=["động vật", "thực vật", "hành tinh"],
    ),
    BenchItem(
        prompt="Blen là một loại flar. Mọi flar đều là zent. Mọi zent đều là sinh vật. Blen là",
        positive=["sinh vật"],
        negative=["quốc gia", "phần mềm", "thiên hà"],
    ),
    BenchItem(
        prompt="""
    Loma là một loài động vật.
    Mọi động vật đều là sinh vật.

    Loma là
    """,
        positive=["sinh vật"],
        negative=["quốc gia","hành tinh","phần mềm"],
    ),
    BenchItem(
        prompt="""
    Kira là bác sĩ.
    Mọi bác sĩ đều làm việc trong ngành y tế.

    Kira thuộc
    """,
        positive=["ngành y tế"],
        negative=["ngành nông nghiệp","thể thao","hàng không"],
    ),
    BenchItem(
        prompt="""
    Mọi flar đều là zent.
    Mọi zent đều là sinh vật.
    Blen là một flar.

    Blen là
    """,
        positive=["sinh vật"],
        negative=["quốc gia","hành tinh","thực vật"],
    ),
    BenchItem(
        prompt="""
    Mọi nori đều là phương tiện.
    Mọi phương tiện đều được dùng để di chuyển.
    Teka là một nori.

    Teka được dùng để
    """,
        positive=["di chuyển"],
        negative=["quang hợp","săn mồi","bay vào vũ trụ"],
    ),
    BenchItem(
        prompt="""
    Mọi vark đều là trilo.
    Mọi trilo đều là zent.
    Mọi zent đều là sinh vật.

    Peko là một vark.

    Peko là
    """,
        positive=["sinh vật"],
        negative=["quốc gia","phần mềm","ngọn núi"],
    ),
    BenchItem(
        prompt="""
    Mọi drako đều là máy móc.
    Mọi máy móc đều là công cụ.
    Mọi công cụ đều phục vụ một mục đích sử dụng.

    Lena là một drako.

    Lena là
    """,
        positive=["công cụ"],
        negative=["động vật","thực vật","hành tinh"],
    )
]


# ══════════════════════════════════════════════════════════════════════════
# Core: log-prob scoring
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def avg_logprob_per_token(
    model,
    tokenizer,
    prompt    : str,
    completion: str,
    device    : torch.device,
    max_seq   : int = 512,
) -> float:
    """
    Tính log-prob trung bình trên MỖI TOKEN của `completion` cho trước `prompt`.

    Normalize theo số token (không phải số từ) — quan trọng với PhoBERT BPE.
    Returns: giá trị trong (-inf, 0], càng gần 0 xác suất càng cao.
    """
    from model import causal_mask

    prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    if not completion_ids:
        return float("-inf")

    full_ids = prompt_ids + completion_ids
    if len(full_ids) > max_seq:
        keep = max_seq - len(completion_ids)
        if keep <= 0:
            return float("-inf")
        full_ids = prompt_ids[-keep:] + completion_ids

    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    T     = ids_t.size(1)
    mask  = causal_mask(T, device)

    logits    = model(ids_t, attn_mask=mask)          # (1, T, vocab)
    log_probs = F.log_softmax(logits[0], dim=-1)      # (T, vocab)

    n_prompt  = T - len(completion_ids)
    total_lp  = 0.0
    for i, tok_id in enumerate(completion_ids):
        pos = n_prompt + i - 1
        if pos < 0:
            continue
        total_lp += log_probs[pos, tok_id].item()

    return total_lp / len(completion_ids)


@torch.no_grad()
def score_item(
    model,
    tokenizer,
    item   : BenchItem,
    device : torch.device,
    max_seq: int = 512,
) -> dict:
    pos_scores = [
        avg_logprob_per_token(model, tokenizer, item.prompt, p, device, max_seq)
        for p in item.positive
    ]
    neg_scores = [
        avg_logprob_per_token(model, tokenizer, item.prompt, n, device, max_seq)
        for n in item.negative
    ]
    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    return {
        "prompt"    : item.prompt,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "pos_mean"  : pos_mean,
        "neg_mean"  : neg_mean,
        "score"     : pos_mean - neg_mean,
    }


# ══════════════════════════════════════════════════════════════════════════
# Language quality benchmark
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_greedy_sample(
    model,
    tokenizer,
    prompt     : str,
    max_new    : int   = 50,
    temperature: float = 0.8,
    top_k      : int   = 50,
    device     : torch.device = None,
) -> list[int]:
    """Sinh 1 mẫu token bằng sampling (temperature + top-k)."""
    device = device or next(model.parameters()).device

    ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long, device=device,
    )

    from model import causal_mask
    for _ in range(max_new):
        T    = ids.size(1)
        mask = causal_mask(T, device)
        logits = model(ids, attn_mask=mask)

        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < v[:, -1:], float("-inf"))

        probs   = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids     = torch.cat([ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_id:
            break

    return ids[0].tolist()


def _compute_language_metrics(token_seqs: list[list[int]]) -> dict:
    """
    Tính Distinct-1, Distinct-2 và repeat_ratio từ nhiều câu sinh ra.

    Distinct-n = số n-gram unique / tổng số n-gram.
    repeat_ratio = tỉ lệ token lặp lại liên tiếp (token[i] == token[i-1]).
    """
    all_tokens    = []
    all_bigrams   = []
    repeat_count  = 0
    total_tokens  = 0

    for seq in token_seqs:
        all_tokens.extend(seq)
        total_tokens += len(seq)

        for i in range(len(seq) - 1):
            all_bigrams.append((seq[i], seq[i + 1]))
            if seq[i] == seq[i + 1]:
                repeat_count += 1

    distinct1 = len(set(all_tokens)) / max(len(all_tokens), 1)
    distinct2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    repeat    = repeat_count / max(total_tokens - 1, 1)

    return {
        "distinct1"   : distinct1,
        "distinct2"   : distinct2,
        "repeat_ratio": repeat,
        # language_score: kết hợp tuyến tính để dùng trong TOTAL_SCORE
        # Distinct cao → tốt, repeat cao → xấu
        "language_score": (distinct1 + distinct2) / 2 - repeat,
    }


def run_language_benchmark(
    model,
    tokenizer,
    prompts   : list[str] = None,
    n_samples : int = 5,
    max_new   : int = 50,
    device    : torch.device = None,
    verbose   : bool = True,
) -> dict:
    """
    Chạy language quality benchmark:
        - Với mỗi prompt, sinh n_samples mẫu
        - Tổng hợp Distinct-1/2 và repeat_ratio trên tất cả mẫu

    language_score = (distinct1 + distinct2) / 2 - repeat_ratio
    Giá trị càng cao = sinh văn bản càng đa dạng, ít lặp.
    """
    model.eval()
    prompts = prompts or LANGUAGE_PROMPTS
    device  = device or next(model.parameters()).device

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  [LANGUAGE QUALITY]")
        print(f"{'─'*60}")

    all_seqs = []
    for prompt in prompts:
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=1, device=device)

        for _ in range(n_samples):
            seq = _generate_greedy_sample(
                model, tokenizer, prompt,
                max_new=max_new, device=device,
            )
            # Chỉ lấy phần sinh ra (bỏ phần prompt)
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            all_seqs.append(seq[prompt_len:])

    metrics = _compute_language_metrics(all_seqs)

    if verbose:
        print(f"\n  distinct-1   : {metrics['distinct1']:.4f}  (1.0 = tất cả token đều unique)")
        print(f"  distinct-2   : {metrics['distinct2']:.4f}  (1.0 = tất cả bigram đều unique)")
        print(f"  repeat_ratio : {metrics['repeat_ratio']:.4f}  (0.0 = không lặp token liên tiếp)")
        print(f"\n  ► language_score = {metrics['language_score']:+.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Log-prob benchmark runner chung
# ══════════════════════════════════════════════════════════════════════════

def run_logprob_benchmark(
    model,
    tokenizer,
    bench     : List[BenchItem],
    level_name: str,
    device    : torch.device,
    max_seq   : int = 512,
    verbose   : bool = True,
) -> float:
    """Chạy 1 cấp log-prob benchmark. Trả về score trung bình."""
    model.eval()
    scores = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  [{level_name.upper()}]")
        print(f"{'─'*60}")

    for item in bench:
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=1, device=device)

        r = score_item(model, tokenizer, item, device, max_seq)
        scores.append(r["score"])

        if verbose:
            pos_fmt = "  ".join(f"{s:+.2f}" for s in r["pos_scores"])
            neg_fmt = "  ".join(f"{s:+.2f}" for s in r["neg_scores"])
            print(f"\n  prompt : {item.prompt!r}")
            print(f"  pos_lp : [{pos_fmt}]  mean={r['pos_mean']:+.3f}")
            print(f"  neg_lp : [{neg_fmt}]  mean={r['neg_mean']:+.3f}")
            status = "✓" if r["score"] > 0 else "✗"
            print(f"  score  : {r['score']:+.3f}  {status}")

    avg    = sum(scores) / len(scores)
    n_pass = sum(1 for s in scores if s > 0)

    if verbose:
        print(f"\n  ► {level_name} avg_score = {avg:+.3f}  |  pass = {n_pass}/{len(scores)}")

    return avg


# ══════════════════════════════════════════════════════════════════════════
# run_all — chạy đủ 5 chiều và tổng hợp TOTAL_SCORE
# ══════════════════════════════════════════════════════════════════════════

# Trọng số theo đề xuất: Fact quan trọng nhất (0.30), Language & Semantic cân bằng (0.20)
WEIGHTS = {
    "semantic": 0.20,
    "entity"  : 0.20,
    "fact"    : 0.30,
    "language": 0.20,
    "ood"     : 0.10,
}


def run_all(
    model,
    tokenizer,
    cfg,
    verbose         : bool = True,
    step            : int  = None,
    n_language_samples: int = 5,
) -> dict:
    """
    Chạy toàn bộ 5 cấp benchmark, trả về dict kết quả + TOTAL_SCORE.

    TOTAL_SCORE = weighted sum theo WEIGHTS (xem đầu file).

    Lưu ý về language_score: giá trị có thể âm khi model còn lặp nhiều
    (repeat_ratio cao hơn average distinct). Đây là hành vi đúng — tín hiệu
    cho thấy model chưa học được đa dạng ngôn ngữ.

    Args:
        step              : global_step hiện tại (chỉ để in log)
        n_language_samples: số mẫu sinh cho mỗi prompt language benchmark
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq

    step_str = f"step {step}" if step is not None else "checkpoint"
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  BENCHMARK  —  {step_str}")
        print(f"{'═'*60}")

    sem = run_logprob_benchmark(model, tokenizer, SEMANTIC_BENCH, "semantic", device, max_seq, verbose)
    ent = run_logprob_benchmark(model, tokenizer, ENTITY_BENCH,   "entity",   device, max_seq, verbose)
    fct = run_logprob_benchmark(model, tokenizer, FACT_BENCH,     "fact",     device, max_seq, verbose)
    ood = run_logprob_benchmark(model, tokenizer, OOD_BENCH,      "ood",      device, max_seq, verbose)

    lang_metrics = run_language_benchmark(
        model, tokenizer,
        n_samples=n_language_samples,
        device=device,
        verbose=verbose,
    )
    lang = lang_metrics["language_score"]

    total = (
        sem  * WEIGHTS["semantic"] +
        ent  * WEIGHTS["entity"]   +
        fct  * WEIGHTS["fact"]     +
        lang * WEIGHTS["language"] +
        ood  * WEIGHTS["ood"]
    )

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  SUMMARY  ({step_str})")
        print(f"{'─'*60}")
        print(f"  semantic  (×{WEIGHTS['semantic']:.2f}) : {sem:+.3f}")
        print(f"  entity    (×{WEIGHTS['entity']:.2f}) : {ent:+.3f}")
        print(f"  fact      (×{WEIGHTS['fact']:.2f}) : {fct:+.3f}")
        print(f"  language  (×{WEIGHTS['language']:.2f}) : {lang:+.4f}  "
              f"[d1={lang_metrics['distinct1']:.3f} d2={lang_metrics['distinct2']:.3f} "
              f"rep={lang_metrics['repeat_ratio']:.3f}]")
        print(f"  ood       (×{WEIGHTS['ood']:.2f}) : {ood:+.3f}")
        print(f"{'─'*60}")
        print(f"  TOTAL               : {total:+.3f}")
        print(f"{'═'*60}\n")

    return {
        "semantic"    : sem,
        "entity"      : ent,
        "fact"        : fct,
        "language"    : lang,
        "ood"         : ood,
        "total"       : total,
        "distinct1"   : lang_metrics["distinct1"],
        "distinct2"   : lang_metrics["distinct2"],
        "repeat_ratio": lang_metrics["repeat_ratio"],
    }


# ══════════════════════════════════════════════════════════════════════════
# So sánh nhiều checkpoint
# ══════════════════════════════════════════════════════════════════════════

def compare_checkpoints(checkpoint_paths: list[str], verbose: bool = False) -> None:
    """
    Load từng checkpoint, chạy benchmark, in bảng so sánh.

    Usage:
        compare_checkpoints([
            "checkpoints/chunk_10.pt",
            "checkpoints/chunk_30.pt",
            "checkpoints/chunk_50.pt",
        ])
    """
    from generate import load_model_for_inference

    rows = []
    for path in checkpoint_paths:
        print(f"\n── Loading {path} ──")
        model, tokenizer, cfg = load_model_for_inference(path)
        r = run_all(model, tokenizer, cfg, verbose=verbose)
        r["checkpoint"] = path
        rows.append(r)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # In bảng tổng hợp
    print(f"\n{'═'*88}")
    print(f"  {'CHECKPOINT':<30} {'sem':>6} {'ent':>6} {'fact':>6} "
          f"{'lang':>6} {'ood':>6} {'total':>8}")
    print(f"{'─'*88}")
    for r in rows:
        name = r["checkpoint"].split("/")[-1]
        print(
            f"  {name:<30} {r['semantic']:>+6.2f} {r['entity']:>+6.2f} "
            f"{r['fact']:>+6.2f} {r['language']:>+6.3f} "
            f"{r['ood']:>+6.2f} {r['total']:>+8.3f}"
        )
    print(f"{'═'*88}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    python benchmark.py checkpoints/chunk_10.pt
    python benchmark.py checkpoints/chunk_10.pt checkpoints/chunk_50.pt
    """
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python benchmark.py <checkpoint> [checkpoint2 ...]")
        sys.exit(1)

    if len(paths) == 1:
        from generate import load_model_for_inference
        model, tokenizer, cfg = load_model_for_inference(paths[0])
        run_all(model, tokenizer, cfg, verbose=True)
    else:
        compare_checkpoints(paths, verbose=False)