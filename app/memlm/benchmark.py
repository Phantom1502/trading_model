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
    Entity    — phân biệt chi tiết trong cùng miền (nhà vật lý vs nhà hóa học)
    Fact      — sự kiện cụ thể không tranh cãi
    Language  — chất lượng sinh văn bản: Distinct-1/2, tỉ lệ lặp
    OOD       — generalization ra ngoài phân phối train

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
# Negative phải là "gần nghĩa" — cùng siêu miền nhưng khác loại cụ thể.
# Ví dụ: "Con mèo" → negative là "loài chim/cá" (không phải "quốc gia").
# Bộ này 15 item, phủ 15 miền khác nhau: không item nào trùng chủ đề.
# ══════════════════════════════════════════════════════════════════════════

SEMANTIC_BENCH: List[BenchItem] = [
    BenchItem(
        prompt   = "Con mèo là",
        positive = ["động vật có vú", "thú nuôi", "sinh vật"],
        negative = ["loài chim", "loài cá", "thực vật", "côn trùng"],
        note     = "sinh vật — hard negative: cùng giới động vật, khác lớp",
    ),
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà vật lý", "nhà khoa học", "học giả"],
        negative = ["nhà hóa học", "nhà toán học", "nhà triết học", "nhà văn"],
        note     = "người — hard negative: cùng là học giả, khác ngành",
    ),
    BenchItem(
        prompt   = "Hà Nội là",
        positive = ["thành phố", "thủ đô", "đô thị lớn"],
        negative = ["thị trấn", "làng quê", "vùng nông thôn", "tỉnh lẻ"],
        note     = "địa danh — hard negative: cùng là địa danh, khác quy mô",
    ),
    BenchItem(
        prompt   = "Python là",
        positive = ["ngôn ngữ lập trình", "công cụ lập trình"],
        negative = ["ngôn ngữ tự nhiên", "ngôn ngữ đánh dấu", "ngôn ngữ truy vấn"],
        note     = "công nghệ — hard negative: cùng là 'ngôn ngữ', khác loại",
    ),
    BenchItem(
        prompt   = "Sông Hồng là",
        positive = ["con sông", "dòng sông"],
        negative = ["hồ nước", "vịnh biển", "suối nhỏ", "kênh đào"],
        note     = "địa lý — hard negative: cùng là thủy vực, khác loại",
    ),
    BenchItem(
        prompt   = "Bóng đá là",
        positive = ["môn thể thao tập thể", "trò chơi vận động"],
        negative = ["môn thể thao cá nhân", "trò chơi điện tử", "bộ môn nghệ thuật"],
        note     = "thể thao — hard negative: cùng là hoạt động giải trí, khác hình thức",
    ),
    BenchItem(
        prompt   = "Mặt Trời là",
        positive = ["ngôi sao", "thiên thể phát sáng"],
        negative = ["hành tinh", "vệ tinh", "sao lùn trắng", "lỗ đen"],
        note     = "thiên văn — hard negative: cùng là thiên thể, khác loại",
    ),
    BenchItem(
        prompt   = "Bác sĩ là",
        positive = ["chuyên gia y tế", "người hành nghề y"],
        negative = ["y tá", "dược sĩ", "kỹ thuật viên xét nghiệm", "hộ lý"],
        note     = "nghề nghiệp — hard negative: cùng là nhân viên y tế, khác vai trò",
    ),
    BenchItem(
        prompt   = "Piano là",
        positive = ["nhạc cụ", "nhạc cụ có phím"],
        negative = ["nhạc cụ có dây", "nhạc cụ hơi", "nhạc cụ gõ màng", "nhạc cụ điện tử"],
        note     = "âm nhạc — hard negative: cùng là nhạc cụ, khác cơ chế phát âm",
    ),
    BenchItem(
        prompt   = "Tiểu thuyết là",
        positive = ["tác phẩm văn học", "thể loại văn xuôi dài"],
        negative = ["truyện ngắn", "bài thơ", "kịch bản", "tản văn"],
        note     = "văn học — hard negative: cùng là thể loại văn học, khác hình thức",
    ),
    BenchItem(
        prompt   = "Luật sư là",
        positive = ["người hành nghề pháp lý", "chuyên gia pháp luật"],
        negative = ["thẩm phán", "công tố viên", "thư ký tòa án", "cảnh sát"],
        note     = "pháp lý — hard negative: cùng là người trong hệ thống tư pháp, khác vai trò",
    ),
    BenchItem(
        prompt   = "Muỗi là",
        positive = ["côn trùng", "động vật chân đốt"],
        negative = ["động vật có vú", "loài chim", "bò sát", "động vật thân mềm"],
        note     = "sinh vật — hard negative: cùng là động vật không xương, khác ngành",
    ),
    BenchItem(
        prompt   = "Đái tháo đường là",
        positive = ["bệnh rối loạn chuyển hóa", "bệnh mãn tính"],
        negative = ["bệnh truyền nhiễm", "bệnh ung thư", "bệnh tim mạch", "bệnh hô hấp"],
        note     = "y tế — hard negative: cùng là bệnh, khác cơ chế",
    ),
    BenchItem(
        prompt   = "Kiến trúc sư là",
        positive = ["người thiết kế công trình", "chuyên gia xây dựng"],
        negative = ["kỹ sư kết cấu", "thợ nề", "nhà điêu khắc", "kỹ sư điện"],
        note     = "nghề — hard negative: cùng liên quan xây dựng/sáng tạo, khác chuyên môn",
    ),
    BenchItem(
        prompt   = "Lúa là",
        positive = ["cây lương thực", "thực vật", "cây trồng"],
        negative = ["cây cảnh", "cây dược liệu", "cây công nghiệp", "cây ăn quả"],
        note     = "nông nghiệp — hard negative: cùng là cây trồng, khác mục đích",
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 2: Entity — phân biệt chi tiết trong cùng miền
# 26 item, phủ 5 nhóm chủ đề: người, địa lý, sinh vật, công nghệ, thiên văn
# Không item nào trùng prompt với nhau hoặc với SEMANTIC_BENCH
# ══════════════════════════════════════════════════════════════════════════

ENTITY_BENCH: List[BenchItem] = [

    # ── Nhóm 1: Người nổi tiếng (8 item, 8 ngành khác nhau) ─────────────
    BenchItem(
        prompt   = "Albert Einstein là",
        positive = ["nhà vật lý", "nhà khoa học"],
        negative = ["nhà hóa học", "nhà văn", "ca sĩ", "vận động viên"],
    ),
    BenchItem(
        prompt   = "Marie Curie là",
        positive = ["nhà khoa học", "nhà vật lý", "nhà hóa học"],
        negative = ["nhà văn", "ca sĩ", "diễn viên", "chính trị gia"],
    ),
    BenchItem(
        prompt   = "Hồ Chí Minh là",
        positive = ["chính trị gia", "lãnh tụ", "nhà cách mạng"],
        negative = ["nhà khoa học", "nhạc sĩ", "vận động viên", "nhà văn"],
    ),
    BenchItem(
        prompt   = "William Shakespeare là",
        positive = ["nhà văn", "kịch tác gia", "nhà thơ"],
        negative = ["nhà khoa học", "chính trị gia", "nhà thám hiểm", "nhạc sĩ"],
    ),
    BenchItem(
        prompt   = "Mozart là",
        positive = ["nhạc sĩ", "nhà soạn nhạc"],
        negative = ["nhà vật lý", "cầu thủ", "bác sĩ", "nhà văn"],
    ),
    BenchItem(
        prompt   = "Leonardo da Vinci là",
        positive = ["họa sĩ", "nhà phát minh", "nghệ sĩ"],
        negative = ["ca sĩ", "vận động viên", "phi hành gia", "chính trị gia"],
    ),
    BenchItem(
        prompt   = "Charles Darwin là",
        positive = ["nhà sinh học", "nhà khoa học", "nhà tự nhiên học"],
        negative = ["ca sĩ", "nhà thơ", "cầu thủ bóng đá", "diễn viên"],
    ),
    BenchItem(
        prompt   = "Nikola Tesla là",
        positive = ["nhà phát minh", "kỹ sư điện"],
        negative = ["ca sĩ", "nhà thơ", "vận động viên", "diễn viên"],
    ),

    # ── Nhóm 2: Địa lý (8 item, trải khắp 5 châu) ───────────────────────
    BenchItem(
        prompt   = "Hà Nội là thủ đô của",
        positive = ["Việt Nam", "nước Việt Nam"],
        negative = ["Trung Quốc", "Nhật Bản", "Thái Lan", "Campuchia"],
    ),
    BenchItem(
        prompt   = "Tokyo là thủ đô của",
        positive = ["Nhật Bản", "nước Nhật"],
        negative = ["Hàn Quốc", "Trung Quốc", "Thái Lan", "Singapore"],
    ),
    BenchItem(
        prompt   = "Berlin là thủ đô của",
        positive = ["Đức", "nước Đức"],
        negative = ["Pháp", "Áo", "Ba Lan", "Bỉ"],
    ),
    BenchItem(
        prompt   = "Tháp Eiffel nằm ở",
        positive = ["Pháp", "Paris"],
        negative = ["Đức", "Ý", "Tây Ban Nha", "Anh"],
    ),
    BenchItem(
        prompt   = "Sông Nile chảy qua",
        positive = ["Ai Cập", "Bắc Phi"],
        negative = ["Việt Nam", "Nhật Bản", "Brazil", "Ấn Độ"],
    ),
    BenchItem(
        prompt   = "Sydney nằm ở",
        positive = ["Úc", "Australia"],
        negative = ["Canada", "Brazil", "Ấn Độ", "Nam Phi"],
    ),
    BenchItem(
        prompt   = "Angkor Wat nằm ở",
        positive = ["Campuchia", "Đông Nam Á"],
        negative = ["Thái Lan", "Việt Nam", "Myanmar", "Lào"],
    ),
    BenchItem(
        prompt   = "Sông Amazon chảy qua",
        positive = ["Nam Mỹ", "Brazil"],
        negative = ["châu Phi", "châu Á", "Bắc Mỹ", "châu Âu"],
    ),

    # ── Nhóm 3: Sinh vật (4 item, 4 lớp khác nhau) ──────────────────────
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
        positive = ["chim", "chim săn mồi", "động vật có cánh"],
        negative = ["thú", "bò sát", "cá", "côn trùng"],
    ),
    BenchItem(
        prompt   = "Cá mập là loài",
        positive = ["cá", "động vật săn mồi biển"],
        negative = ["động vật có vú", "chim biển", "bò sát", "động vật giáp xác"],
    ),

    # ── Nhóm 4: Công nghệ & khoa học (4 item) ────────────────────────────
    BenchItem(
        prompt   = "Ngôn ngữ Python thường được dùng để",
        positive = ["lập trình", "phân tích dữ liệu", "xây dựng ứng dụng"],
        negative = ["nấu ăn", "leo núi", "chơi thể thao", "vẽ tranh sơn dầu"],
    ),
    BenchItem(
        prompt   = "Trí tuệ nhân tạo là lĩnh vực thuộc",
        positive = ["khoa học máy tính", "công nghệ thông tin"],
        negative = ["y học", "nông nghiệp", "nghệ thuật truyền thống", "thể thao"],
    ),
    BenchItem(
        prompt   = "Máy tính được dùng để",
        positive = ["xử lý thông tin", "tính toán", "lưu trữ dữ liệu"],
        negative = ["trồng cây", "chữa bệnh", "xây nhà", "nấu ăn"],
    ),
    BenchItem(
        prompt   = "Điện thoại thông minh là",
        positive = ["thiết bị điện tử", "công cụ liên lạc", "thiết bị di động"],
        negative = ["dụng cụ nấu ăn", "nhạc cụ", "phương tiện giao thông", "vũ khí"],
    ),

    # ── Nhóm 5: Thiên văn (2 item) ────────────────────────────────────────
    BenchItem(
        prompt   = "Mặt Trăng là",
        positive = ["vệ tinh của Trái Đất", "thiên thể tự nhiên"],
        negative = ["ngôi sao", "hành tinh độc lập", "tiểu hành tinh", "sao chổi"],
    ),
    BenchItem(
        prompt   = "Hành tinh Sao Hỏa có màu",
        positive = ["đỏ", "đỏ cam"],
        negative = ["xanh lam", "vàng kim", "trắng bạch", "đen"],
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 3: Fact — sự kiện không tranh cãi, chỉ có 1 đáp án đúng
# Tránh: sông dài nhất (Nile vs Amazon tranh luận), thủ đô ít biết
# Khoảng trắng đầu positive để ghép sát prompt, tránh BPE tokenize khác
# ══════════════════════════════════════════════════════════════════════════

FACT_BENCH: List[BenchItem] = [
    # ── Địa lý ───────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Thủ đô của Việt Nam là",
        positive = [" Hà Nội"],
        negative = [" Thành phố Hồ Chí Minh", " Đà Nẵng", " Huế", " Cần Thơ"],
    ),
    BenchItem(
        prompt   = "Thủ đô của Pháp là",
        positive = [" Paris"],
        negative = [" London", " Berlin", " Rome", " Madrid"],
    ),
    BenchItem(
        prompt   = "Nước láng giềng phía Bắc của Việt Nam là",
        positive = [" Trung Quốc"],
        negative = [" Lào", " Campuchia", " Thái Lan", " Myanmar"],
    ),
    BenchItem(
        prompt   = "Thủ đô của Nhật Bản là",
        positive = [" Tokyo"],
        negative = [" Osaka", " Kyoto", " Hiroshima", " Nagoya"],
    ),
    # ── Hóa học / Vật lý ─────────────────────────────────────────────────
    BenchItem(
        prompt   = "Ký hiệu hóa học của vàng là",
        positive = [" Au"],
        negative = [" Ag", " Fe", " Cu", " Pt"],
    ),
    BenchItem(
        prompt   = "Ký hiệu hóa học của bạc là",
        positive = [" Ag"],
        negative = [" Au", " Fe", " Cu", " Zn"],
    ),
    BenchItem(
        prompt   = "Đơn vị đo nhiệt độ trong hệ SI là",
        positive = [" Kelvin"],
        negative = [" Celsius", " Fahrenheit", " Rankine"],
    ),
    BenchItem(
        prompt   = "Tốc độ ánh sáng trong chân không xấp xỉ",
        positive = [" 300.000 km/s", " 3×10⁸ m/s"],
        negative = [" 1.000 km/s", " 30.000 km/s", " 3.000.000 km/s"],
    ),
    BenchItem(
        prompt   = "Nước sôi ở nhiệt độ",
        positive = [" 100 độ Celsius", " 100°C"],
        negative = [" 0 độ Celsius", " 37 độ Celsius", " 200 độ Celsius"],
    ),
    BenchItem(
        prompt   = "Đơn vị cơ bản đo khối lượng trong hệ SI là",
        positive = [" kilogram"],
        negative = [" gram", " pound", " ounce", " tấn"],
    ),
    # ── Thiên văn ─────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "Hành tinh thứ ba tính từ Mặt Trời là",
        positive = [" Trái Đất"],
        negative = [" Sao Hỏa", " Sao Kim", " Sao Mộc", " Sao Thủy"],
    ),
    BenchItem(
        prompt   = "Hành tinh lớn nhất trong hệ Mặt Trời là",
        positive = [" Sao Mộc"],
        negative = [" Sao Thổ", " Sao Hải Vương", " Trái Đất", " Sao Hỏa"],
    ),
    BenchItem(
        prompt   = "Hành tinh gần Mặt Trời nhất là",
        positive = [" Sao Thủy"],
        negative = [" Sao Kim", " Trái Đất", " Sao Hỏa", " Sao Mộc"],
    ),
    BenchItem(
        prompt   = "Trái Đất quay quanh",
        positive = [" Mặt Trời"],
        negative = [" Mặt Trăng", " Sao Hỏa", " Sao Mộc", " trục của nó"],
    ),
    # ── Sinh học ──────────────────────────────────────────────────────────
    BenchItem(
        prompt   = "DNA là viết tắt của",
        positive = [" Deoxyribonucleic Acid"],
        negative = [" Digital Network Access", " Dynamic Numeric Array", " Direct Neural Architecture"],
    ),
    BenchItem(
        prompt   = "Quang hợp là quá trình",
        positive = [" thực vật tổng hợp chất hữu cơ từ ánh sáng", " chuyển hóa năng lượng ánh sáng"],
        negative = [" động vật tiêu hóa thức ăn", " vi khuẩn phân hủy chất hữu cơ", " con người hít thở oxygen"],
    ),
    # ── Lịch sử / Toán học ───────────────────────────────────────────────
    BenchItem(
        prompt   = "Chiến tranh thế giới thứ hai kết thúc vào năm",
        positive = [" 1945"],
        negative = [" 1918", " 1939", " 1950", " 1975"],
    ),
    BenchItem(
        prompt   = "Số nguyên tố nhỏ nhất là",
        positive = [" 2"],
        negative = [" 1", " 3", " 5", " 0"],
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 4: Language Quality — Distinct-1/2 và repetition rate
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
    "Văn học Việt Nam có",
    "Lịch sử Việt Nam bắt đầu",
]


# ══════════════════════════════════════════════════════════════════════════
# Cấp 5: OOD — generalization ra ngoài phân phối train
# 2 nhóm:
#   A. Topic OOD — concept hiện đại ít xuất hiện trong Wikipedia tiếng Việt
#   B. Reasoning OOD — từ vựng bịa đặt, đo khả năng suy luận thuần túy
# ══════════════════════════════════════════════════════════════════════════

OOD_BENCH: List[BenchItem] = [

    # ── Nhóm A: Topic OOD (6 item) ────────────────────────────────────────
    BenchItem(
        prompt   = "Robot là",
        positive = ["máy móc tự động", "thiết bị cơ điện tử"],
        negative = ["sinh vật sống", "loài động vật", "thực vật", "khoáng vật"],
        note     = "công nghệ hiện đại",
    ),
    BenchItem(
        prompt   = "Blockchain là",
        positive = ["công nghệ lưu trữ dữ liệu phân tán", "chuỗi khối"],
        negative = ["loài động vật", "địa danh", "môn thể thao", "nhạc cụ"],
    ),
    BenchItem(
        prompt   = "Vaccine là",
        positive = ["chế phẩm sinh học phòng bệnh", "thuốc phòng ngừa"],
        negative = ["loại thực phẩm", "loại máy móc", "vũ khí", "dụng cụ thể thao"],
    ),
    BenchItem(
        prompt   = "Năng lượng mặt trời là",
        positive = ["nguồn năng lượng tái tạo", "năng lượng sạch"],
        negative = ["nhiên liệu hóa thạch", "khoáng sản", "thực phẩm", "vũ khí"],
    ),
    BenchItem(
        prompt   = "Mạng xã hội là",
        positive = ["nền tảng kết nối trực tuyến", "dịch vụ internet"],
        negative = ["mạng lưới điện", "mạng giao thông đường bộ", "loài sinh vật", "loại thực phẩm"],
    ),
    BenchItem(
        prompt   = "Trí tuệ nhân tạo tổng quát là",
        positive = ["hệ thống AI có khả năng tổng quát", "công nghệ AI tiên tiến"],
        negative = ["loài động vật", "địa danh", "môn thể thao", "phương tiện giao thông"],
        note     = "AGI — concept rất mới, ít trong Wikipedia cũ",
    ),

    # ── Nhóm B: Reasoning OOD — từ vựng bịa đặt (6 item) ─────────────────
    BenchItem(
        prompt   = "Zorb là một loài động vật. Mọi động vật đều là sinh vật. Zorb là",
        positive = ["sinh vật"],
        negative = ["quốc gia", "thành phố", "hành tinh", "phần mềm"],
        note     = "syllogism 2 bước, từ vựng bịa",
    ),
    BenchItem(
        prompt   = "Mọi flar đều là zent. Mọi zent đều là sinh vật. Blen là một flar. Blen là",
        positive = ["sinh vật"],
        negative = ["quốc gia", "hành tinh", "phần mềm", "thiên hà"],
        note     = "syllogism 3 bước, từ vựng bịa hoàn toàn",
    ),
    BenchItem(
        prompt   = "Kira là bác sĩ. Mọi bác sĩ đều làm việc trong ngành y tế. Kira thuộc",
        positive = ["ngành y tế"],
        negative = ["ngành nông nghiệp", "ngành thể thao", "ngành hàng không", "ngành giáo dục"],
        note     = "syllogism với nghề nghiệp thật",
    ),
    BenchItem(
        prompt   = "Mọi nori đều là phương tiện. Mọi phương tiện đều được dùng để di chuyển. Teka là một nori. Teka được dùng để",
        positive = ["di chuyển"],
        negative = ["quang hợp", "săn mồi", "lập trình", "nấu ăn"],
        note     = "syllogism với đặc tính chức năng",
    ),
    BenchItem(
        prompt   = "Mọi drako đều là máy móc. Mọi máy móc đều phục vụ một mục đích. Lena là một drako. Lena là",
        positive = ["máy móc", "công cụ"],
        negative = ["động vật", "thực vật", "hành tinh", "cảm xúc"],
        note     = "syllogism 2 bước với đặc tính phân loại",
    ),
    BenchItem(
        prompt   = "An là kỹ sư. Mọi kỹ sư đều là người lao động. Mọi người lao động đều có thu nhập. An có",
        positive = ["thu nhập"],
        negative = ["vây cá", "cánh chim", "vỏ cây", "nhiệt độ sôi"],
        note     = "syllogism 3 bước xuyên nhiều đặc tính",
    ),
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

    logits    = model(ids_t, attn_mask=mask)
    log_probs = F.log_softmax(logits[0], dim=-1)

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
def _generate_sample(
    model,
    tokenizer,
    prompt     : str,
    max_new    : int   = 50,
    temperature: float = 0.8,
    top_k      : int   = 50,
    device     : torch.device = None,
) -> list[int]:
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
    """Distinct-1, Distinct-2, repeat_ratio."""
    all_tokens, all_bigrams = [], []
    repeat_count = total_tokens = 0

    for seq in token_seqs:
        all_tokens.extend(seq)
        total_tokens += len(seq)
        for i in range(len(seq) - 1):
            all_bigrams.append((seq[i], seq[i + 1]))
            if seq[i] == seq[i + 1]:
                repeat_count += 1

    distinct1 = len(set(all_tokens))  / max(len(all_tokens),  1)
    distinct2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    repeat    = repeat_count / max(total_tokens - 1, 1)

    return {
        "distinct1"     : distinct1,
        "distinct2"     : distinct2,
        "repeat_ratio"  : repeat,
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
            seq = _generate_sample(model, tokenizer, prompt, max_new=max_new, device=device)
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
    verbose           : bool = True,
    step              : int  = None,
    n_language_samples: int  = 5,
) -> dict:
    """
    Chạy toàn bộ 5 cấp benchmark, trả về dict kết quả + TOTAL_SCORE.

    TOTAL_SCORE = weighted sum theo WEIGHTS (xem đầu file).

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