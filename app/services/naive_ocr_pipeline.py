import time
from fuzzywuzzy import fuzz
import easyocr

class NaiveOCRPipeline:
    def __init__(self):
        # Khởi tạo EasyOCR chạy bằng CPU (Đọc toàn bộ ảnh gốc sẽ rất chậm)
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)

    def process(self, image):
        start_time = time.time()
        
        # 1. KHÔNG TIỀN XỬ LÝ: Không xoay, không cắt góc. Đưa thẳng ảnh gốc vào
        ocr_start = time.time()
        raw_ocr_results = self.reader.readtext(image)
        ocr_time = time.time() - ocr_start

        # Gộp toàn bộ chữ đọc được thành một khối duy nhất
        full_text = " ".join([res[1] for res in raw_ocr_results]).upper()

        # 2. BÓC TÁCH KÉM: Dò tìm thông tin bằng Fuzzy Matching O(N^2)
        parse_start = time.time()
        extracted_data = {
            "id": self._naive_extract(full_text, "SỐ", 12),
            "name": self._naive_extract(full_text, "HỌ VÀ TÊN", 20),
            "dob": self._naive_extract(full_text, "NGÀY SINH", 10),
        }
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time

        return {
            "data": extracted_data,
            "metrics": {
                "total_time_ms": round(total_time * 1000, 2),
                "ocr_time_ms": round(ocr_time * 1000, 2),
                "parse_time_ms": round(parse_time * 1000, 2),
                "card_detection_time_ms": 0 
            }
        }

    def _naive_extract(self, full_text, keyword, value_length):
        words = full_text.split()
        best_match_score = 0
        best_index = -1

        for i in range(len(words)):
            score = fuzz.ratio(keyword, words[i])
            if score > best_match_score:
                best_match_score = score
                best_index = i

        if best_index != -1 and best_index + 1 < len(words):
            return " ".join(words[best_index+1 : best_index+4])
        return ""