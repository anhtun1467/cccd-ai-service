import time
from fuzzywuzzy import fuzz
import easyocr

class NaiveOCRPipeline:
    def __init__(self):
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)

    def process(self, image):
        start_time = time.time()
        
        ocr_start = time.time()
        raw_ocr_results = self.reader.readtext(image)
        ocr_time = time.time() - ocr_start

        # Gộp toàn bộ chữ đọc được thành một khối duy nhất
        full_text = " ".join([res[1] for res in raw_ocr_results]).upper()

        confidences = [res[2] for res in raw_ocr_results]
        average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        parse_start = time.time()
        
        # BỔ SUNG THÊM CÁC TRƯỜNG THÔNG TIN VÀO ĐÂY
        extracted_data = {
            "idNumber": self._naive_extract(full_text, ["SỐ", "NO"], 12),
            "fullName": self._naive_extract(full_text, ["HỌ VÀ TÊN", "FULL NAME"], 20),
            "dateOfBirth": self._naive_extract(full_text, ["NGÀY SINH", "DATE OF BIRTH"], 10),
            "gender": self._naive_extract(full_text, ["GIỚI TÍNH", "SEX"], 5),
            "nationality": self._naive_extract(full_text, ["QUỐC TỊCH", "NATIONALITY"], 10),
            "placeOfOrigin": self._naive_extract(full_text, ["QUÊ QUÁN", "PLACE OF ORIGIN"], 30),
            "placeOfResidence": self._naive_extract(full_text, ["NƠI THƯỜNG TRÚ", "PLACE OF RESIDENCE"], 40),
            "dateOfExpiry": self._naive_extract(full_text, ["CÓ GIÁ TRỊ ĐẾN", "DATE OF EXPIRY"], 10),
        }
        
        parse_time = time.time() - parse_start
        total_time = time.time() - start_time

        total_time_ms = round(total_time * 1000, 2)
        ocr_time_ms = round(ocr_time * 1000, 2)
        parse_time_ms = round(parse_time * 1000, 2)

        return {
            "success": True,
            "message": "Naive OCR hoàn tất",
            "data": {
                "status": "OCR_SUCCESS",
                "message": "Naive OCR thành công",
                "cccdData": extracted_data,
                "metadata": {
                    "engine": "EasyOCR-Naive",
                    "processingTime": round(total_time, 3),
                    "processingStagesMs": {
                        "cardDetection": 0,
                        "initialFullCardOcr": ocr_time_ms,
                        "fieldCropAndOcr": 0,
                        "parsingAndRegex": parse_time_ms,
                        "total": total_time_ms
                    },
                    "averageConfidence": average_confidence,
                    "fieldConfidences": {
                        key: average_confidence for key in extracted_data.keys()
                    }
                },
                "metrics": {
                    "total_time_ms": total_time_ms,
                    "ocr_time_ms": ocr_time_ms,
                    "parse_time_ms": parse_time_ms,
                    "card_detection_time_ms": 0 
                }
            }
        }

    def _naive_extract(self, full_text, keywords, value_length):
        """
        Cải tiến hàm extract để hỗ trợ danh sách nhiều từ khóa (ví dụ tiếng Việt hoặc tiếng Anh)
        """
        words = full_text.split()
        best_match_score = 0
        best_index = -1

        # Nếu truyền vào 1 chuỗi từ khóa, chuyển thành mảng
        if isinstance(keywords, str):
            keywords = [keywords]

        for i in range(len(words)):
            for kw in keywords:
                score = fuzz.ratio(kw, words[i])
                if score > best_match_score:
                    best_match_score = score
                    best_index = i

        # Nếu tìm thấy từ khóa, lấy một đoạn văn bản nằm ngay sau nó
        if best_index != -1 and best_index + 1 < len(words):
            # Lấy nhiều từ hơn phía sau từ khóa tùy thuộc vào độ dài dữ liệu cần lấy
            take_words = max(3, value_length // 4) 
            return " ".join(words[best_index + 1 : best_index + 1 + take_words])
        return ""