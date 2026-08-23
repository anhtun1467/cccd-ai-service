import os
import requests
import json
import time

# Khai báo chính xác URL OCR của bạn
API_OCR_URL = "http://localhost:8000/ocr/cccd"
DIR_CCCD = "datasets/source/images"
DIR_GT = "datasets/source/ground_truth"

def extract_field(data, keys):
    """Hàm tìm kiếm thông minh một trường thông tin bất kể nằm ở cấp JSON nào"""
    if not isinstance(data, dict):
        return ""
    for k in keys:
        if k in data and data[k]:
            return data[k]
    for k, v in data.items():
        if isinstance(v, dict):
            res = extract_field(v, keys)
            if res:
                return res
    return ""

def generate_ground_truth():
    os.makedirs(DIR_GT, exist_ok=True)
    image_files = [f for f in os.listdir(DIR_CCCD) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print("=" * 60)
    print(f"BẮT ĐẦU TỰ ĐỘNG SINH FILE JSON CHO {len(image_files)} ẢNH...")
    print("=" * 60)

    for img_name in image_files:
        img_path = os.path.join(DIR_CCCD, img_name)
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(DIR_GT, f"{base_name}.json")

        # Nếu file JSON đã tồn tại và có nội dung rồi thì bỏ qua để không ghi đè công sức bạn sửa tay
        if os.path.exists(json_path) and os.path.getsize(json_path) > 100:
            print(f"[BỎ QUA] Đã có sẵn file hợp lệ: {base_name}.json")
            continue

        try:
            # Gửi file đúng chuẩn tham số 'file' của FastAPI backend
            with open(img_path, "rb") as f:
                resp = requests.post(API_OCR_URL, files={"file": f})
                
            if resp.status_code == 200:
                res_json = resp.json()
                
                # In ra để debug nếu cần, lấy phần data bên trong ApiResponse
                data = res_json.get("data", res_json)
                
                # Dùng hàm thông minh quét các biến có thể xảy ra trong backend
                gt_data = {
                    "idNumber": extract_field(data, ["idNumber", "id", "card_number", "so_cccd", "number"]),
                    "name": extract_field(data, ["name", "fullName", "ho_va_ten", "fullname"]),
                    "dob": extract_field(data, ["dob", "birthDay", "ngay_sinh", "birth"]),
                    "gender": extract_field(data, ["gender", "gioi_tinh", "sex"]),
                    "nationality": extract_field(data, ["nationality", "quoc_tich"]) or "Việt Nam",
                    "homeTown": extract_field(data, ["homeTown", "que_quan", "placeOfOrigin"]),
                    "residence": extract_field(data, ["residence", "noi_tru", "placeOfResidence", "address"]),
                    "expiry": extract_field(data, ["expiry", "ngay_het_han", "validUntil"])
                }
                
                # Lưu ra file JSON
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(gt_data, json_file, ensure_ascii=False, indent=4)
                
                print(f"[THÀNH CÔNG] Đã tạo: {base_name}.json -> Tên: {gt_data['name']}")
            else:
                print(f"[LỖI API] Ảnh {img_name} - Mã lỗi {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[LỖI NGOẠI LỆ] Xử lý {img_name}: {e}")
        
        time.sleep(0.3)

    print("\nHOÀN THÀNH! Bây giờ các file JSON đã được điền dữ liệu đầy đủ.")

if __name__ == "__main__":
    generate_ground_truth()