import os
import time
import requests
import matplotlib.pyplot as plt
from tabulate import tabulate

API_OCR_URL = "http://localhost:8000/ocr/cccd"
API_FACE_URL = "http://localhost:8000/api/face-verification/verify-from-ocr"
DIR_CCCD = "datasets/source/images"
DIR_SELFIE = "datasets/face"

def find_image(folder, index):
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        path = os.path.join(folder, f"{index}{ext}")
        if os.path.exists(path):
            return path
    return None

def send_verify_request(cccd_path, selfie_path):
    try:
        with open(cccd_path, "rb") as f_cccd:
            ocr_resp = requests.post(API_OCR_URL, files={"file": f_cccd})
    except Exception as e:
        return None, f"Lỗi OCR: {e}"
        
    if ocr_resp.status_code != 200:
        return ocr_resp, "Lỗi OCR"
        
    ocr_data = ocr_resp.json()
    session_id = (ocr_data.get("session_id") or ocr_data.get("sessionId") or 
                  ocr_data.get("data", {}).get("session_id") or ocr_data.get("data", {}).get("sessionId"))
    
    payload = {"session_id": session_id}
    with open(selfie_path, "rb") as f_selfie:
        face_resp = requests.post(API_FACE_URL, files={"selfie_image": f_selfie}, data=payload)
        if face_resp.status_code not in [400, 422]:
            return face_resp, "Thành công"
            
    with open(cccd_path, "rb") as f_cccd, open(selfie_path, "rb") as f_selfie:
        face_resp = requests.post(API_FACE_URL, files={"card_image": f_cccd, "selfie_image": f_selfie}, data=payload)
    return face_resp, "Thành công"

def extract_similarity_smart(data):
    if isinstance(data, (int, float)): return float(data)
    if isinstance(data, dict):
        for k in ["similarity", "similarity_score", "score", "match_score", "sim"]:
            if k in data: return float(data[k])
        for k, v in data.items():
            if isinstance(v, (int, float)) and any(x in k.lower() for x in ['sim', 'score']): return float(v)
            if isinstance(v, dict):
                res = extract_similarity_smart(v)
                if res is not None: return res
    return None

def run_face_test():
    test_pairs = []
    for i in range(1, 11):
        cccd_path = find_image(DIR_CCCD, i)
        selfie_path = find_image(DIR_SELFIE, i)
        is_same = True if i <= 6 else False
        if cccd_path and selfie_path:
            test_pairs.append((i, cccd_path, selfie_path, is_same))

    scores, ground_truths, detailed_rows = [], [], []

    for idx, cccd, selfie, label in test_pairs:
        resp, _ = send_verify_request(cccd, selfie)
        if resp and resp.status_code == 200:
            sim = extract_similarity_smart(resp.json()) or 0.0
            if sim > 1.0 and sim > 10.0: sim = sim / 100.0
                
            scores.append(sim)
            ground_truths.append(label)

            pred_match = sim >= 0.80 # ĐÃ FIX THRESHOLD THÀNH 0.80
            is_correct = (pred_match == label)
            detailed_rows.append([
                f"Cặp {idx:02d}", os.path.basename(cccd), os.path.basename(selfie),
                "Trùng" if label else "Khác", f"{sim:.4f}",
                "Match" if pred_match else "Non-Match", "PASS" if is_correct else "FAIL"
            ])

    print("\n1. KẾT QUẢ CHI TIẾT 10 CẶP (TẠI THRESHOLD = 0.80):")
    print(tabulate(detailed_rows, headers=["STT", "Ảnh CCCD", "Ảnh Selfie", "Thực tế", "Similarity", "Dự đoán", "Kết quả"], tablefmt="grid"))

    # Đánh giá dải Threshold mới phù hợp với model của bạn
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    fpr_list, fnr_list, acc_list = [], [], []

    for th in thresholds:
        tp = sum(1 for s, y in zip(scores, ground_truths) if y and s >= th)
        tn = sum(1 for s, y in zip(scores, ground_truths) if not y and s < th)
        fp = sum(1 for s, y in zip(scores, ground_truths) if not y and s >= th)
        fn = sum(1 for s, y in zip(scores, ground_truths) if y and s < th)

        p_count = sum(1 for y in ground_truths if y)      
        n_count = sum(1 for y in ground_truths if not y)  

        acc = (tp + tn) / len(ground_truths) * 100
        fpr = (fp / n_count * 100) if n_count > 0 else 0
        fnr = (fn / p_count * 100) if p_count > 0 else 0

        fpr_list.append(fpr)
        fnr_list.append(fnr)
        acc_list.append(acc)

    # Vẽ đồ thị Hình 5.1
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fpr_list, marker='o', label='FPR (Tỷ lệ sai - Khác thành Trùng)', color='red')
    plt.plot(thresholds, fnr_list, marker='s', label='FNR (Tỷ lệ sai - Trùng thành Khác)', color='blue')
    plt.plot(thresholds, acc_list, marker='^', label='Accuracy', color='green', linestyle='--')
    plt.axvline(x=0.80, color='gray', linestyle=':', label='Threshold (0.80)')
    plt.title('Quan hệ giữa Threshold và kết quả Face Verification')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Tỷ lệ (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('hinh_5_1_threshold.png', dpi=300, bbox_inches='tight')
    print(f"\n[THÀNH CÔNG] Đã tạo file đồ thị: hinh_5_1_threshold.png")

if __name__ == "__main__":
    run_face_test()