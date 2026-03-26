import cv2
import numpy as np
import easyocr

class MangaCleaner:
    def __init__(self):
        # โหลด AI อ่านข้อความ
        self.reader = easyocr.Reader(['ch_sim'])

    def process_image(self, img, detected_bubbles):
        h, w = img.shape[:2]
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        cleaned_text_data = []

        for i, bubble in enumerate(detected_bubbles):
            # 1. ดึงพิกัดจาก YOLO
            bx, by, bw, bh = bubble["position"]["x"], bubble["position"]["y"], bubble["position"]["w"], bubble["position"]["h"]
            
            # 2. Crop ภาพเฉพาะส่วน Bubble (บวก Padding เล็กน้อยกันตัวอักษรชิดขอบ)
            pad = 5
            y1, y2 = max(0, by-pad), min(h, by+bh+pad)
            x1, x2 = max(0, bx-pad), min(w, bx+bw+pad)
            crop = img[y1:y2, x1:x2]

            # 3. ส่งภาพ Crop ไป OCR
            # การส่งภาพเล็กเข้าไปจะทำให้อ่านแม่นขึ้นมาก
            results = self.reader.readtext(crop)

            for (bbox, text, prob) in results:
                # 4. แปลงพิกัดจากพิกัดรูป Crop กลับไปเป็นพิกัดรูปใหญ่ (Global Coordinates)
                pts = np.array(bbox, dtype=np.int32)
                pts[:, 0] += x1  # บวกค่า X ของจุดเริ่มต้น Crop
                pts[:, 1] += y1  # บวกค่า Y ของจุดเริ่มต้น Crop

                cleaned_text_data.append({
                    "bubble_id": i + 1,
                    "text": text,
                    "position": pts.tolist(),
                    "confidence": float(prob)
                })

                # วาด Mask สำหรับลบในรูปใหญ่
                cv2.fillPoly(inpaint_mask, [pts], 255)

        # 5. ลบภาพ
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5,5), np.uint8), iterations=3)
        cleaned_img = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_NS)

        return cleaned_img, cleaned_text_data