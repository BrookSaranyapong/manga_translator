import cv2
import numpy as np
import easyocr

class MangaCleaner:
    def __init__(self):
        # โหลด AI ไว้รอใช้งาน (ทำครั้งเดียวตอนเริ่ม)
        self.reader = easyocr.Reader(['ch_sim'])

    def detect_text(self, img):
        # ให้ AI อ่านข้อความ
        return self.reader.readtext(img)

    def process_image(self, img, results):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cleaned_text_data = []

        for (bbox, text, prob) in results:
            pts = np.array(bbox, dtype=np.int32)
            rect = cv2.boundingRect(pts)
            x, y, rw, rh = rect
            
            # เช็คความสว่างพื้นหลัง
            roi = img[y:y+rh, x:x+rw]
            if roi.size > 0 and np.mean(roi) > 180:
                # เก็บข้อมูลไว้ลง JSON
                cleaned_text_data.append({
                    "text": text,
                    "position": {"x": x, "y": y, "w": rw, "h": rh},
                    "confidence": float(prob)
                })
                # วาด Mask เตรียมลบ
                cv2.fillPoly(mask, [pts], 255)

        # ขยายยางลบและลบภาพ
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        cleaned_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        
        return cleaned_img, cleaned_text_data