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
            
            cleaned_text_data.append({
                "text": text,
                "position": {"x": x, "y": y, "w": rw, "h": rh},
                "confidence": float(prob)
            })
            
            # วาด Mask ตาม Bounding Box
            cv2.fillPoly(mask, [pts], 255)

        # ขยายยางลบและลบภาพ
        # *** ปรับ iterations จาก 1 เป็น 4 เพื่อขยายมาสก์ "กว้าง" ขึ้น ครอบคลุม SFX และขอบ ***
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4) 
        # เพิ่มรัศมีของ Telea เพื่อการผสมสีที่นุ่มนวลขึ้นในพื้นที่ที่ขยาย
        cleaned_img = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA) 
        
        return cleaned_img, cleaned_text_data