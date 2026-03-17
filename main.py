import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['ch_sim'])
img = cv2.imread('image.jpg')
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# 1. อ่านข้อความ
results = reader.readtext(img)

for (bbox, text, prob) in results:
    # แปลงพิกัดเป็น Numpy Array
    pts = np.array(bbox, dtype=np.int32)
    
    # 2. คำนวณความสว่าง "เฉพาะจุด" ของตัวอักษรนั้นๆ
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    
    if roi.size > 0:
        avg_brightness = np.mean(roi)
        
        # ถ้าความสว่างเฉลี่ย > 180 (ครอบคลุมถึงสีเทาอ่อนใน Bubble) ให้ลบเลย
        if avg_brightness > 180:
            print(f"ทำความสะอาด: {text}")
            # ขยาย Mask ออกไป 2 พิกเซลแทนการหด (เพื่อให้ลบตัวอักษรได้หมดจด)
            cv2.fillPoly(mask, [pts], 255)

# 3. ขยายขอบ Mask เล็กน้อยเพื่อให้เก็บขอบตัวอักษรที่ฟุ้งๆ
kernel = np.ones((3,3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

# 4. ลบด้วย Inpaint (เพิ่มรัศมีการลบเป็น 5 เพื่อความเนียน)
dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

cv2.imwrite('cleaned_v4.jpg', dst)