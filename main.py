import cv2
import os
import numpy as np # เพิ่ม numpy
import json # ✨ เพิ่ม json

from modules.yolo_detect_bubbles import YoloBubbleDetector
from modules.image_processor import MangaCleaner
from modules.save_to_json import save_to_json
from modules.debug_utils import draw_detected_bubbles, save_cropped_bubbles # เรียกมาใช้
from modules.text_renderer import MangaTypesetter # ✨ นำเข้าโมดูลวาดข้อความ

# ฟังก์ชันอ่านไฟล์ที่รองรับ Unicode (ไทย/จีน)
def cv2_imread_unicode(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

# ฟังก์ชันเซฟไฟล์ที่รองรับ Unicode
def cv2_imwrite_unicode(path, img):
    extension = os.path.splitext(path)[1]
    _, res = cv2.imencode(extension, img)
    res.tofile(path)

def main():
    input_dir = 'image/input'
    output_dir = 'image/output'
    os.makedirs(output_dir, exist_ok=True)

    filename = '5 05.jpg' 
    input_file = os.path.join(input_dir, filename)
    
    output_img = os.path.join(output_dir, f'cleaned_{filename}')
    final_output_img = os.path.join(output_dir, f'final_{filename}') # ✨ ภาพสุดท้ายที่มีตัวหนังสือไทย
    output_json = os.path.join(output_dir, 'extracted_text.json')
    translation_file = os.path.join(output_dir, 'translated_text.json') # ✨ ไฟล์คำแปล
    debug_img_path = os.path.join(output_dir, f'debug_yolo_{filename}')
    
    model_path = os.path.join('models', 'model.pt')
    font_path = os.path.join('fonts', 'THSarabun.ttf') # ✨ พาร์ทสำหรับไฟล์ฟอนต์

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    print("กำลังเตรียม AI...")
    detector = YoloBubbleDetector(model_path)
    cleaner = MangaCleaner()

    # ใช้ฟังก์ชันใหม่เพื่อให้อ่านชื่อไฟล์ภาษาไทย/จีนได้
    img = cv2_imread_unicode(input_file)
    if img is None:
        print(f"❌ ไม่พบไฟล์ภาพที่: {input_file}")
        return

    # ---------------------------------------------------------
    # จังหวะที่ 1: YOLO สแกนหา Bubble
    # ---------------------------------------------------------
    print("🤖 [1/3] YOLO กำลังค้นหาพื้นที่กรอบคำพูด...")
    # ลด confidence ลงเหลือ 0.25 เพื่อให้เจอ Bubble ที่จางๆ หรือไม่มีขอบ
    detected_bubbles = detector.detect(img, confidence_threshold=0.25)

    # ใช้ debug_utils ที่คุณเขียนไว้
    print("🐛 กำลังสร้างภาพ Debug และ Crop...")
    debug_img = draw_detected_bubbles(img, detected_bubbles)
    cv2_imwrite_unicode(debug_img_path, debug_img)
    
    # ลองเซฟ Crop ออกมาดูว่า YOLO มองเห็น Bubble ตัวล่างไหม
    save_cropped_bubbles(img, detected_bubbles, output_dir, filename)

    # ---------------------------------------------------------
    # จังหวะที่ 2: EasyOCR อ่านและลบ
    # ---------------------------------------------------------
    print("👁️ [2/3] EasyOCR กำลังอ่านข้อความและทำความสะอาดภาพ...")
    cleaned_img, text_data = cleaner.process_image(img, detected_bubbles)

    # บันทึกผลลัพธ์ภาพคลีน และ JSON
    cv2_imwrite_unicode(output_img, cleaned_img)
    save_to_json(text_data, output_json, merge=True)
    
    print(f"✅ สกัดข้อความและคลีนภาพสำเร็จ {len(text_data)} ประโยค")

    # ---------------------------------------------------------
    # ✨ จังหวะที่ 3: ประกอบร่างข้อความแปล (Typesetting) ✨
    # ---------------------------------------------------------
    print("✍️ [3/3] กำลังประกอบร่างข้อความแปลลงบนภาพ...")
    
    # โหลดไฟล์คำแปล (ถ้ามี)
    translated_dict = {}
    if os.path.exists(translation_file):
        with open(translation_file, 'r', encoding='utf-8') as f:
            translated_dict = json.load(f)
    else:
        print(f"   ⚠️ ไม่พบไฟล์แปลภาษา ({translation_file}) จะใช้ข้อความต้นฉบับทดแทน")

    # แมปคำแปลเข้ากับ text_data
    for data in text_data:
        b_id = str(data["bubble_id"])
        if b_id in translated_dict:
            data["translated_text"] = translated_dict[b_id]
        else:
            data["translated_text"] = data["text"] # ใช้ของเดิมถ้าไม่มีคำแปล

    # วาดข้อความ
    if os.path.exists(font_path):
        typesetter = MangaTypesetter(font_path=font_path, font_size=26)
        final_img = typesetter.draw_text(cleaned_img, text_data)
        
        # เซฟภาพสุดท้าย
        cv2_imwrite_unicode(final_output_img, final_img)
        print("🎉 พิมพ์ข้อความแปลไทยลงภาพเรียบร้อย!")
    else:
        print(f"   ❌ ไม่พบไฟล์ฟอนต์ที่ {font_path} ข้ามขั้นตอนการพิมพ์ข้อความ")

    print(f"📂 ผลลัพธ์ทั้งหมดอยู่ที่โฟลเดอร์ {output_dir}")

if __name__ == "__main__":
    main()