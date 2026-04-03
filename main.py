import cv2
import os
import numpy as np 
import json 

from modules.yolo_detect_bubbles import YoloBubbleDetector
from modules.image_processor import MangaCleaner
from modules.save_to_json import save_to_json
from modules.debug_utils import draw_detected_bubbles, save_cropped_bubbles
from modules.text_renderer import MangaTypesetter 
from modules.translator import MangaTranslatorRAG # ✨ นำเข้า AI แปลภาษา

def cv2_imread_unicode(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

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
    final_output_img = os.path.join(output_dir, f'final_{filename}')
    output_json = os.path.join(output_dir, 'extracted_text.json')
    translation_file = os.path.join(output_dir, 'translated_text.json') 
    debug_img_path = os.path.join(output_dir, f'debug_yolo_{filename}')
    
    model_path = os.path.join('models', 'model.pt')
    font_path = os.path.join('fonts', 'THSarabun.ttf')

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    print("กำลังเตรียม AI...")
    detector = YoloBubbleDetector(model_path)
    cleaner = MangaCleaner()

    img = cv2_imread_unicode(input_file)
    if img is None:
        print(f"❌ ไม่พบไฟล์ภาพที่: {input_file}")
        return

    # ---------------------------------------------------------
    # จังหวะที่ 1: YOLO สแกนหา Bubble
    # ---------------------------------------------------------
    print("🤖 [1/3] YOLO กำลังค้นหาพื้นที่กรอบคำพูด...")
    detected_bubbles = detector.detect(img, confidence_threshold=0.25)

    print("🐛 กำลังสร้างภาพ Debug และ Crop...")
    debug_img = draw_detected_bubbles(img, detected_bubbles)
    cv2_imwrite_unicode(debug_img_path, debug_img)
    save_cropped_bubbles(img, detected_bubbles, output_dir, filename)

    # ---------------------------------------------------------
    # จังหวะที่ 2: EasyOCR อ่านและลบ
    # ---------------------------------------------------------
    print("👁️ [2/3] EasyOCR กำลังอ่านข้อความและทำความสะอาดภาพ...")
    cleaned_img, text_data = cleaner.process_image(img, detected_bubbles)

    cv2_imwrite_unicode(output_img, cleaned_img)
    save_to_json(text_data, output_json, merge=True)
    print(f"✅ สกัดข้อความและคลีนภาพสำเร็จ {len(text_data)} ประโยค")

    # ---------------------------------------------------------
    # ✨ จังหวะที่ 2.5: AI RAG แปลภาษา และเซฟไฟล์ JSON ✨
    # ---------------------------------------------------------
    print("🌍 [2.5/3] AI กำลังแปลภาษาพร้อมค้นหาบริบท (RAG)...")
    
    GOOGLE_API_KEY = "Your_api_token" # ⚠️ ระบุ API Key ตรงนี้!
    
    # ✨ นำเข้าฟังก์ชันจัดการไฟล์ CSV
    from modules.glossary_manager import update_glossary_auto 
    
    translated_dict = {}
    
    try:
        translator = MangaTranslatorRAG(csv_path="glossary.csv", api_key=GOOGLE_API_KEY)
        
        # ⚡ [ส่วนที่เพิ่มเข้ามา] ให้ AI วิเคราะห์หาชื่อเฉพาะก่อนแปล ⚡
        all_chinese_text = " ".join([data.get("text", "") for data in text_data if data.get("text")])
        if all_chinese_text:
            print(f"🕵️‍♀️ กำลังวิเคราะห์หน้ามังงะเพื่อหาชื่อเฉพาะ...")
            new_entities = translator.extract_new_entities(all_chinese_text)
            
            if new_entities:
                print(f"   ✨ พบคำศัพท์ใหม่ {len(new_entities)} คำ: {new_entities}")
                update_glossary_auto(new_entities, csv_path="glossary.csv")
                
                # โหลดฐานข้อมูลศัพท์ใหม่เพื่อให้ RAG รู้จักคำที่เพิ่งเซฟไป
                print("🔄 รีโหลดฐานข้อมูลคำศัพท์...")
                translator = MangaTranslatorRAG(csv_path="glossary.csv", api_key=GOOGLE_API_KEY)
            else:
                print("   🔍 ไม่พบคำศัพท์เฉพาะใหม่ในหน้านี้ (AI ไม่บันทึกคำทั่วไป)")

        # เริ่มทำการแปลทีละประโยค
        for data in text_data:
            chinese_text = data.get("text", "")
            b_id = str(data["bubble_id"])
            if chinese_text:
                print(f"   🇨🇳 ต้นฉบับ: {chinese_text}")
                thai_translation = translator.translate(chinese_text)
                data["translated_text"] = thai_translation
                translated_dict[b_id] = thai_translation # เตรียมเซฟลง JSON
                print(f"   🇹🇭 คำแปล: {thai_translation}\n")
            else:
                data["translated_text"] = ""
                translated_dict[b_id] = ""
                
        # สร้างไฟล์ translated_text.json แบบสวยงาม
        with open(translation_file, 'w', encoding='utf-8') as f:
            json.dump(translated_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ บันทึกไฟล์คำแปลเรียบร้อย: {translation_file}")
            
    except Exception as e:
         print(f"❌ [RAG] เกิดข้อผิดพลาดในการแปล: {e}")
         # ถ้าแปลล้มเหลว ให้ใช้ของเดิม
         for data in text_data:
             data["translated_text"] = data.get("text", "")

    # ---------------------------------------------------------
    # จังหวะที่ 3: ประกอบร่างข้อความแปล (Typesetting)
    # ---------------------------------------------------------
    print("✍️ [3/3] กำลังประกอบร่างข้อความแปลลงบนภาพ...")
    
    if os.path.exists(font_path):
        typesetter = MangaTypesetter(font_path=font_path, font_size=26)
        final_img = typesetter.draw_text(cleaned_img, text_data)
        
        cv2_imwrite_unicode(final_output_img, final_img)
        print("🎉 พิมพ์ข้อความแปลไทยลงภาพเรียบร้อย!")
    else:
        print(f"   ❌ ไม่พบไฟล์ฟอนต์ที่ {font_path} ข้ามขั้นตอนการพิมพ์ข้อความ")

    print(f"📂 ผลลัพธ์ทั้งหมดอยู่ที่โฟลเดอร์ {output_dir}")

if __name__ == "__main__":
    main()