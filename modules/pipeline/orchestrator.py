import json
import os

from modules.detection import YoloBubbleDetector
from modules.ocr import MangaCleaner, draw_detected_bubbles, save_cropped_bubbles
from modules.translation import MangaTranslatorRAG, update_glossary_auto
from modules.rendering import MangaTypesetter
from modules.utils import cv2_imread_unicode, cv2_imwrite_unicode, save_to_json


class MangaPipeline:
    """Orchestrates the full manga processing pipeline:
    Detection → OCR → Translation → Typesetting
    """

    def __init__(self, model_path: str, font_path: str, api_key: str,
                 glossary_path: str = "glossary.csv", extract_entities: bool = True):
        self.detector = YoloBubbleDetector(model_path)
        self.cleaner = MangaCleaner()
        self.font_path = font_path
        self.api_key = api_key
        self.glossary_path = glossary_path
        self.extract_entities = extract_entities

    def run(self, input_file: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(input_file)
        output_img = os.path.join(output_dir, f'cleaned_{filename}')
        final_output_img = os.path.join(output_dir, f'final_{filename}')
        output_json = os.path.join(output_dir, 'extracted_text.json')
        translation_file = os.path.join(output_dir, 'translated_text.json')
        debug_img_path = os.path.join(output_dir, f'debug_yolo_{filename}')

        img = cv2_imread_unicode(input_file)
        if img is None:
            return

        # Stage 1: YOLO bubble detection
        detected_bubbles = self.detector.detect(img, confidence_threshold=0.25)

        debug_img = draw_detected_bubbles(img, detected_bubbles)
        cv2_imwrite_unicode(debug_img_path, debug_img)
        save_cropped_bubbles(img, detected_bubbles, output_dir, filename)

        # Stage 2: EasyOCR + cleaning
        cleaned_img, text_data = self.cleaner.process_image(img, detected_bubbles)

        cv2_imwrite_unicode(output_img, cleaned_img)
        save_to_json(text_data, output_json, merge=True)
        print(f"✅ สกัดข้อความและคลีนภาพสำเร็จ {len(text_data)} ประโยค")

        # Stage 2.5: RAG translation
        translated_dict = self._translate(text_data)

        # Stage 3: Typesetting
        self._typeset(cleaned_img, text_data, final_output_img)

        print(f"📂 ผลลัพธ์ทั้งหมดอยู่ที่โฟลเดอร์ {output_dir}")

    def _translate(self, text_data):
        print("🌍 [2.5/3] AI กำลังแปลภาษาพร้อมค้นหาบริบท (RAG)...")
        translated_dict = {}

        try:
            translator = MangaTranslatorRAG(
                csv_path=self.glossary_path, 
                api_key=self.api_key,
                extract_entities=self.extract_entities
            )
            existing_terms = translator.get_existing_chinese_terms()

            # Extract new entities before translating (runs once per image)
            if self.extract_entities:
                all_chinese_text = " ".join([d.get("text", "") for d in text_data if d.get("text")])
                if all_chinese_text:
                    print("🕵️‍♀️ กำลังวิเคราะห์หน้ามังงะเพื่อหาชื่อเฉพาะใหม่...")
                    new_entities = translator.extract_new_entities(all_chinese_text, existing_terms=existing_terms)

                    if new_entities:
                        print(f"   ✨ พบคำศัพท์ใหม่ {len(new_entities)} คำ: {new_entities}")
                        update_glossary_auto(new_entities, csv_path=self.glossary_path)
                        translator.refresh_glossary()
                        print("🔄 รีโหลดฐานข้อมูลคำศัพท์เรียบร้อย")
                    else:
                        print("   🔍 ไม่พบคำศัพท์เฉพาะใหม่ในหน้านี้")
            else:
                print("   ⏭️ ข้ามการสกัดคำศัพท์ใหม่ (เพื่อประหยัด token)")

            for data in text_data:
                chinese_text = data.get("text", "")
                b_id = str(data["bubble_id"])
                if chinese_text:
                    print(f"   🇨🇳 ต้นฉบับ: {chinese_text}")
                    thai_translation = translator.translate(chinese_text)
                    data["translated_text"] = thai_translation
                    translated_dict[b_id] = thai_translation
                    print(f"   🇹🇭 คำแปล: {thai_translation}\n")
                else:
                    data["translated_text"] = ""
                    translated_dict[b_id] = ""

        except Exception as e:
            print(f"❌ [RAG] เกิดข้อผิดพลาดในการแปล: {e}")
            for data in text_data:
                data["translated_text"] = data.get("text", "")

        return translated_dict

    def _typeset(self, cleaned_img, text_data, final_output_img):
        print("✍️ [3/3] กำลังประกอบร่างข้อความแปลลงบนภาพ...")

        if not os.path.exists(self.font_path):
            print(f"   ❌ ไม่พบไฟล์ฟอนต์ที่ {self.font_path} ข้ามขั้นตอนการพิมพ์ข้อความ")
            return

        typesetter = MangaTypesetter(font_path=self.font_path, font_size=26)
        final_img = typesetter.draw_text(cleaned_img, text_data)
        cv2_imwrite_unicode(final_output_img, final_img)
        print("🎉 พิมพ์ข้อความแปลไทยลงภาพเรียบร้อย!")
