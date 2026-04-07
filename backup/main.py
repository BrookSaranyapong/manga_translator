import cv2
import os

# นำเข้าคลาส AI ทั้ง 2 ตัวที่เราแยกหน้าที่กันไว้
from modules.detection.yolo_detect_bubbles import YoloBubbleDetector
from modules.ocr.image_processor import MangaCleaner
from modules.utils.io import save_to_json

def main():
    # 1. ตั้งค่าที่อยู่ไฟล์ให้เป็นระบบ
    input_dir = 'image/input'
    output_dir = 'image/output'
    os.makedirs(output_dir, exist_ok=True) # สร้างโฟลเดอร์ output ถ้ายังไม่มี

    filename = '5 05.jpg' # ชื่อรูป
    input_file = os.path.join(input_dir, filename)
    output_img = os.path.join(output_dir, f'cleaned_{filename}')
    output_json = os.path.join(output_dir, 'extracted_text.json')
    
    # กำหนดที่อยู่ไฟล์โมเดล YOLO (ตั้งให้ตรงกับที่คุณมี)
    model_path = os.path.join('models', 'model.pt')

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    # 2. เรียกใช้คลาสเตรียมไว้ทั้ง 2 ตัว
    print("กำลังเตรียม AI...")
    detector = YoloBubbleDetector(model_path)
    cleaner = MangaCleaner()

    # 3. อ่านรูป
    img = cv2.imread(input_file)
    if img is None:
        print(f"❌ ไม่พบไฟล์ภาพที่: {input_file}")
        return

    # 4. ประมวลผลขั้นที่ 1: ให้ YOLO สแกนหา "พื้นที่ปลอดภัย (กรอบคำพูด)" ก่อน
    print("🤖 [1/2] YOLO กำลังค้นหาพื้นที่กรอบคำพูด...")
    detected_bubbles = detector.detect(img, confidence_threshold=0.5)

    # 5. ประมวลผลขั้นที่ 2: ส่งภาพและข้อมูลพื้นที่ปลอดภัยให้ EasyOCR ทำงานต่อ
    print("👁️ [2/2] EasyOCR กำลังอ่านข้อความและทำความสะอาดภาพ...")
    final_img, text_data = cleaner.process_image(img, detected_bubbles)

    # 6. บันทึกผล
    cv2.imwrite(output_img, final_img)
    save_to_json(text_data, output_json)
    
    print(f"✅ เสร็จแล้ว! สกัดข้อความสำเร็จ {len(text_data)} ประโยค")
    print(f"📂 ผลลัพธ์อยู่ที่โฟลเดอร์ {output_dir}")

if __name__ == "__main__":
    main()