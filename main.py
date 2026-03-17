import cv2
import os
# เรียกใช้จากโฟลเดอร์ modules
from modules.image_processor import MangaCleaner
from modules.save_to_json import save_to_json

def main():
    # 1. ตั้งค่าที่อยู่ไฟล์ให้เป็นระบบ
    input_dir = 'image/input'
    output_dir = 'image/output'
    os.makedirs(output_dir, exist_ok=True) # สร้างโฟลเดอร์ output ถ้ายังไม่มี

    filename = 'image.jpg' # ชื่อรูปที่คุณมี
    input_file = os.path.join(input_dir, filename)
    output_img = os.path.join(output_dir, f'cleaned_{filename}')
    output_json = os.path.join(output_dir, 'extracted_text.json')

    # 2. เรียกใช้คลาสจากไฟล์ใน modules
    cleaner = MangaCleaner()

    # 3. อ่านรูป
    img = cv2.imread(input_file)
    if img is None:
        print(f"ไม่พบไฟล์ภาพที่: {input_file}")
        return

    # 4. ประมวลผล
    print("กำลังสแกนและลบข้อความ...")
    results = cleaner.detect_text(img)
    final_img, text_data = cleaner.process_image(img, results)

    # 5. บันทึกผล
    cv2.imwrite(output_img, final_img)
    save_to_json(text_data, output_json)
    print(f"เสร็จแล้ว! ผลลัพธ์อยู่ที่โฟลเดอร์ {output_dir}")

if __name__ == "__main__":
    main()