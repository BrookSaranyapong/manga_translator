import os
from modules.pipeline import MangaPipeline

def main():
    input_dir = 'image/input'
    output_dir = 'image/output'
    model_path = 'models/model.pt'
    font_path = 'fonts/THSarabun.ttf'
    api_key = "Your_token"

    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    print("กำลังเตรียม AI...")
    pipeline = MangaPipeline(model_path=model_path, font_path=font_path, api_key=api_key)

    filename = '5 ตอนที่ 5 罗斯袭杀.jpg'
    input_file = os.path.join(input_dir, filename)

    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์ภาพที่: {input_file}")
        return

    pipeline.run(input_file, output_dir)

if __name__ == "__main__":
    main()
