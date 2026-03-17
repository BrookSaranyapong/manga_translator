# 🤖 Manga Bubble Cleaner & Text Extractor

โปรแกรมอัตโนมัติสำหรับ **ลบข้อความในฟองคำพูด (Speech Bubbles)** ของมังงะ/ม่านฮวา พร้อมดึงข้อความต้นฉบับออกมาเป็นไฟล์ **JSON** เพื่อใช้สำหรับการแปลหรือการทำ Type-setting ต่อไป

## ✨ ความสามารถ (Features)

- **Text Detection:** ใช้ AI (EasyOCR) ในการหาตำแหน่งตัวอักษรภาษาจีน
- **Smart Cleaning:** ระบบลบข้อความอัจฉริยะที่เช็คความสว่างพื้นหลังก่อนลบ (ป้องกันการลบรายละเอียดภาพ เช่น ปากคน หรือเส้นผม)
- **JSON Extraction:** เก็บข้อมูลข้อความ, พิกัดตำแหน่ง, และค่าความเชื่อมั่น (Confidence) ลงไฟล์ JSON
- **Modular Design:** แยกส่วนการประมวลผลภาพและการจัดการข้อมูลออกจากกันเพื่อให้ง่ายต่อการพัฒนาต่อ

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
clean/
├── main.py                # ไฟล์หลักสำหรับรันโปรแกรม
├── modules/               # โฟลเดอร์เก็บโมดูลย่อย
│   ├── image_processor.py # ระบบประมวลผลภาพและ AI
│   └── save_to_json.py    # ระบบจัดการข้อมูล JSON
├── image/                 # โฟลเดอร์เก็บรูปภาพ
│   ├── input/             # ใส่รูปต้นฉบับที่นี่
│   └── output/            # รูปที่คลีนแล้วและไฟล์ JSON จะอยู่ที่นี่
├── requirements.txt       # รายการ Library ที่ต้องใช้
└── .gitignore             # ไฟล์ยกเว้นการอัปโหลดไฟล์ขยะขึ้น Git
```

## 🚀 วิธีใช้งาน (How to Use)

# สร้าง venv

```text
python -m venv .venv
```

# เปิดการใช้งาน venv (Windows)

```text
.venv\Scripts\activate
```

# ติดตั้ง Library

```text
pip install -r requirements.txt
```

# การใช้งาน

```text
python main.py
```
