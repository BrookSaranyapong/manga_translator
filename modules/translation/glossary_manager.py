import pandas as pd
import os

def update_glossary_auto(new_terms, csv_path="glossary.csv"):
    """
    new_terms: list ของ dict เช่น [{'Chinese': '云韵', 'Thai': 'อวิ๋นยุ่น', 'Note': 'ตัวละครใหม่'}]
    """
    if not new_terms:
        return

    # Normalize whitespace on Chinese keys before dedup
    for term in new_terms:
        if 'Chinese' in term and isinstance(term['Chinese'], str):
            term['Chinese'] = term['Chinese'].strip()

    # 1. โหลดไฟล์เดิมที่มีอยู่
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Chinese'] = df['Chinese'].str.strip()
    else:
        df = pd.DataFrame(columns=['Chinese', 'Thai', 'Note'])

    # 2. แปลงข้อมูลใหม่เป็น DataFrame
    new_df = pd.DataFrame(new_terms)

    # 3. รวมร่างข้อมูล และลบตัวที่ซ้ำ (เอาตามตัวอักษรจีน)
    # keep='first' คือถ้าซ้ำให้เอาตัวที่มีอยู่เดิมไว้
    combined_df = pd.concat([df, new_df]).drop_duplicates(subset=['Chinese'], keep='first')

    # 4. เซฟกลับลงไฟล์ ใช้ utf-8-sig เพื่อเปิดใน Excel ให้ภาษาไทย/จีน ไม่เพี้ยน
    combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✨ [Glossary] อัปเดตคำศัพท์ใหม่ {len(combined_df) - len(df)} คำ ลงในไฟล์เรียบร้อย!")