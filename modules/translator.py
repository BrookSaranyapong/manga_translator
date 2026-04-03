import os
import pandas as pd
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class MangaTranslatorRAG:
    def __init__(self, csv_path="glossary.csv", api_key=None):
        print("🧠 [RAG] กำลังเตรียมฐานข้อมูลคำศัพท์และโหลดโมเดล...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.glossary_texts = self._load_csv_to_texts(csv_path)
        
        if self.glossary_texts:
            self.vector_db = Chroma.from_texts(self.glossary_texts, self.embeddings)
        else:
            self.vector_db = None

        if not api_key:
            raise ValueError("❌ ต้องใส่ API Key สำหรับ OpenAI")
            
        self.llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-2.5-flash", google_api_key=api_key)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "raw_text"],
            template="""Translate to Thai (Manhua style).

    Rules:
    1. Use Glossary strictly.
    2. Fix OCR errors by context.
    3. Manhua tone only.
    4. Title before name.
    5. Output Thai only.

    [Glossary]
    {context}

    {raw_text}
    """
        )

    def _load_csv_to_texts(self, csv_path):
        if not os.path.exists(csv_path): return []
        try:
            df = pd.read_csv(csv_path)
            texts = []
            for _, row in df.iterrows():
                cn, th, note = row.get('Chinese', ''), row.get('Thai', ''), row.get('Note', '')
                if pd.notna(cn) and pd.notna(th):
                    texts.append(f"คำศัพท์: '{cn}' แปลว่า '{th}' (หมายเหตุ: {note})")
            return texts
        except: return []

    def translate(self, chinese_text):
        context = "ไม่มีคู่มือคำศัพท์"
        if self.vector_db:
            docs = self.vector_db.similarity_search(chinese_text, k=2)
            if docs: context = "\n".join([doc.page_content for doc in docs])
                
        final_prompt = self.prompt_template.format(context=context, raw_text=chinese_text)
        return self.llm.invoke(final_prompt).content.strip()
    
    def extract_new_entities(self, text_chunk):
        """ให้ AI สแกนหาคำศัพท์เฉพาะจากข้อความ แล้วส่งกลับมาเป็น List of Dict"""
        if not text_chunk.strip():
            return []

        print("🔍 [RAG] กำลังให้ AI สแกนหาคำศัพท์เฉพาะจากหน้ามังงะ...")
        
        extract_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract entities Manhua (Character, Location, Skill) from text.

        Rules:
        - Fix OCR errors using context.
        - Do not invent entities.
        - Use consistent Thai naming.
        - Deduplicate results.

        Output: JSON array only
        Keys: Chinese, Thai, Note

        {text}
        """
        )
        
        try:
            # สั่งให้ AI วิเคราะห์
            prompt_str = extract_prompt.format(text=text_chunk)
            response = self.llm.invoke(prompt_str).content.strip()
            
            # ลบพวก markdown code block ที่ AI ชอบแถมมา (เช่น ```json ... ```)
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
                
            new_terms = json.loads(response.strip())
            return new_terms
        except Exception as e:
            print(f"⚠️ [RAG] AI สกัดคำศัพท์ล้มเหลว หรือไม่พบคำศัพท์เฉพาะ: {e}")
            return []