import os
import pandas as pd
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Singleton: _embeddings is loaded once and reused across all translator
# instances.  Vector-db (Chroma) is lightweight to rebuild from the glossary,
# so it stays per-instance; the heavy embedding model does not.
# ---------------------------------------------------------------------------
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("🧠 [RAG] กำลังโหลด embedding model (ครั้งเดียว)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _embeddings


class MangaTranslatorRAG:
    def __init__(self, csv_path="glossary.csv", api_key=None, extract_entities=True):
        self.embeddings = _get_embeddings()
        self.glossary_texts = self._load_csv_to_texts(csv_path)
        self.extract_entities = extract_entities  # Flag to enable/disable entity extraction

        if self.glossary_texts:
            self.vector_db = Chroma.from_texts(self.glossary_texts, self.embeddings)
        else:
            self.vector_db = None

        if not api_key:
            raise ValueError("❌ ต้องใส่ API Key สำหรับ OpenAI")

        self.api_key = api_key
        self.csv_path = csv_path
        self._llm = None
        self._prompt_template = None

    # Lazy-init LLM so it's created only once per translator instance
    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                temperature=0.3, model="gemini-2.5-flash", google_api_key=self.api_key
            )
        return self._llm

    @property
    def prompt_template(self):
        if self._prompt_template is None:
            self._prompt_template = PromptTemplate(
                input_variables=["context", "raw_text"],
                template="""Translate to Thai manhua. Natural, conversational tone. Use glossary strictly. Follow Thai word order.

[Glossary]
{context}

{raw_text}"""
            )
        return self._prompt_template

    def refresh_glossary(self):
        """Reload glossary CSV and rebuild vector-db WITHOUT re-loading models."""
        self.glossary_texts = self._load_csv_to_texts(self.csv_path)
        if self.glossary_texts:
            self.vector_db = Chroma.from_texts(self.glossary_texts, self.embeddings)
        else:
            self.vector_db = None

    def _load_csv_to_texts(self, csv_path):
        if not os.path.exists(csv_path):
            return []
        try:
            df = pd.read_csv(csv_path)
            texts = []
            for _, row in df.iterrows():
                cn, th, note = row.get('Chinese', ''), row.get('Thai', ''), row.get('Note', '')
                if bool(pd.notna(cn)) and bool(pd.notna(th)):
                    texts.append(f"คำศัพท์: '{cn}' แปลว่า '{th}' (หมายเหตุ: {note})")
            return texts
        except Exception:
            return []

    def get_existing_chinese_terms(self):
        """ดึงรายการศัพท์จีนที่มีอยู่ในฐานข้อมูล"""
        if not os.path.exists(self.csv_path):
            return []
        try:
            df = pd.read_csv(self.csv_path)
            chinese_col = df['Chinese'] if 'Chinese' in df.columns else pd.Series()
            return chinese_col.astype(str).str.strip().tolist()
        except Exception:
            return []

    def translate(self, chinese_text):
        context = "ไม่มีคู่มือคำศัพท์"
        if self.vector_db:
            docs = self.vector_db.similarity_search(chinese_text, k=5)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])

        final_prompt = self.prompt_template.format(context=context, raw_text=chinese_text)
        return str(self.llm.invoke(final_prompt).content).strip()

    def extract_new_entities(self, text_chunk, existing_terms=None):
        """ให้ AI สแกนหาคำศัพท์เฉพาะจากข้อความ (เฉพาะที่ยังไม่มีในฐานข้อมูล)"""
        if not text_chunk.strip():
            return []

        if existing_terms is None:
            existing_terms = self.get_existing_chinese_terms()

        print("🔍 [RAG] กำลังให้ AI สแกนหาคำศัพท์ใหม่จากหน้ามังงะ...")

        existing_terms_str = ", ".join([f"'{t}'" for t in existing_terms[:30]])

        extract_prompt = PromptTemplate(
            input_variables=["text", "existing"],
            template="""Extract NEW entities: Character, Location, Skill names.
Fix OCR typos. Skip if exists: {existing}
Confidence > 90% only. Max 5. JSON output.
[Chinese, Thai, Note]

{text}
"""
        )

        try:
            prompt_str = extract_prompt.format(text=text_chunk, existing=existing_terms_str)
            response = str(self.llm.invoke(prompt_str).content).strip()

            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            new_terms = json.loads(response.strip())
            
            # Double-check: filter out any that somehow still match existing terms
            filtered_terms = []
            for term in new_terms:
                chinese = term.get('Chinese', '').strip()
                if chinese and chinese not in existing_terms:
                    filtered_terms.append(term)
            
            return filtered_terms
        except Exception as e:
            print(f"⚠️ [RAG] AI สกัดคำศัพท์ล้มเหลว: {e}")
            return []