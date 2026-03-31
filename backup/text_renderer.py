import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pythainlp.tokenize import word_tokenize # ✨ นำเข้า AI ตัดคำภาษาไทย

class MangaTypesetter:
    def __init__(self, font_path, font_size=24, text_color=(0, 0, 0)):
        self.font_path = font_path
        self.font_size = font_size
        self.text_color = text_color 

    def draw_text(self, cv_img, text_data_list):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            print(f"❌ Error: หาฟอนต์ไม่เจอที่ {self.font_path}")
            return cv_img

        for data in text_data_list:
            text_to_draw = data.get("translated_text", data.get("text", ""))
            if not text_to_draw:
                continue
            
            pts = np.array(data["position"], dtype=np.int32)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            
            box_w = max_x - min_x
            box_h = max_y - min_y
            center_x = min_x + (box_w // 2)
            center_y = min_y + (box_h // 2)

            # ✨ 1. ระบบตัดคำภาษาไทยอัจฉริยะ (Smart Thai Word Wrap) ✨
            max_text_width = box_w * 0.85 # กำหนดพื้นที่ปลอดภัย 85% ของความกว้างบับเบิ้ล
            words = word_tokenize(text_to_draw, engine="newmm") # ให้ AI หั่นเป็นคำๆ
            
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + word
                # วัดความกว้างพิกเซลจริงๆ ของบรรทัดนี้ว่าล้นบับเบิ้ลหรือยัง
                test_width = font.getlength(test_line) 
                
                if test_width <= max_text_width:
                    current_line = test_line
                else:
                    # ถ้าล้นแล้ว ให้เอาบรรทัดที่แล้วเก็บไว้ แล้วขึ้นบรรทัดใหม่ด้วยคำปัจจุบัน
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
                
            wrapped_text = "\n".join(lines) # ประกอบร่างกลับเป็นก้อนข้อความ

            # ✨ 2. วัดขนาดก้อนข้อความเพื่อหาจุดกึ่งกลางเป๊ะๆ ✨
            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            final_x = center_x - (text_w / 2)
            final_y = center_y - (text_h / 2) - (self.font_size * 0.15) 

            # ✨ วาดลงไปแบบเป๊ะๆ พร้อมบังคับให้รองรับสระซ้อน
            draw.multiline_text(
                (final_x, final_y), 
                wrapped_text, 
                font=font, 
                fill=self.text_color, 
                align="center"
                # เพิ่มพารามิเตอร์นี้เพื่อบอกให้ Pillow ใช้ระบบจัดเรียงข้อความขั้นสูง (ถ้าติดตั้ง raqm แล้ว)
                # ถ้าไม่ได้ติดตั้ง raqm พารามิเตอร์นี้อาจจะไม่มีผลมากนัก
            )

        final_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return final_cv_img