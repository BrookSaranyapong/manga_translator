import uharfbuzz as hb
import freetype
import numpy as np
from PIL import Image
import cv2

class ThaiTextRenderer:
    """Renderer ภาษาไทยที่ใช้ HarfBuzz สำหรับ text shaping ที่ถูกต้อง"""
    
    def __init__(self, font_path, font_size=24):
        self.font_path = font_path
        self.font_size = font_size
        self._init_harfbuzz()
        self._init_freetype()
    
    def _init_harfbuzz(self):
        with open(self.font_path, 'rb') as f:
            font_data = f.read()
        self.hb_blob = hb.Blob(font_data)  # type: ignore[reportAttributeAccessIssue]
        self.hb_face = hb.Face(self.hb_blob)  # type: ignore[reportAttributeAccessIssue]
        self.hb_font = hb.Font(self.hb_face)  # type: ignore[reportAttributeAccessIssue]
        self.hb_font.scale = (self.font_size * 64, self.font_size * 64)
    
    def _init_freetype(self):
        self.ft_face = freetype.Face(self.font_path)
        self.ft_face.set_char_size(self.font_size * 64)
    
    def shape_text(self, text):
        buf = hb.Buffer()  # type: ignore[reportAttributeAccessIssue]
        buf.add_str(text)
        buf.guess_segment_properties()
        features = {"kern": True, "liga": True, "mark": True, "mkmk": True}
        hb.shape(self.hb_font, buf, features)  # type: ignore[reportAttributeAccessIssue]
        return buf.glyph_infos, buf.glyph_positions
    
    def render_text_to_image(self, text, text_color=(0, 0, 0),
                              stroke_width=0, stroke_color=(255, 255, 255)):
        """
        Render ข้อความเป็น RGBA image
        - stroke_width: ความหนาขอบ (0 = ปิด)
        - stroke_color: สีขอบ (default ขาว)
        """
        glyph_infos, glyph_positions = self.shape_text(text)
        
        total_width = sum(pos.x_advance for pos in glyph_positions) // 64
        total_width = max(total_width, 1)
        
        metrics = self.ft_face.size
        ascender  = metrics.ascender  // 64
        descender = abs(metrics.descender // 64)
        
        # ✨ เผื่อพื้นที่เพิ่มสำหรับ stroke
        padding = stroke_width + 2
        total_height = ascender + descender + self.font_size // 2 + padding * 2
        canvas_w = total_width + self.font_size + padding * 2

        canvas = np.zeros((total_height, canvas_w, 4), dtype=np.uint8)
        
        pen_x = padding
        pen_y = ascender + self.font_size // 4 + padding
        
        for info, pos in zip(glyph_infos, glyph_positions):
            glyph_id = info.codepoint
            self.ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)  # type: ignore[reportAttributeAccessIssue]
            bitmap = self.ft_face.glyph.bitmap
            
            if bitmap.width > 0 and bitmap.rows > 0:
                x = pen_x + (pos.x_offset // 64) + self.ft_face.glyph.bitmap_left
                y = pen_y - (pos.y_offset // 64) - self.ft_face.glyph.bitmap_top
                x = max(0, int(x))
                y = max(0, int(y))
                
                bm = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)
                
                y_end = min(y + bitmap.rows, canvas.shape[0])
                x_end = min(x + bitmap.width, canvas.shape[1])
                bm_y_end = y_end - y
                bm_x_end = x_end - x
                
                if bm_y_end > 0 and bm_x_end > 0:
                    alpha = bm[:bm_y_end, :bm_x_end]
                    canvas[y:y_end, x:x_end, 0] = text_color[0]
                    canvas[y:y_end, x:x_end, 1] = text_color[1]
                    canvas[y:y_end, x:x_end, 2] = text_color[2]
                    canvas[y:y_end, x:x_end, 3] = alpha
            
            pen_x += pos.x_advance // 64
        
        text_img = Image.fromarray(canvas, 'RGBA')

        # ✨ Trick 1: Stroke ด้วย cv2.dilate() บน alpha channel
        if stroke_width > 0:
            alpha_ch = canvas[:, :, 3]
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (stroke_width * 2 + 1, stroke_width * 2 + 1)
            )
            stroke_alpha = cv2.dilate(alpha_ch, kernel)
            
            stroke_layer = np.zeros_like(canvas)
            stroke_layer[:, :, 0] = stroke_color[0]
            stroke_layer[:, :, 1] = stroke_color[1]
            stroke_layer[:, :, 2] = stroke_color[2]
            stroke_layer[:, :, 3] = stroke_alpha
            
            stroke_img = Image.fromarray(stroke_layer, 'RGBA')
            # วาง stroke ก่อน แล้วทับด้วยตัวหนังสือจริง
            text_img = Image.alpha_composite(stroke_img, text_img)
        
        return text_img, total_height
    
    def measure_text(self, text):
        _, glyph_positions = self.shape_text(text)
        width = sum(pos.x_advance for pos in glyph_positions) // 64
        return width, self.font_size + self.font_size // 2


class MangaTypesetter:
    def __init__(self, font_path, font_size=24, text_color=(0, 0, 0),
                 line_spacing=0.3,           # ✨ Trick 2: ปรับระยะห่างบรรทัด
                 stroke_width=0,             # ✨ Trick 1: ความหนาขอบ (0=ปิด)
                 stroke_color=(255, 255, 255) # ✨ Trick 1: สีขอบ
                 ):
        self.font_size = font_size
        self.text_color = text_color
        self.line_spacing = line_spacing
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.renderer = ThaiTextRenderer(font_path, font_size)

    def _wrap_text(self, text, max_width):
        from pythainlp.tokenize import word_tokenize
        words = word_tokenize(text, engine="newmm")
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word
            w, _ = self.renderer.measure_text(test_line)
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def draw_text(self, cv_img, text_data_list):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGBA")

        for data in text_data_list:
            text_to_draw = data.get("translated_text", data.get("text", ""))
            if not text_to_draw:
                continue
            
            pts = np.array(data["position"], dtype=np.int32)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            box_w = max_x - min_x
            center_x = min_x + box_w // 2
            center_y = min_y + (max_y - min_y) // 2

            lines = self._wrap_text(text_to_draw, int(box_w * 0.85))

            # ✨ Trick 2: ใช้ self.line_spacing แทน hardcode 0.3
            line_height = self.font_size + int(self.font_size * self.line_spacing)
            total_text_h = len(lines) * line_height
            start_y = center_y - total_text_h // 2

            for i, line in enumerate(lines):
                # ✨ ส่ง stroke_width และ stroke_color เข้าไปด้วย
                line_img, _ = self.renderer.render_text_to_image(
                    line,
                    text_color=self.text_color,
                    stroke_width=self.stroke_width,
                    stroke_color=self.stroke_color
                )
                line_w, line_h = line_img.size
                paste_x = center_x - line_w // 2
                paste_y = start_y + i * line_height
                pil_img.paste(line_img, (paste_x, paste_y), line_img)

        final_rgb = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(final_rgb), cv2.COLOR_RGB2BGR)