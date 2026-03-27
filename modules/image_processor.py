import cv2
import numpy as np
import easyocr

class MangaCleaner:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim'])
    
    def _is_valid_text(self, text, confidence):
        """
        Filter out corrupted/noise OCR results.
        Valid text should have meaningful content and reasonable confidence.
        """
        if not text or len(text.strip()) == 0:
            return False
        
        stripped = text.strip()
        
        # Punctuation and common symbols
        punctuation = set('!！？?｜|;；:：,，。.·～~-_（）()「」『』【】{}<>《》\'"\'""''…、')
        
        # Count actual content chars (non-punctuation, non-space)
        content_chars = [c for c in stripped if c not in punctuation and not c.isspace()]
        
        # No real content = filter
        if len(content_chars) == 0:
            return False
        
        # Single character - only accept if:
        # 1. No punctuation mixed in, OR
        # 2. High confidence
        if len(content_chars) == 1:
            punctuation_count = sum(1 for c in stripped if c in punctuation)
            if punctuation_count > 0 and confidence < 0.8:
                # Single char mixed with punctuation at low confidence = noise (like "一 !")
                return False
            return True
        
        # Multiple characters
        # Repeated characters like "吱吱" are OK (onomatopoeia/sound effects)
        # Just check if confidence is acceptable
        if len(stripped) >= 2 and confidence < 0.2:
            return False
        
        return True

    def process_image(self, img, detected_bubbles):
        h, w = img.shape[:2]
        valid_bubble_id = 1 
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        cleaned_text_data = []

        for bubble_idx, bubble in enumerate(detected_bubbles):
            bx, by, bw, bh = bubble["position"]["x"], bubble["position"]["y"], bubble["position"]["w"], bubble["position"]["h"]
            
            pad = 10
            y1, y2 = max(0, by-pad), min(h, by+bh+pad)
            x1, x2 = max(0, bx-pad), min(w, bx+bw+pad)
            crop = img[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                continue

            # Preprocessing - more lenient for small text
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_enhanced = cv2.equalizeHist(crop_gray)
            # ✨ Lower threshold to catch smaller text
            _, crop_binary = cv2.threshold(crop_enhanced, 60, 255, cv2.THRESH_BINARY)
            crop_enhanced = cv2.cvtColor(crop_binary, cv2.COLOR_GRAY2BGR)
            
            # ✨ Also try original crop without threshold
            results = self.reader.readtext(crop)
            if len(results) == 0:
                # Try with enhanced version if original fails
                results = self.reader.readtext(crop_enhanced)

            if len(results) > 0:
                print(f"   ✅ Bubble #{bubble_idx+1}: {len(results)} text(s) detected")
                
                # Filter corrupted texts first
                valid_results = []
                for (bbox, text, prob) in results:
                    if self._is_valid_text(text, prob):
                        valid_results.append((bbox, text, prob))
                        print(f"       └─ \"{text}\" ({prob:.3f})")
                    else:
                        print(f"       └─ [FILTERED] \"{text}\" ({prob:.3f}) - noise/corrupted")
                
                combined_text = ""
                all_points = []
                overall_conf = 0.0
                text_count = 0
                
                for (bbox, text, prob) in valid_results:
                    combined_text += text
                    for point in bbox:
                        all_points.append([int(point[0] + x1), int(point[1] + y1)])
                    
                    overall_conf += prob
                    text_count += 1

                if text_count > 0 and len(all_points) > 0:
                    all_points_arr = np.array(all_points, dtype=np.int32)
                    min_x = int(np.min(all_points_arr[:, 0]))
                    max_x = int(np.max(all_points_arr[:, 0]))
                    min_y = int(np.min(all_points_arr[:, 1]))
                    max_y = int(np.max(all_points_arr[:, 1]))
                    
                    cleaned_text_data.append({
                        "bubble_id": valid_bubble_id,
                        "text": combined_text.strip(),
                        "position": [[int(min_x), int(min_y)], [int(max_x), int(min_y)], [int(max_x), int(max_y)], [int(min_x), int(max_y)]],
                        "confidence": float(overall_conf / text_count)
                    })

                    # Mark text area for inpainting
                    erase_bbox = np.array([
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y]
                    ], dtype=np.int32)
                    
                    padding = 5
                    erase_bbox[0][0] = max(0, erase_bbox[0][0] - padding)
                    erase_bbox[0][1] = max(0, erase_bbox[0][1] - padding)
                    erase_bbox[1][0] = min(w, erase_bbox[1][0] + padding)
                    erase_bbox[1][1] = max(0, erase_bbox[1][1] - padding)
                    erase_bbox[2][0] = min(w, erase_bbox[2][0] + padding)
                    erase_bbox[2][1] = min(h, erase_bbox[2][1] + padding)
                    erase_bbox[3][0] = max(0, erase_bbox[3][0] - padding)
                    erase_bbox[3][1] = min(h, erase_bbox[3][1] + padding)
                    
                    cv2.fillPoly(inpaint_mask, [erase_bbox], 255)
                    
                    valid_bubble_id += 1
            else:
                print(f"   ❌ Bubble #{bubble_idx+1}: No text detected by OCR")

        # Apply inpainting to remove text
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5,5), np.uint8), iterations=2)
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)
        cleaned_img = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)

        return cleaned_img, cleaned_text_data