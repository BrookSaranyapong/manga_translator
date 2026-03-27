import cv2
import numpy as np
import easyocr

class MangaCleaner:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim'])

    def process_image(self, img, detected_bubbles):
        h, w = img.shape[:2]
        valid_bubble_id = 1
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        cleaned_text_data = []

        for bubble_idx, bubble in enumerate(detected_bubbles):
            bx, by, bw, bh = bubble["position"]["x"], bubble["position"]["y"], bubble["position"]["w"], bubble["position"]["h"]

            pad = 10
            y1, y2 = max(0, by - pad), min(h, by + bh + pad)
            x1, x2 = max(0, bx - pad), min(w, bx + bw + pad)
            crop = img[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                continue

            # --- OCR ---
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_enhanced = cv2.equalizeHist(crop_gray)
            _, crop_binary = cv2.threshold(crop_enhanced, 60, 255, cv2.THRESH_BINARY)
            crop_enhanced_bgr = cv2.cvtColor(crop_binary, cv2.COLOR_GRAY2BGR)

            results = self.reader.readtext(crop)
            if len(results) == 0:
                results = self.reader.readtext(crop_enhanced_bgr)

            # --- bubble polygon mask ---
            bubble_poly_mask = np.zeros((h, w), dtype=np.uint8)
            polygon = bubble.get("polygon", None)
            if polygon is not None and len(polygon) > 0:
                cv2.fillPoly(bubble_poly_mask, [polygon], 255)
            else:
                cv2.rectangle(bubble_poly_mask, (bx, by), (bx + bw, by + bh), 255, -1)

            # ✨ shrink แรงขึ้น (15px) เพื่อตัดเส้นขอบ bubble ออก
            shrunk_bubble_mask = cv2.erode(
                bubble_poly_mask,
                np.ones((15, 15), np.uint8),
                iterations=2
            )

            # dark pixels ในภาพ
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, dark_pixels = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

            # ตัวอักษรดำที่อยู่ใน shrunk area (ไม่รวมเส้นขอบ)
            text_candidates = cv2.bitwise_and(shrunk_bubble_mask, dark_pixels)

            if len(results) > 0:
                print(f"   ✅ Bubble #{bubble_idx+1}: {len(results)} text(s) detected")
                combined_text = ""
                all_points = []
                overall_conf = 0.0
                text_count = 0

                for (bbox, text, prob) in results:
                    if prob < 0.01:
                        continue
                    combined_text += text
                    for point in bbox:
                        all_points.append([int(point[0] + x1), int(point[1] + y1)])
                    overall_conf += prob
                    text_count += 1
                    print(f"       └─ \"{text}\" ({prob:.3f})")

                if text_count > 0 and len(all_points) > 0:
                    all_points_arr = np.array(all_points, dtype=np.int32)
                    min_x = int(np.min(all_points_arr[:, 0]))
                    max_x = int(np.max(all_points_arr[:, 0]))
                    min_y = int(np.min(all_points_arr[:, 1]))
                    max_y = int(np.max(all_points_arr[:, 1]))

                    cleaned_text_data.append({
                        "bubble_id": valid_bubble_id,
                        "text": combined_text.strip(),
                        "position": [[int(min_x), int(min_y)], [int(max_x), int(min_y)],
                                     [int(max_x), int(max_y)], [int(min_x), int(max_y)]],
                        "confidence": float(overall_conf / text_count)
                    })

                    # ✨ ใช้ connected components กรอง
                    # เพื่อเอาเฉพาะ cluster ที่ overlap กับ OCR bbox เท่านั้น
                    # วิธีนี้จับ ! ที่หลุดจาก OCR ได้ โดยไม่กินเส้นขอบ
                    ocr_padding = 25
                    ocr_region = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(ocr_region,
                                  (max(0, min_x - ocr_padding), max(0, min_y - ocr_padding)),
                                  (min(w, max_x + ocr_padding), min(h, max_y + ocr_padding)),
                                  255, -1)

                    # หา connected components ใน text_candidates
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        text_candidates, connectivity=8
                    )

                    final_text_mask = np.zeros((h, w), dtype=np.uint8)
                    for label_id in range(1, num_labels):  # skip background (0)
                        component_mask = (labels == label_id).astype(np.uint8) * 255
                        # เอาเฉพาะ component ที่ overlap กับ OCR region
                        overlap = cv2.bitwise_and(component_mask, ocr_region)
                        if cv2.countNonZero(overlap) > 0:
                            final_text_mask = cv2.bitwise_or(final_text_mask, component_mask)

                    final_text_mask = cv2.dilate(
                        final_text_mask, np.ones((5, 5), np.uint8), iterations=2
                    )
                    inpaint_mask = cv2.bitwise_or(inpaint_mask, final_text_mask)
                    valid_bubble_id += 1

            else:
                print(f"   ❌ Bubble #{bubble_idx+1}: No text or face detected by OCR")

        # ไม่ต้อง Dilate เพิ่มแล้ว แค่เกลี่ยช่องโหว่เล็กๆ ภายในตัวหนังสือด้วย MORPH_CLOSE ก็พอ
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        cleaned_img = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)

        return cleaned_img, cleaned_text_data