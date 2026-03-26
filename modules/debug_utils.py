import cv2
import os

def draw_detected_bubbles(image, detected_bubbles):
    """
    วาดเส้นขอบสีเขียวล้อมรอบกรอบคำพูด พร้อมตัวเลขกำกับ
    """
    debug_image = image.copy()
    for i, bubble in enumerate(detected_bubbles):
        polygon = bubble["polygon"]
        if len(polygon) > 0:
            # ✨ วาดเส้นขอบสีเขียว (หนาขึ้น)
            cv2.polylines(debug_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
            
            x, y = bubble["position"]["x"], bubble["position"]["y"]
            text_pos = (x, y - 5) if (y - 5) > 0 else (x, y + 20)
            cv2.putText(debug_image, f"#{i+1}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return debug_image

def save_cropped_bubbles(image, detected_bubbles, output_dir, filename):
    """
    ตัดภาพเฉพาะส่วน Bubble ออกมาเป็นไฟล์ย่อยๆ เพื่อง่ายต่อการตรวจสอบ
    """
    base_name = os.path.splitext(filename)[0]
    crop_dir = os.path.join(output_dir, f"crops_{base_name}")
    os.makedirs(crop_dir, exist_ok=True)

    for i, bubble in enumerate(detected_bubbles):
        x = bubble["position"]["x"]
        y = bubble["position"]["y"]
        w = bubble["position"]["w"]
        h = bubble["position"]["h"]

        y1, y2 = max(0, y), min(image.shape[0], y + h)
        x1, x2 = max(0, x), min(image.shape[1], x + w)

        cropped_img = image[y1:y2, x1:x2]
        
        if cropped_img.size > 0:
            crop_path = os.path.join(crop_dir, f"bubble_{i+1}.jpg")
            cv2.imwrite(crop_path, cropped_img)