import numpy as np
from ultralytics import YOLO

class YoloBubbleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img, confidence_threshold=0.5):
        results = self.model(img, conf=confidence_threshold)
        detected_bubbles = []
        all_detections = []  # ✨ สำหรับ Debug

        for result in results:
            if result.masks is None:
                continue
                
            for box, mask in zip(result.boxes, result.masks):
                class_id = int(box.cls[0].cpu().numpy())
                b = box.xyxy[0].cpu().numpy().astype(int)
                x, y = int(b[0]), int(b[1])
                w, h = int(b[2] - b[0]), int(b[3] - b[1])
                conf = float(box.conf[0].cpu().numpy())
                polygon = mask.xy[0].astype(np.int32)
                
                # ✨ เพิ่ม: บันทึกทุก Detection เพื่อ Debug
                det_info = {
                    "group": "face" if self._is_face_detection(w, h) else "bubble",
                    "class_id": class_id,
                    "conf": conf,
                    "size": f"{w}x{h}",
                    "aspect_ratio": round(w/h if h > 0 else 0, 2)
                }
                all_detections.append(det_info)
                
                # ✨ ตัวกรอง 1: ตรวจสอบ class ID
                if class_id != 0:
                    continue
                
                # ✨ ตัวกรอง 2: ลบ Detection ที่เหมือน Face อย่างชัดเจน
                if self._is_face_detection(w, h):
                    continue
                
                detected_bubbles.append({
                    "polygon": polygon,
                    "position": {"x": x, "y": y, "w": w, "h": h},
                    "confidence": conf
                })
        
        # ---------------------------------------------------------
        # ✨ เพิ่มบรรทัดนี้: เรียงลำดับ Bubble จากบนลงล่าง (ตามค่า Y)
        # ถ้าระดับความสูงใกล้เคียงกัน (บรรทัดเดียวกัน) ให้เรียงจากซ้ายไปขวา (ตามค่า X)
        # ---------------------------------------------------------
        
        # ✨ Debug Output
        print(f"\n🔍 Debug YOLO Detections:")
        print(f"   ✅ Valid Bubbles: {len(detected_bubbles)} | ⛔ Filtered out: {len(all_detections) - len(detected_bubbles)}")
        for det in all_detections:
            marker = "✅" if det['group'] == 'bubble' else "⛔"
            print(f"   {marker} Class: {det['class_id']} | Group: {det['group']:8} | Size: {det['size']:8} | Ratio: {det['aspect_ratio']:5} | Conf: {det['conf']:.2f}")
        
        detected_bubbles = sorted(detected_bubbles, key=lambda b: (b['position']['y'] // 30, b['position']['x']))
        
        return detected_bubbles
    
    def _is_face_detection(self, w, h):
        """ 
        ✨ ฟังก์ชั่นตรวจสอบว่า Detection นี้เป็น Face หรือไม่
        *** Conservative: ไม่กรอง Bubbles ปกติ เฉพาะ Faces ที่ชัดเจน ***
        """
        if h == 0:
            return False
        
        aspect_ratio = w / h
        area = w * h
        
        # ✨ คุณลักษณะของตัวละคร Face ที่ควร Filter:
        # - BOTH เงื่อนไข:
        #   1. ขนาด VERY LARGE (> 100,000)
        #   2. ใกล้จตุรัส (0.9 < ratio < 1.1)
        # Only filter things that REALLY look like faces
        is_nearly_square = 0.9 < aspect_ratio < 1.1
        is_very_large = area > 100000
        
        return is_nearly_square and is_very_large