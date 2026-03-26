import json
import numpy as np

def merge_bubbles(json_data):
    """ รวมข้อความที่อยู่ bubble เดียวกันเข้าด้วยกัน """
    merged_results = {}
    for item in json_data:
        bid = item["bubble_id"]
        if bid not in merged_results:
            merged_results[bid] = {
                "bubble_id": bid,
                "full_text": item["text"],
                "confidence_avg": item["confidence"],
                "count": 1
            }
        else:
            merged_results[bid]["full_text"] += " " + item["text"]
            merged_results[bid]["confidence_avg"] += item["confidence"]
            merged_results[bid]["count"] += 1
            
    final_list = []
    for bid, data in merged_results.items():
        data["confidence_avg"] /= data["count"]
        del data["count"]
        final_list.append(data)
    
    return final_list

def convert_numpy(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj

def save_to_json(data, filename, merge=False):
    """ 
    บันทึกข้อมูล JSON 
    ถ้าตั้ง merge=True จะรวมประโยคให้ก่อนเซฟ 
    """
    if merge:
        data = merge_bubbles(data)
    
    # Convert numpy types to Python types
    data = convert_numpy(data)
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ บันทึกข้อมูล JSON เรียบร้อย: {filename}")