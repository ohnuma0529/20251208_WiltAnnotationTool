
import requests
import json

API_URL = "http://localhost:8001/api"

def test_preview_padding():
    # 1. Set Filter
    print("Setting filter...")
    requests.post(f"{API_URL}/set_filter", json={"unit": "31", "date": "20250430", "frequency": 30})

    # 2. Get Images
    res = requests.get(f"{API_URL}/images")
    frames = res.json()
    if not frames:
        print("No frames found")
        return

    # 3. Call preview_points with a known area
    # BBox: 300,300 to 500,500
    # Expected Points Min/Max: approx 300-500 depending on content.
    # Expected Result BBox: (Min-10, Min-10) to (Max+10, Max+10)
    
    bbox = {
        "x_min": 300,
        "y_min": 300,
        "x_max": 500,
        "y_max": 500
    }
    
    print(f"Requesting preview with BBox: {bbox}")
    try:
        res = requests.post(f"{API_URL}/preview_points", json={
            "frame_index": 0,
            "bbox": bbox
        })
        
        if res.status_code != 200:
            print(f"Error: {res.status_code} - {res.text}")
            return
            
        data = res.json()
        points = data['points']
        new_bbox = data['new_bbox']
        
        if not points:
            print("No points found. Cannot verify padding.")
            return

        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        print(f"Points Bounds: x[{min_x:.1f}, {max_x:.1f}], y[{min_y:.1f}, {max_y:.1f}]")
        print(f"Result BBox: {new_bbox}")
        
        # Verify Padding (Allow small float error)
        pad = 10
        tol = 0.5
        
        check_min_x = abs(new_bbox['x_min'] - (min_x - pad)) < tol
        check_max_x = abs(new_bbox['x_max'] - (max_x + pad)) < tol
        check_min_y = abs(new_bbox['y_min'] - (min_y - pad)) < tol
        check_max_y = abs(new_bbox['y_max'] - (max_y + pad)) < tol # might be clipped if near edge? 300-500 is safe.
        
        if check_min_x and check_max_x and check_min_y and check_max_y:
            print("SUCCESS: Padding (10px) correctly applied!")
        else:
            print("FAILURE: Padding check failed.")
            print(f"Diff X_Min: {new_bbox['x_min'] - (min_x - pad)}")
            print(f"Diff X_Max: {new_bbox['x_max'] - (max_x + pad)}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_preview_padding()
