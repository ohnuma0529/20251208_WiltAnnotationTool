
import requests
import json
import math

API_URL = "http://localhost:8001/api"

def test_preview_points():
    # 1. Set Filter to ensure images are loaded
    print("Setting filter...")
    res = requests.post(f"{API_URL}/set_filter", json={"unit": "31", "date": "20250430", "frequency": 30})
    if res.status_code != 200:
        print("Failed to set filter")
        return

    # 2. Get Images
    print("Getting images...")
    res = requests.get(f"{API_URL}/images")
    images = res.json()
    if not images:
        print("No images found")
        return
    
    print(f"Found {len(images)} images.")
    frame_index = 0
    
    # 3. Call preview_points with a loose BBox
    # Assume image size is large, create a box that should contain something.
    # We'll use a box in the middle.
    bbox = {
        "x_min": 300,
        "y_min": 300,
        "x_max": 500,
        "y_max": 500
    }
    
    print(f"Requesting preview with BBox: {bbox}")
    try:
        res = requests.post(f"{API_URL}/preview_points", json={
            "frame_index": frame_index,
            "bbox": bbox
        })
        
        if res.status_code != 200:
            print(f"Error: {res.status_code} - {res.text}")
            return
            
        data = res.json()
        points = data['points']
        new_bbox = data['new_bbox']
        
        print(f"Received {len(points)} support points.")
        print(f"New BBox: {new_bbox}")
        
        if not points:
            print("No points found (maybe empty area or SAM skipped).")
            # If no points, new_bbox should match input bbox
            return

        # 4. Verify Tightness
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        print(f"Calculated Bounds: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
        
        # Check if new_bbox matches calculated bounds (Float comparison)
        tol = 0.01
        is_tight = (
            abs(new_bbox['x_min'] - min_x) < tol and
            abs(new_bbox['x_max'] - max_x) < tol and
            abs(new_bbox['y_min'] - min_y) < tol and
            abs(new_bbox['y_max'] - max_y) < tol
        )
        
        if is_tight:
            print("SUCCESS: BBox is tight (Bitabita)!")
        else:
            print("FAILURE: BBox is NOT tight.")
            print(f"Diff: {new_bbox['x_min'] - min_x}, ...")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_preview_points()
