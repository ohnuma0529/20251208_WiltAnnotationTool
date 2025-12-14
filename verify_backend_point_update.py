
import requests
import json
import time

API_URL = "http://localhost:8001/api"

def test_point_update():
    # 1. Set Filter
    print("Setting filter...")
    requests.post(f"{API_URL}/set_filter", json={"unit": "31", "date": "20250430", "frequency": 30})

    # 2. Init Dummy Tracking or just inject a leaf?
    # We can inject a manual leaf using save_frame
    print("Injecting leaf...")
    leaf_id = 999
    points = [{"x": 300, "y": 300, "id": 0}, {"x": 400, "y": 400, "id": 1}]
    support_points = [{"x": 350, "y": 350, "id": -1}]
    
    # Initial Tight BBox (Approx 300-400)
    # We purposefully set a WRONG bbox to see if update fixes it?
    # No, setting correct initial bbox.
    initial_bbox = {"x_min": 290, "y_min": 290, "x_max": 410, "y_max": 410} # with padding
    
    leaf = {
        "id": leaf_id,
        "bbox": initial_bbox,
        "points": points,
        "support_points": support_points,
        "manual": True,
        "color": "#FF0000"
    }
    
    requests.post(f"{API_URL}/save_frame", json={
        "frame_index": 0,
        "leaves": [leaf]
    })
    
    # 3. Update Point (Move Point 1 to 500, 500)
    print("Updating Point ID 1 to (500, 500)...")
    res = requests.post(f"{API_URL}/update_point", json={
        "frame_index": 0,
        "leaf_id": leaf_id,
        "point_id": 1,
        "x": 500,
        "y": 500
    })
    
    if res.status_code != 200:
        print(f"Update failed: {res.text}")
        return
        
    data = res.json()
    new_bbox = data.get('bbox')
    print(f"Received Updated BBox: {new_bbox}")
    
    if not new_bbox:
        print("FAILURE: No BBox returned.")
        return

    # Expected: x_max should be around 500 + 10 = 510
    pad = 10
    tol = 1.0
    expected_max = 500 + pad
    
    if abs(new_bbox['x_max'] - expected_max) < tol:
        print("SUCCESS: BBox expanded correctly!")
    else:
        print(f"FAILURE: Expected Max {expected_max}, got {new_bbox['x_max']}")

if __name__ == "__main__":
    test_point_update()
