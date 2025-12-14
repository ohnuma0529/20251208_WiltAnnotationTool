import requests
import time
import json

API_URL = "http://localhost:8000/api"

def run_test():
    print("1. Fetching Images...")
    res = requests.get(f"{API_URL}/images")
    if res.status_code != 200:
        print("Failed to fetch images")
        return
    images = res.json()
    if not images:
        print("No images found.")
        return
    
    print(f"Found {len(images)} images.")
    start_frame = 0
    
    print("2. Creating Dummy Manual Leaf at Frame 0...")
    dummy_leaf = {
        "id": 999,
        "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
        "points": [{"x": 100, "y": 150, "id": 1}, {"x": 200, "y": 150, "id": 2}],
        "support_points": [],
        "manual": True
    }
    
    res = requests.post(f"{API_URL}/save_frame", json={
        "frame_index": start_frame,
        "leaves": [dummy_leaf]
    })
    print(f"Save Frame Status: {res.status_code}, {res.json()}")

    print("3. Initializing Tracking...")
    # Send same leaf config as if user clicked "Run Tracking"
    res = requests.post(f"{API_URL}/init_tracking", json={
        "frame_index": start_frame,
        "leaves": [dummy_leaf]
    })
    print(f"Init Tracking Status: {res.status_code}, {res.json()}")

    print("4. Polling Status...")
    while True:
        res = requests.get(f"{API_URL}/tracking_status")
        status = res.json()
        print(f"Status: {status['status']}, Progress: {status['progress']}%")
        
        if status['status'] in ['idle', 'error']:
            break
        time.sleep(1)

    print("5. Verifying Results...")
    res = requests.get(f"{API_URL}/annotations")
    annotations = res.json()
    
    results_count = len(annotations)
    print(f"Total Annotated Frames: {results_count}")
    
    # Check if frame 1 exists (Forward tracking)
    if "1" in annotations:
        print("Success: Frame 1 has annotations.")
        print(json.dumps(annotations["1"], indent=2))
    else:
        print("FAILURE: Frame 1 has NO annotations.")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Test Failed: {e}")
