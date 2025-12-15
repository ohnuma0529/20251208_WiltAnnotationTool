
import json
import os

path = "/media/HDD-6TB/Wilt_Project_Work/31/20250524/annotation_state.json"
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
        # Check first frame (assuming start frame)
        # Keys are filenames
        keys = sorted(data.keys())
        if keys:
            first = keys[0]
            leaves = data[first].get("leaves", [])
            print(f"Frame {first} has {len(leaves)} leaves.")
            for l in leaves:
                print(f" - Leaf ID: {l.get('id')}, Manual: {l.get('manual', False)}")
else:
    print("File not found.")
