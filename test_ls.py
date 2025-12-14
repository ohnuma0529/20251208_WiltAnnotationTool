import os
import time

path = "/media/HDD-6TB/Leaf_Images"
print(f"Listing {path}...")
start = time.time()
if os.path.exists(path):
    try:
        entries = os.listdir(path)
        print(f"Found {len(entries)} entries.")
        print(f"First 5: {entries[:5]}")
        
        # Test the loop
        units = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        print(f"Units found: {units}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Path does not exist")
print(f"Done in {time.time() - start:.4f}s")
