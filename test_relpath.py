
import os
import shutil

CACHE_DIR = "/home/happyai2023/20251208_WiltAnnotationTool/fast_cache"
dst_path = os.path.join(CACHE_DIR, "31_20250430_0700.jpg")
print(f"CACHE_DIR: {CACHE_DIR}")
print(f"dst_path: {dst_path}")
try:
    rel = os.path.relpath(dst_path, CACHE_DIR)
    print(f"Relpath: {rel}")
except Exception as e:
    print(f"Error: {e}")

# Check actual directory existence
print(f"Exists CACHE_DIR: {os.path.exists(CACHE_DIR)}")
