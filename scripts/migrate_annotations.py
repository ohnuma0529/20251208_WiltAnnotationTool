
import os
import sys
import json
import argparse

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.config import settings
from backend.core.image_loader import image_loader
from backend.core.persistence import persistence

def migrate(unit, date, frequency):
    print(f"Migrating {unit}/{date} assuming original frequency {frequency}...")
    
    # 1. Load Images at specified frequency to establish "Index -> Filename" map
    image_loader.load_images(unit, date, frequency)
    images = image_loader.images
    print(f"Loaded {len(images)} images (Freq {frequency}).")
    
    if not images:
        print("No images found. Aborting.")
        return

    # 2. Load RAW annotation state (don't use persistence.load_state as it parses Objects)
    # We want to manipulate keys directly.
    state_file = persistence.get_state_file(unit, date)
    if not os.path.exists(state_file):
        print(f"No annotation file found at {state_file}")
        return
        
    with open(state_file, 'r') as f:
        data = json.load(f)
        
    new_data = {}
    migrated_count = 0
    skipped_count = 0
    
    for key, value in data.items():
        if key.isdigit():
            # It's an index
            idx = int(key)
            if 0 <= idx < len(images):
                path = images[idx]
                fname = os.path.basename(path)
                new_data[fname] = value
                migrated_count += 1
            else:
                print(f"Warning: Index {idx} out of bounds for {len(images)} images. Keeping as is.")
                new_data[key] = value
                skipped_count += 1
        else:
            # Already a filename
            new_data[key] = value
            
    # 3. Save back
    # Backup first
    import shutil
    shutil.copy2(state_file, state_file + ".bak")
    print(f"Backed up to {state_file}.bak")
    
    with open(state_file, 'w') as f:
        json.dump(new_data, f, indent=2)
        
    print(f"Migration Complete. Migrated: {migrated_count}, Skipped: {skipped_count}, Total: {len(new_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--freq", type=int, default=30)
    args = parser.parse_args()
    
    migrate(args.unit, args.date, args.freq)
