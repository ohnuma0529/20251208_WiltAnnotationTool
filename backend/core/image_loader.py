import os
import glob
from typing import List
from ..config import settings

class ImageLoader:
    def __init__(self):
        self.image_dir = settings.IMAGE_DIR
        self.images: List[str] = []
        self.current_unit = None
        self.current_date = None
        self.current_frequency = settings.DEFAULT_FREQUENCY
        
        # Initial interaction: load first unit and date
        self.units = self.get_available_units()
        if self.units:
            self.current_unit = self.units[0]
            self.dates = self.get_available_dates(self.current_unit)
            if self.dates:
                self.load_images(self.current_unit, self.dates[0], self.current_frequency)
            else:
                print(f"No dates found in {self.current_unit}")
        else:
            print("No units found in image directory.")

    def get_available_units(self) -> List[str]:
        if not os.path.exists(self.image_dir):
            return []
        entries = os.listdir(self.image_dir)
        units = [e for e in entries if os.path.isdir(os.path.join(self.image_dir, e)) and e.isdigit()] # Assuming units are numbers like 31, 32...
        return sorted(units)

    def get_available_dates(self, unit: str) -> List[str]:
        if not unit:
            return []
        unit_dir = os.path.join(self.image_dir, unit)
        if not os.path.exists(unit_dir):
            return []
        entries = os.listdir(unit_dir)
        dates = [e for e in entries if os.path.isdir(os.path.join(unit_dir, e))]
        return sorted(dates)

    def load_images(self, unit: str, date_str: str, frequency: int):
        self.current_unit = unit
        self.current_date = date_str
        self.current_frequency = frequency # Stored for reference, but not used for filtering
        
        target_dir = os.path.join(self.image_dir, unit, date_str)
        if not os.path.exists(target_dir):
            print(f"Directory {target_dir} does not exist.")
            self.images = []
            return

        all_files = sorted(glob.glob(os.path.join(target_dir, "*.jpg")))
        
        # Prepare Cache Dir
        cache_target_dir = os.path.join(settings.CACHE_DIR, unit, date_str)
        if not os.path.exists(cache_target_dir):
            os.makedirs(cache_target_dir, exist_ok=True)
            
        print(f"Scanning {len(all_files)} files in {target_dir} (Loading ALL for Dense Tracking)...")
        
        filtered_files = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            try:
                # Format: 31_04_HDR_20251101-0730.jpg
                # Parse HHMM
                time_part = fname.replace(".jpg", "").split("-")[-1]
                if len(time_part) == 4 and time_part.isdigit():
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    
                    # 1. Time Range
                    if not (settings.START_TIME_HOUR <= hour < settings.END_TIME_HOUR):
                        continue
                        
                    # Frequency filtering REMOVED to allow Dense Tracking. 
                    # Display sparsity will be handled by API.
                    
                    filtered_files.append(fpath)
            except Exception as e:
                pass
        
        
        # Check Metadata for Persistence Truncation
        from .persistence import persistence
        meta = persistence.load_metadata(unit, date_str)
        if "valid_count" in meta:
            limit = meta["valid_count"]
            print(f"Applying Persistent Truncation: Limit to {limit} frames.")
            filtered_files = filtered_files[:limit]

        # Cache and Set Paths
        self.images = []
        import shutil
        total_files = len(filtered_files)
        print(f"Caching {total_files} images to {cache_target_dir}...")
        
        for i, src_path in enumerate(filtered_files):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(cache_target_dir, fname)
            
            # Copy if not exists - optimize speed
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            self.images.append(dst_path)
            
        print(f"Loaded {len(self.images)} images (Cached) for {unit}/{date_str}.")


    def get_image_path(self, index: int) -> str:
        if 0 <= index < len(self.images):
            return self.images[index]
        return None
    
    def get_total_frames(self) -> int:
        return len(self.images)

image_loader = ImageLoader()
