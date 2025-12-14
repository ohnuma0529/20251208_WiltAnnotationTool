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
            print(f"Error: Image directory not found at {self.image_dir}")
            return []
        entries = os.listdir(self.image_dir)
        # Relaxed filter: Allow any directory, not just digits
        units = [e for e in entries if os.path.isdir(os.path.join(self.image_dir, e))] 
        print(f"Found units in {self.image_dir}: {units}")
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
        self.current_frequency = frequency
        
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
            
        print(f"Scanning {len(all_files)} files in {target_dir} with freq={frequency}...")
        
        filtered_files = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            try:
                # Format: 31_04_HDR_20251101-0730.jpg
                time_part = fname.replace(".jpg", "").split("-")[-1]
                if len(time_part) == 4 and time_part.isdigit():
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    
                    if not (settings.START_TIME_HOUR <= hour < settings.END_TIME_HOUR): continue
                    if minute % frequency != 0: continue
                        
                    filtered_files.append(fpath)
            except Exception as e:
                pass
        
        # Use simple count check for optimization
        # If cache dir has same number of files as filtered_files, assume sync is done.
        existing_cache_files = os.listdir(cache_target_dir)
        # Filter for jpg only in cache to be safe
        existing_jpgs = [f for f in existing_cache_files if f.endswith('.jpg')]
        
        should_resync = len(existing_jpgs) != len(filtered_files)
        
        self.images = []

        if should_resync:
            print(f"Resyncing cache... (Source: {len(filtered_files)}, Cache: {len(existing_jpgs)})")
            import shutil
            # Clear invalid cache if count mismatch? better to just overwrite/add.
            # actually safe to just ensure all filtered files exist.
            for i, src_path in enumerate(filtered_files):
                fname = os.path.basename(src_path)
                dst_path = os.path.join(cache_target_dir, fname)
                
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                
                self.images.append(dst_path)
        else:
            print(f"Cache hit! Loaded {len(existing_jpgs)} images from SSD.")
            # Reconstruct paths from filtered_files basenames to ensure order
            for src_path in filtered_files:
                 fname = os.path.basename(src_path)
                 self.images.append(os.path.join(cache_target_dir, fname))
                 
        print(f"Loaded {len(self.images)} images for {unit}/{date_str} (Freq: {frequency}m).")

    def get_all_files(self, unit: str, date_str: str) -> List[str]:
        """
        Get ALL image files (Freq 1) for a unit/date without changing internal state.
        Effective for Dense Tracking.
        """
        target_dir = os.path.join(self.image_dir, unit, date_str)
        if not os.path.exists(target_dir):
            return []

        all_files = sorted(glob.glob(os.path.join(target_dir, "*.jpg")))
        filtered_files = []
        
        # Use cache dir for consistency? 
        # Ideally we want absolute paths that match what load_images returns (which might be cache paths).
        # But load_images resyncs cache on demand.
        # If we return raw paths from HDD, CoTracker works fine.
        # But persistence uses basename mapping, so path doesn't matter as long as basename is correct.
        
        for fpath in all_files:
            fname = os.path.basename(fpath)
            try:
                # Format: 31_04_HDR_20251101-0730.jpg
                time_part = fname.replace(".jpg", "").split("-")[-1]
                if len(time_part) == 4 and time_part.isdigit():
                    hour = int(time_part[:2])
                    
                    if not (settings.START_TIME_HOUR <= hour < settings.END_TIME_HOUR): continue
                    # Freq 1 check (always pass)
                    filtered_files.append(fpath)
            except Exception:
                pass
                
        return filtered_files


    def get_image_path(self, index: int) -> str:
        if 0 <= index < len(self.images):
            return self.images[index]
        return None
    
    def get_total_frames(self) -> int:
        return len(self.images)

    def delete_images_from_index(self, start_index: int) -> int:
        if start_index < 0 or start_index >= len(self.images):
            return 0
        
        to_delete = self.images[start_index:]
        deleted_count = 0
        
        # Source directory for current session
        if not self.current_unit or not self.current_date:
             return 0
             
        source_dir = os.path.join(self.image_dir, self.current_unit, self.current_date)
        
        for cache_path in to_delete:
            fname = os.path.basename(cache_path)
            
            # 1. Delete from Cache
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except Exception as e:
                print(f"Error deleting cache file {cache_path}: {e}")
                
            # 2. Delete from Source (to prevent resurrection)
            source_path = os.path.join(source_dir, fname)
            try:
                if os.path.exists(source_path):
                    os.remove(source_path)
            except Exception as e:
                print(f"Error deleting source file {source_path}: {e}")
                
            deleted_count += 1
            
        # Update internal list
        self.images = self.images[:start_index]
        return deleted_count

image_loader = ImageLoader()
