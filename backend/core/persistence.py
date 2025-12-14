import os
import json
from typing import Dict
from ..config import settings
from ..models.schemas import TrackingResult

from ..core.image_loader import image_loader

class PersistenceManager:
    def __init__(self):
        self.work_dir = settings.WORK_DIR
        self.state_file = os.path.join(self.work_dir, "annotation_state.json")
        self._ensure_dir()

    def _ensure_dir(self):
        if not os.path.exists(self.work_dir):
            try:
                os.makedirs(self.work_dir, exist_ok=True)
            except OSError as e:
                print(f"Warning: Could not create WORK_DIR at {self.work_dir}: {e}")

    def get_state_file(self, unit, date):
        if not unit or not date:
            return None
        dir_path = os.path.join(self.work_dir, str(unit), str(date))
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "annotation_state.json")

    def save_state(self, unit: str, date: str, results: Dict[int, TrackingResult]):
        state_file = self.get_state_file(unit, date)
        if not state_file:
            print("Cannot save state: Unit or Date not set.")
            return

        try:
            # Convert TrackingResult objects to dicts
            # V48: Use Filenames as keys to prevent drift when freq changes
            data = {}
            # print(f"DEBUG: Saving state with {len(results)} frames. Current Freq: {image_loader.current_frequency}. Total Images: {len(image_loader.images)}")
            
            for idx, res in results.items():
                if not res.leaves: continue 
                
                # Get filename for this index
                path = image_loader.get_image_path(idx)
                if path:
                    fname = os.path.basename(path)
                    data[fname] = res.model_dump()
                    # print(f"DEBUG: Saved index {idx} as {fname}") # Verbose
                else:
                    # Fallback: keep index if path lookup fails (rare)
                    data[str(idx)] = res.model_dump()
                    print(f"WARNING: Could not resolve path for index {idx}. Saving as int.")

            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Auto-saved state to {state_file}. Keys: {list(data.keys())[:5]}...")
        except Exception as e:
            print(f"Failed to auto-save state: {e}")

    def load_state(self, unit: str, date: str) -> Dict[int, TrackingResult]:
        state_file = self.get_state_file(unit, date)
        if not state_file or not os.path.exists(state_file):
            return {}
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct objects
            results = {}
            # Build reverse lookup map: Filename -> Index
            # Only needed if keys are filenames.
            
            # Current image list
            # We assume image_loader has already loaded images for this unit/date!
            # endpoints.py calls persistence.load_state AFTER image_loader.load_images?
            # Yes, check endpoints.set_filter: load_images called first.
            
            # Build map
            fname_to_idx = {}
            for i in range(image_loader.get_total_frames()):
                path = image_loader.get_image_path(i)
                if path:
                    fname_to_idx[os.path.basename(path)] = i
            
            for k, v in data.items():
                # k could be "image.jpg" or "123" (legacy)
                
                target_idx = -1
                
                if k in fname_to_idx:
                    target_idx = fname_to_idx[k]
                elif k.isdigit():
                    # Legacy: direct index
                    # Note: If freq changed, this MIGHT be wrong, but it's legacy behavior.
                    target_idx = int(k)
                    # If target_idx is out of bounds for current freq, it might be dropped or mapped dangerously.
                    # e.g. freq=30 (few frames), legacy idx=1000 (from freq=1). 
                    # If we accept it, it just won't be visible/accessible if frames list is short.
                    # Or we should try to map it? We can't mapping generic int.
                    pass
                else:
                    # Filename not found in current image list?
                    # This happens if freq=30 and saved file is from freq=1 (intermediate frame).
                    # We should SKIP it strictly speaking, as it's not part of current view.
                    # Or we assume it's lost for this session.
                    continue
                
                if target_idx != -1:
                    results[target_idx] = TrackingResult(**v)
                    
            print(f"Loaded {len(results)} annotations from {state_file}")
            return results
        except Exception as e:
            print(f"Failed to load state: {e}")
            return {}

    def load_raw_state(self, unit: str, date: str) -> Dict[str, dict]:
        """
        Load the raw state (Filename -> Dict) without index mapping.
        Used for Dense Export.
        """
        state_file = self.get_state_file(unit, date)
        if not state_file or not os.path.exists(state_file):
            return {}
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load raw state: {e}")
            return {}

persistence = PersistenceManager()
