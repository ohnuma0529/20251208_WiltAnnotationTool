import os
import json
from typing import Dict
from ..config import settings
from ..models.schemas import TrackingResult

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
                # Fallback to local if HDD not mounted? 
                # User requested HDD specifically. We'll assume it works or fail.

    def get_state_file(self, unit, date):
        if not unit or not date:
            return None
        dir_path = os.path.join(self.work_dir, str(unit), str(date))
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "annotation_state.json")

    def get_meta_file(self, unit, date):
        if not unit or not date:
            return None
        dir_path = os.path.join(self.work_dir, str(unit), str(date))
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "session_meta.json")

    def save_state(self, unit: str, date: str, results: Dict[int, TrackingResult]):
        state_file = self.get_state_file(unit, date)
        if not state_file:
            print("Cannot save state: Unit or Date not set.")
            return

        try:
            # Convert TrackingResult objects to dicts
            data = {str(k): v.model_dump() for k, v in results.items()}
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Auto-saved state to {state_file}")
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
            for k, v in data.items():
                results[int(k)] = TrackingResult(**v)
            print(f"Loaded {len(results)} annotations from {state_file}")
            return results
        except Exception as e:
            print(f"Failed to load state: {e}")
            return {}

    def save_metadata(self, unit: str, date: str, metadata: dict):
        meta_file = self.get_meta_file(unit, date)
        if not meta_file: return
        try:
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {meta_file}")
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    def load_metadata(self, unit: str, date: str) -> dict:
        meta_file = self.get_meta_file(unit, date)
        if not meta_file or not os.path.exists(meta_file):
            return {}
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            return {}

persistence = PersistenceManager()
