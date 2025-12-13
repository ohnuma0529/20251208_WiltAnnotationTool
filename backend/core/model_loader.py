import torch
import os
import sys
from threading import Lock

class ModelLoader:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoader, cls).__new__(cls)
                cls._instance.sam2_predictor = None # Video
                cls._instance.sam2_image_predictor = None # Image (for single frame prompts)
                cls._instance.cotracker_predictor = None
                
                # Device Detection
                device_count = torch.cuda.device_count()
                if device_count >= 2:
                    print("Dual GPU Mode Detected: Splitting models.")
                    cls._instance.sam2_device = 'cuda:0'
                    cls._instance.cotracker_device = 'cuda:1'
                    cls._instance.device = 'cuda:0' # Default fallback
                else:
                    cls._instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    cls._instance.sam2_device = cls._instance.device
                    cls._instance.cotracker_device = cls._instance.device

        return cls._instance

    def load_models(self):
        if self.sam2_predictor is not None and self.cotracker_predictor is not None:
            return

        print(f"Loading models... (SAM2: {self.sam2_device}, CoTracker: {self.cotracker_device})")
        
        # Load Sam 2
        try:
            from sam2.build_sam import build_sam2_video_predictor, build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            
            if os.path.exists(sam2_checkpoint):
                 self.sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.sam2_device)
                 
                 # Image Predictor needs a built model
                 base_model = build_sam2(model_cfg, sam2_checkpoint, device=self.sam2_device)
                 self.sam2_image_predictor = SAM2ImagePredictor(base_model)
                 
                 print("SAM 2 (Video & Image) loaded.")
            else:
                print(f"SAM 2 checkpoint not found at {sam2_checkpoint}. Please download it.")

        except ImportError as e:
            print(f"Could not import SAM 2: {e}")

        # Load CoTracker3
        try:
            print("Loading CoTracker3 Offline from torch.hub...")
            # Use torch.hub to load the official offline model (best quality)
            # This handles downloading correct checkpoints automatically
            self.cotracker_predictor = torch.hub.load(
                "facebookresearch/co-tracker", 
                "cotracker3_offline",
                trust_repo=True
            ).to(self.cotracker_device).eval()
            print("CoTracker3 Offline loaded via torch.hub.")
            
        except Exception as e:
            print(f"CoTracker Hub Loading Error: {e}")
            print("CoTracker3 failed to load.")
                 
        except ImportError as e:
            print(f"Could not import CoTracker: {e}")

model_loader = ModelLoader()
