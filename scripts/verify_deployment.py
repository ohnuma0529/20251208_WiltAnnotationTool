import sys
import os
import traceback

# Add project root to sys.path
sys.path.append(os.getcwd())

try:
    print("Verifying backend imports...")
    from backend.core.image_loader import image_loader
    from backend.core.tracking import tracking_engine
    from backend.api.endpoints import router
    from backend.main import app
    print("Backend imports successful.")
    
    # Optional: Logic Check
    # Verify image_loader can find images (dry run)
    # total = image_loader.get_total_frames()
    # print(f"Image Loader found {total} frames.")
    
    sys.exit(0)
except Exception as e:
    print("VERIFICATION FAILED")
    traceback.print_exc()
    sys.exit(1)
