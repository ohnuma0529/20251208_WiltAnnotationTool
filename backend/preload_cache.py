
import sys
import os

# Ensure backend can be imported
sys.path.append(os.getcwd())

from backend.core.image_loader import image_loader

def main():
    print("Starting Global Pre-caching...")
    
    # Reload units/dates to be sure
    units = image_loader.get_available_units()
    print(f"Found Units: {units}")
    
    for u in units:
        dates = image_loader.get_available_dates(u)
        print(f"Unit {u} has Dates: {dates}")
        
        for d in dates:
            print(f"----- Pre-loading Cache for {u} / {d} -----")
            try:
                # Frequency 1 ensures all valid images are processed
                image_loader.load_images(u, d, 1)
            except Exception as e:
                print(f"Error loading {u}/{d}: {e}")
                
    print("Global Pre-caching Completed.")

if __name__ == "__main__":
    main()
