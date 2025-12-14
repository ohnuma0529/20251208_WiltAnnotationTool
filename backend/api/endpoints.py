from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import Response
from typing import List, Dict
from pydantic import BaseModel
import os
import json
import zipfile
import io
import csv
from PIL import Image
from ..core.image_loader import image_loader
from ..core.tracking import tracking_engine
from ..core.persistence import persistence
from ..models.schemas import InitTrackingRequest, UpdateRegionRequest, UpdatePointRequest, TrackingResult, FrameData, SetFilterRequest, PreviewPointsRequest, DeleteLeafRequest, LeafAnnotation, BBox

class DeleteFramesRequest(BaseModel):
    frame_index: int
from ..config import settings

router = APIRouter()

# Load state on startup
tracking_results: Dict[int, TrackingResult] = persistence.load_state(image_loader.current_unit, image_loader.current_date)

@router.delete("/delete_leaf")
def delete_leaf(req: DeleteLeafRequest):
    """
    Delete a specific leaf (or all) from current frame onwards.
    """
    try:
        # Ensure latest state is loaded if empty (e.g. backend restarted)
        # Load latest state to ensure sync
        tracking_results = persistence.load_state(image_loader.current_unit, image_loader.current_date)
        
        # Determine range
        total_frames = image_loader.get_total_frames()
        
        if req.delete_global and req.delete_all:
             # Explicit robust handling for Global Delete All
             tracking_results.clear()
             persistence.save_state(image_loader.current_unit, image_loader.current_date, {})
             
             # Reload just in case (though it should be empty)
             # tracking_results = persistence.load_state(...) # No need, keep it empty/synced.
             
             return {"status": "success", "message": "All annotations deleted globally"}

        updated_count = 0
        if req.delete_global:
            # Delete from ALL frames (0 to End)
            # This is "Delete from DB" effectively for this leaf.
            frames_to_update = list(tracking_results.keys())
        else:
            # Forward: From current frame onwards
            frames_to_update = [idx for idx in tracking_results.keys() if idx >= req.frame_index]
        
        for idx in frames_to_update:
            res = tracking_results[idx]
            original_len = len(res.leaves)
            
            if req.delete_all:
                res.leaves = []
            elif req.leaf_id is not None:
                res.leaves = [l for l in res.leaves if l.id != req.leaf_id]
            
            if len(res.leaves) != original_len:
                updated_count += 1
                
        if updated_count > 0:
            persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
            
        return {"status": "success", "updated_frames": updated_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/delete_frames")
def delete_frames(req: DeleteFramesRequest):
    """
    Delete image files from frame_index onwards.
    Destructive: deletes from source to prevent resurrection.
    """
    try:
        # Delete images
        count = image_loader.delete_images_from_index(req.frame_index)
        
        # Clean up tracking results for deleted frames
        keys_to_remove = [k for k in tracking_results.keys() if k >= req.frame_index]
        for k in keys_to_remove:
            del tracking_results[k]
            
        # Save state
        persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
        
        return {"status": "deleted", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images", response_model=List[FrameData])
def get_images():
    images = []
    total = image_loader.get_total_frames()
    for i in range(total):
        path = image_loader.get_image_path(i)
        if not path: continue
        
        # Use relative path or just filename as ID?
        # Frontend needs a way to request image.
        # Let's send relative path from image_dir
        relative_path = os.path.relpath(path, image_loader.image_dir)
        ts_pretty = f"{i//60:02d}:{i%60:02d}" # Dummy timestamp logic or parse filename
        
        # Attempt to parse timestamp from filename properly?
        # Filename: Unit_Cam_Type_YYYYMMDD-HHMM.jpg
        try:
             # Basic parse: Last chunk after '-'
             base = os.path.splitext(os.path.basename(path))[0]
             parts = base.split('-')
             if len(parts) >= 2:
                 time_part = parts[-1]
                 if len(time_part) == 4:
                     ts_pretty = f"{time_part[:2]}:{time_part[2:]}"
        except:
             pass
             
        images.append(FrameData(filename=relative_path, frame_index=i, timestamp=ts_pretty))
    return images

# Helper function (Background Task)
def run_tracking_background(req: InitTrackingRequest):
    try:
        # Run Tracking (Dense Mode V49)
        # 1. Get Dense Image List (Freq 1)
        current_unit = image_loader.current_unit
        current_date = image_loader.current_date
        
        backup_state(current_unit, current_date)
        
        full_image_list = image_loader.get_all_files(current_unit, current_date)
        
        # Build Map: Filename -> DenseIndex
        fname_to_dense_idx = {os.path.basename(p): i for i, p in enumerate(full_image_list)}
        
        # Build Map: CurrentViewIndex -> Filename (For current sparse view)
        view_idx_to_fname = {}
        for i in range(image_loader.get_total_frames()):
            p = image_loader.get_image_path(i)
            if p: view_idx_to_fname[i] = os.path.basename(p)

        # 2. Map Start Frame & Keyframes to Dense Indices
        # Start Frame
        start_fname = view_idx_to_fname.get(req.frame_index)
        if not start_fname or start_fname not in fname_to_dense_idx:
            print(f"Error: Start frame {req.frame_index} ({start_fname}) not found in dense list.")
            return

        dense_start_idx = fname_to_dense_idx[start_fname]
        print(f"DEBUG: Dense Tracking - Start Frame Mapped: View {req.frame_index} -> Dense {dense_start_idx}")

        # Keyframes
        dense_keyframes = {}
        for idx, res in tracking_results.items():
            if idx != req.frame_index:
                 manual_leaves = [l for l in res.leaves if l.manual]
                 if manual_leaves:
                     fname = view_idx_to_fname.get(idx)
                     if fname and fname in fname_to_dense_idx:
                         d_idx = fname_to_dense_idx[fname]
                         dense_keyframes[d_idx] = manual_leaves
        
        print(f"DEBUG: Found {len(dense_keyframes)} existing keyframes mapped for dense tracking.")

        # CALLBACK for Incremental Saving
        def on_step_save(partial_results: Dict[int, TrackingResult]):
            # Reuse saving logic
            # This is effectively the same as final save logic but called incrementally.
            # Warning: Overwrites file.
            filename_data = {}
            for d_idx, res in partial_results.items():
                if d_idx < len(full_image_list):
                    path = full_image_list[d_idx]
                    fname = os.path.basename(path)
                    filename_data[fname] = {
                        "leaves": [l.model_dump() for l in res.leaves] 
                    }
            
            state_file = persistence.get_state_file(current_unit, current_date)
            try:
                with open(state_file, 'w') as f:
                    json.dump(filename_data, f, indent=2)
                print(f"Incremental Save: Preserved {len(filename_data)} frames.")
            except Exception as e:
                print(f"Incremental Save Failed: {e}")

        # 3. Run Tracking on Full List
        # Note: returns results keyed by Dense Index
        print(f"DEBUG: Calling execute_tracking with {len(full_image_list)} images (Dense)")
        results_dense = tracking_engine.execute_tracking(
            dense_start_idx, 
            req.leaves, 
            keyframes=dense_keyframes, 
            image_paths=full_image_list,
            step_callback=on_step_save
        )
        
        # 4. Save Results (Using Filenames for Persistence)
        # We need to construct the full Dictionary for persistence.save_state.
        # However, persistence.save_state currently takes Dict[int, Result] and uses image_loader to map back to files.
        # But image_loader is in SPARSE mode. It cannot map Dense Index 500 if that frame isn't loaded.
        
        # WE MUST USE A NEW PERSISTENCE METHOD OR HACK IT.
        # Better: Convert our Dense Results (Dict[int, Result]) into Dict[str, dict] (Filename -> Dump) manually here,
        # and save it directly or via a new persistence method.
        # Let's modify persistence to verify this logic?
        # Actually simplest way:
        # Create a dict keyed by filename.
        filename_data = {}
        
        # Merge existing sparse tracking_results into it first (to keep non-tracked data? actually tracking overwrites)
        # But tracking might be partial? No, dense tracking covers everything.
        
        for d_idx, res in results_dense.items():
            if d_idx < len(full_image_list):
                path = full_image_list[d_idx]
                fname = os.path.basename(path)
                # res is TrackingResult. Extract leaves.
                filename_data[fname] = {
                    "leaves": [l.model_dump() for l in res.leaves] 
                }
        
        # Save directly to file
        state_file = persistence.get_state_file(current_unit, current_date)
        with open(state_file, 'w') as f:
            json.dump(filename_data, f, indent=2)
        print(f"Auto-saved DENSE state to {state_file}")

        # 5. Update In-Memory View (Sparse)
        tracking_results.clear()
        
        for view_idx, fname in view_idx_to_fname.items():
            if fname in fname_to_dense_idx:
                d_idx = fname_to_dense_idx[fname]
                if d_idx in results_dense:
                    # res is already TrackingResult
                    tracking_results[view_idx] = results_dense[d_idx]

        # Finalize Status
        tracking_engine.status = "idle"
        
    except Exception as e:
        print(f"Background tracking task failed: {e}")
        import traceback
        traceback.print_exc()
        tracking_engine.status = "error"

@router.post("/init_tracking")
def init_tracking(req: InitTrackingRequest, background_tasks: BackgroundTasks):
    """
    Initialize tracking with SAM 2 and CoTracker.
    Runs in background.
    """
    if tracking_engine.status == "running":
        raise HTTPException(status_code=400, detail="Tracking already and running")
        
    background_tasks.add_task(run_tracking_background, req)
    
    return {"status": "queued"}

@router.get("/tracking_status")
def get_tracking_status():
    return {
        "status": tracking_engine.status,
        "progress": float(tracking_engine.progress)
    }

@router.post("/preview_points")
def preview_points(req: PreviewPointsRequest):
    """
    Generate support points within BBox using SAM mask, without starting tracking.
    Returns points, polygon, and TIGHTENED BBox.
    """
    path = image_loader.get_image_path(req.frame_index)
    if not path:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    try:
        # Generate points
        _, support_points, polygon = tracking_engine.generate_mask_and_support_points(path, req.bbox, [])
        
        # Calculate Tight BBox with Padding (V5: "Slightly Larger")
        if support_points:
            with Image.open(path) as img:
                img_w, img_h = img.size

            xs = [p['x'] for p in support_points]
            ys = [p['y'] for p in support_points]

            pad = 10
            min_x = max(0, min(xs) - pad)
            min_y = max(0, min(ys) - pad)
            max_x = min(img_w, max(xs) + pad)
            max_y = min(img_h, max(ys) + pad)
            
            # Update BBox
            req.bbox.x_min = float(min_x)
            req.bbox.y_min = float(min_y)
            req.bbox.x_max = float(max_x)
            req.bbox.y_max = float(max_y)

        return {
            "points": support_points,
            "polygon": polygon,
            "new_bbox": req.bbox
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export_yolo")
def export_yolo():
    """
    Export all annotations (DENSE) as a ZIP of YOLO format text files.
    Format: <class> <x_center> <y_center> <width> <height> <px1> <py1> <v1> ...
    Normalized to [0, 1].
    """
    buffer = io.BytesIO()
    
    # 1. Get Dense Image List
    current_unit = image_loader.current_unit
    current_date = image_loader.current_date
    full_image_list = image_loader.get_all_files(current_unit, current_date)
    
    # 2. Load Raw State (Filename -> Result)
    # This ensures we get annotations even if current view is sparse.
    raw_state = persistence.load_raw_state(current_unit, current_date)
    
    # 3. Get Image Dimensions (Cache from first available image)
    img_w, img_h = 0, 0
    # Search for first existing file to get dims
    for path in full_image_list:
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    img_w, img_h = img.size
                break
            except Exception:
                continue
                
    if img_w == 0 or img_h == 0:
        raise HTTPException(status_code=500, detail="Could not determine image dimensions from any file.")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, path in enumerate(full_image_list):
            fname = os.path.basename(path)
            
            # Check if we have annotation
            # State keys might be "image.jpg"
            if fname not in raw_state:
                continue
                
            res_dict = raw_state[fname]
            # Convert dict to object for typesafety or just use dict
            leaves = res_dict.get('leaves', [])
            if not leaves: continue
            
            txt_name = os.path.splitext(fname)[0] + ".txt"
            content = ""
            
            for leaf_data in leaves:
                # LeafAnnotation structure
                bbox = leaf_data.get('bbox')
                if not bbox: continue
                
                # YOLO BBox: x_center, y_center, width, height (Normalized)
                bx = (bbox['x_min'] + bbox['x_max']) / 2.0
                by = (bbox['y_min'] + bbox['y_max']) / 2.0
                bw = bbox['x_max'] - bbox['x_min']
                bh = bbox['y_max'] - bbox['y_min']
                
                nx = bx / img_w
                ny = by / img_h
                nw = bw / img_w
                nh = bh / img_h
                
                # YOLO-Pose Keypoints
                # Find Base (ID 0) and Tip (ID 1)
                points = leaf_data.get('points', [])
                p_base = next((p for p in points if p['id'] == 0), None)
                p_tip = next((p for p in points if p['id'] == 1), None)
                
                def fmt_kp(p):
                    if p:
                        return f"{p['x'] / img_w:.6f} {p['y'] / img_h:.6f} 2"
                    return "0.0 0.0 0"
                
                kps_str = f"{fmt_kp(p_base)} {fmt_kp(p_tip)}"
                
                # Class 0 for Leaf
                line = f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {kps_str}\n"
                content += line
            
            if content:
                zip_file.writestr(txt_name, content)

    buffer.seek(0)
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=annotations_yolo.zip"}
    )

@router.get("/export_csv")
def export_csv():
    """
    Export annotations as CSV (Dense).
    Columns: frame_index, filename, leaf_id, 
             bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
             base_x, base_y, tip_x, tip_y,
             image_width, image_height
    All coordinates are NORMALIZED [0, 1].
    """
    output = io.StringIO()
    writer = csv.writer(output)
    # Header
    writer.writerow([
        "frame_index", "filename", "leaf_id", 
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "base_x", "base_y", "tip_x", "tip_y",
        "image_width", "image_height", "is_manual"
    ])
    
    # 1. Get Dense Image List
    current_unit = image_loader.current_unit
    current_date = image_loader.current_date
    full_image_list = image_loader.get_all_files(current_unit, current_date)
    
    # 2. Load Raw State
    raw_state = persistence.load_raw_state(current_unit, current_date)
    
    # 3. Get Dims (Cache)
    img_w, img_h = 0, 0
    for path in full_image_list:
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    img_w, img_h = img.size
                break
            except Exception:
                continue
    
    if img_w == 0 or img_h == 0:
         # Fallback default?
         pass

    # Iterate Dense List
    for idx, path in enumerate(full_image_list):
        fname = os.path.basename(path)
        if fname not in raw_state: continue
        
        leaves = raw_state[fname].get('leaves', [])
        
        for leaf in leaves:
            bbox = leaf.get('bbox')
            if not bbox: continue
            
            points = leaf.get('points', [])
            p_base = next((p for p in points if p['id'] == 0), None)
            p_tip = next((p for p in points if p['id'] == 1), None)
            
            # Manual Flag
            is_manual = leaf.get('manual', False)
            
            # Raw coords for calc
            raw_xmin = bbox['x_min']
            raw_ymin = bbox['y_min']
            raw_xmax = bbox['x_max']
            raw_ymax = bbox['y_max']
            
            raw_base_x = p_base['x'] if p_base else None
            raw_base_y = p_base['y'] if p_base else None
            raw_tip_x = p_tip['x'] if p_tip else None
            raw_tip_y = p_tip['y'] if p_tip else None
            
            # Normalized
            n_xmin = raw_xmin / img_w if img_w else ""
            n_ymin = raw_ymin / img_h if img_h else ""
            n_xmax = raw_xmax / img_w if img_w else ""
            n_ymax = raw_ymax / img_h if img_h else ""
            
            n_base_x = raw_base_x / img_w if (raw_base_x is not None and img_w) else ""
            n_base_y = raw_base_y / img_h if (raw_base_y is not None and img_h) else ""
            n_tip_x = raw_tip_x / img_w if (raw_tip_x is not None and img_w) else ""
            n_tip_y = raw_tip_y / img_h if (raw_tip_y is not None and img_h) else ""
            
            writer.writerow([
                idx, fname, leaf['id'],
                n_xmin, n_ymin, n_xmax, n_ymax,
                n_base_x, n_base_y, n_tip_x, n_tip_y,
                img_w, img_h, is_manual
            ])
            
    csv_content = output.getvalue()
    
    # Save Locally
    filename = f"{current_unit}_{current_date}.csv"
    export_dir = os.path.join(settings.BASE_DIR, "backend", "exports") # Assume BASE_DIR is root
    # Fallback if BASE_DIR not set in config use explicit path or relative
    # Better to use relative to current working dir or just hardcode for safety in this specific env
    export_path = os.path.join("backend", "exports", filename)
    
    try:
        with open(export_path, "w") as f:
            f.write(csv_content)
        print(f"DEBUG: Saved CSV export to {export_path}")
    except Exception as e:
        print(f"ERROR: Failed to save local CSV: {e}")

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/annotations")
def get_annotations():
    count = len(tracking_results)
    frames_with_leaves = sum(1 for res in tracking_results.values() if res.leaves)
    print(f"DEBUG: get_annotations returning {count} frames, {frames_with_leaves} with leaves.")
    
    count = len(tracking_results)
    frames_with_leaves = sum(1 for res in tracking_results.values() if res.leaves)
    print(f"DEBUG: get_annotations returning {count} frames, {frames_with_leaves} with leaves.")
    return tracking_results

@router.post("/update_region")
def update_region(req: UpdateRegionRequest):
    """
    Update BBox for a leaf and REGENERATE support points (Dynamic Resampling).
    This serves as the "Face Correction" step before backward tracking.
    """
    if req.frame_index not in tracking_results:
        # Implicitly creating frame if it doesn't exist?
        # Usually update assumes existence.
        raise HTTPException(status_code=404, detail="Frame has no annotations yet. Use save_frame or init_tracking.")
    
    res = tracking_results[req.frame_index]
    target_leaf = next((l for l in res.leaves if l.id == req.leaf_id), None)
    
    if not target_leaf:
        raise HTTPException(status_code=404, detail=f"Leaf {req.leaf_id} not found in frame {req.frame_index}")
        
    # Update BBox
    target_leaf.bbox = req.bbox
    target_leaf.manual = True
    
    # Regenerate Support Points
    path = image_loader.get_image_path(req.frame_index)
    if path:
        try:
            # We preserve Main Points? Yes.
            # We ONLY regenerate support points.
            _, new_support, new_poly = tracking_engine.generate_mask_and_support_points(path, req.bbox, [])
            target_leaf.support_points = new_support
            target_leaf.mask_polygon = new_poly
        except Exception as e:
            print(f"Warning: Failed to regenerate support points: {e}")
            
    persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
    return {"status": "updated", "leaf": target_leaf}

@router.post("/update_point")
def update_point(req: UpdatePointRequest):
    """
    Update a single point position (Main or Support).
    """
    if req.frame_index not in tracking_results:
         raise HTTPException(status_code=404, detail="Frame not found")
         
    res = tracking_results[req.frame_index]
    target_leaf = next((l for l in res.leaves if l.id == req.leaf_id), None)
    
    if not target_leaf:
        raise HTTPException(status_code=404, detail="Leaf not found")
        
    found = False
    # Check Main Points
    for p in target_leaf.points:
        if p.id == req.point_id:
            p.x = req.x
            p.y = req.y
            found = True
            break
            
    # Check Support Points if not found
    if not found:
        for p in target_leaf.support_points:
            if p.id == req.point_id: # ID logic for support points? usually -1.
                # If all support points have ID -1, we can't update specific one by ID.
                # But Point struct has ID.
                # tracking.py assigns ID -1.
                # If we need to update specific support point, they need unique IDs.
                # For now, let's assume this endpoint is mostly for Main Points (0, 1).
                # Support points are usually regenerated, not dragged individually?
                # User prompt: "Keypoint設定，Keypoint修正". Likely Main Points.
                pass
    
    
    # Recalculate BBox with Padding (V5 logic)
    # Gather all points
    all_points = target_leaf.points + target_leaf.support_points
    if all_points:
        xs = [p.x for p in all_points]
        ys = [p.y for p in all_points]
        
        # Get Image Dims for Clamping
        # We can try cache or file load. File load is safer.
        path = image_loader.get_image_path(req.frame_index)
        if path:
             with Image.open(path) as img:
                  img_w, img_h = img.size
                  
             pad = 10
             min_x = max(0, min(xs) - pad)
             min_y = max(0, min(ys) - pad)
             max_x = min(img_w, max(xs) + pad)
             max_y = min(img_h, max(ys) + pad)
             
             target_leaf.bbox = BBox(x_min=float(min_x), y_min=float(min_y), x_max=float(max_x), y_max=float(max_y))
             
    target_leaf.manual = True
    persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
    
    return {"status": "updated", "bbox": target_leaf.bbox}

class SaveFrameRequest(BaseModel):
    frame_index: int
    leaves: List[LeafAnnotation]

@router.post("/save_frame")
def save_frame(req: SaveFrameRequest):
    """
    Save annotations for a single frame (manual annotation).
    Forces manual=True for all leaves.
    """
    # Create or update result for this frame
    if req.frame_index not in tracking_results:
        tracking_results[req.frame_index] = TrackingResult(leaves=[])
    
    # Enforce manual=True
    for l in req.leaves:
        l.manual = True
        
    tracking_results[req.frame_index].leaves = req.leaves
    
    persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
    return {"status": "saved", "leaf_count": len(req.leaves)}

@router.get("/units")
def get_units():
    return {"units": image_loader.get_available_units()}

@router.get("/dates")
def get_dates(unit: str = None):
    # If unit not provided, use current.
    target_unit = unit if unit else image_loader.current_unit
    return {"dates": image_loader.get_available_dates(target_unit)}

@router.post("/set_filter")
def set_filter(req: SetFilterRequest):
    image_loader.load_images(req.unit, req.date, req.frequency)
    
    # Reload annotations for the new context
    new_state = persistence.load_state(req.unit, req.date)
    tracking_results.clear()
    tracking_results.update(new_state)
    
    return {
        "message": f"Filter set to {req.unit}/{req.date} with freq {req.frequency}m",
        "total_frames": image_loader.get_total_frames()
    }

