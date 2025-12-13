from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import Response
from typing import List, Dict
from pydantic import BaseModel
import os
import json
import zipfile
import io
from ..core.image_loader import image_loader
from ..core.tracking import tracking_engine
from ..core.persistence import persistence
from ..models.schemas import InitTrackingRequest, UpdateRegionRequest, UpdatePointRequest, TrackingResult, FrameData, SetFilterRequest, PreviewPointsRequest, DeleteLeafRequest, LeafAnnotation
from ..config import settings

router = APIRouter()

# Load state on startup
tracking_results: Dict[int, TrackingResult] = persistence.load_state(image_loader.current_unit, image_loader.current_date)

@router.post("/delete_leaf")
def delete_leaf(req: DeleteLeafRequest):
    """
    Delete a specific leaf (or all) from current frame onwards.
    """
    try:
        updated_count = 0
        # V36: Delete from ALL frames (Global Delete)
        frames_to_update = list(tracking_results.keys())
        
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

@router.get("/images", response_model=List[FrameData])
def get_images(frequency: int = 1):
    images = []
    total = image_loader.get_total_frames()
    
    # Identify frames with manual annotations to force include
    manual_indices = {
        idx for idx, res in tracking_results.items()
        if any(l.manual for l in res.leaves)
    }

    for i in range(total):
        path = image_loader.get_image_path(i)
        if not path: continue
        
        # Filter by Display Frequency (Skip filter if frame is manually annotated)
        if frequency > 1 and i not in manual_indices:
            basename = os.path.basename(path)
            try:
                # Format ...-HHMM.jpg
                ts = basename.replace(".jpg", "").split("-")[-1]
                if len(ts) == 4 and ts.isdigit():
                    minute = int(ts[2:])
                    if minute % frequency != 0:
                        continue
            except:
                pass
        
        basename = os.path.basename(path)
        relative_path = os.path.relpath(path, settings.CACHE_DIR)
        
        # Parse timestamp from filename again or cache it
        ts = basename.replace(".jpg", "").split("-")[-1] 
        # Format TS as HH:MM
        ts_pretty = f"{ts[:2]}:{ts[2:]}"
        images.append(FrameData(filename=relative_path, frame_index=i, timestamp=ts_pretty))
    return images

class TruncateRequest(BaseModel):
    frame_index: int

@router.post("/truncate_frames")
def truncate_frames(req: TruncateRequest):
    """
    Delete all frames from frame_index onwards (Files + Cache + State).
    Removes them from the current session and tracking state.
    """
    try:
        total = image_loader.get_total_frames()
        if req.frame_index >= total:
            return {"status": "no_change", "msg": "Index out of range"}
            
        print(f"Truncating from index {req.frame_index} (Total: {total})...")
        deleted_count = 0
        
        # 1. Update Tracking Results (Delete future keys)
        keys_to_del = [k for k in tracking_results.keys() if k >= req.frame_index]
        for k in keys_to_del:
            del tracking_results[k]
            
        # 2. Delete Cache Files and Update Loader
        # Helper to delete physical file
        frames_to_remove = [] 
        for i in range(req.frame_index, total):
            p = image_loader.get_image_path(i)
            if p:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass
            frames_to_remove.append(i)
            deleted_count += 1
            
        # Truncate the list in memory
        image_loader.images = image_loader.images[:req.frame_index]
        
        # 3. Save State (and Metadata)
        persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
        
        # Save valid_count to metadata for persistence
        persistence.save_metadata(image_loader.current_unit, image_loader.current_date, {"valid_count": req.frame_index})
        
        return {"status": "success", "deleted_frames": deleted_count}
    except Exception as e:
        print(f"Truncate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/init_tracking")
def run_tracking_background(req: InitTrackingRequest):
    try:
        # 1. Identify Keyframes (Manual Annotations)
        # Scan persistent state for any frame with manual leaves
        keyframes: Dict[int, List[LeafAnnotation]] = {}
        for idx, res in tracking_results.items():
            if any(l.manual for l in res.leaves):
                keyframes[idx] = res.leaves
        
        # Add the current request trigger frame as keyframe (conceptually, if not already saved)
        # But usually user calls save_frame before track. 
        # Just in case, we trust tracking_results more? 
        # Or if req.leaves are passed, we treat them as current manual Prompt?
        # Yes, init_tracking prompt is definitely a keyframe source.
        # But let's assume save_frame was called.
        
        print(f"DEBUG: Found {len(keyframes)} Keyframes for Global Tracking.")

        # 2. Run Tracking (Segmented)
        new_results = tracking_engine.execute_tracking(req.frame_index, req.leaves, keyframes)
        
        # 3. Merge new results (Respect Manual Keyframes)
        # We assume new_results contains predicted interpolated frames.
        # We must NOT overwrite manual frames.
        
        for frame_idx, res in new_results.items():
            # Check if this frame is manually annotated in current state
            if frame_idx in tracking_results:
                existing_res = tracking_results[frame_idx]
                
                # Extract Manual Leaves (Keep these!)
                manual_leaves = [l for l in existing_res.leaves if l.manual]
                manual_ids = {l.id for l in manual_leaves}
                
                # Filter New Results (Skip leaves that confuse with manual ones)
                # Correction: If manual leaf exists for ID X, don't overwrite with ID X prediction.
                # If manual leaf doesn't exist for ID Y, ADD ID Y prediction.
                new_leaves = [l for l in res.leaves if l.id not in manual_ids]
                
                # Merge
                merged_leaves = manual_leaves + new_leaves
                tracking_results[frame_idx] = TrackingResult(leaves=merged_leaves)
            else:
                # No existing data, safe to overwrite
                tracking_results[frame_idx] = res
        
        # Auto-save
        persistence.save_state(image_loader.current_unit, image_loader.current_date, tracking_results)
        print("Background tracking task completed and saved.")
        
        # Finalize Status
        tracking_engine.status = "idle"
        
    except Exception as e:
        print(f"Background tracking task failed: {e}")
        tracking_engine.status = "error"
        import traceback
        traceback.print_exc()

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
    """
    path = image_loader.get_image_path(req.frame_index)
    if not path:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    try:
        # We reuse generate_mask_and_support_points but with empty points list (if logic allows)
        # Or we add a specific method.
        # Let's pass empty points list to generate_mask_and_support_points.
        _, support_points, polygon = tracking_engine.generate_mask_and_support_points(path, req.bbox, [])
        return {"points": support_points, "polygon": polygon}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
def export_dataset():
    """
    Export all annotations as a ZIP of YOLO format text files.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for frame_idx, res in tracking_results.items():
            path = image_loader.get_image_path(frame_idx)
            if not path: continue
            
            # Use original filename for txt: image.jpg -> image.txt
            fname = os.path.basename(path)
            txt_name = os.path.splitext(fname)[0] + ".txt"
            
            content = f"# Frame {frame_idx}\n"
            for leaf in res.leaves:
                content += f"Leaf ID: {leaf.id}\n"
                content += f"  BBox: {leaf.bbox}\n"
                points_str = ", ".join([f"({p.id}: {p.x:.1f},{p.y:.1f})" for p in leaf.points])
                content += f"  Points: [{points_str}]\n"
                content += f"  SupportPoints: {len(leaf.support_points)}\n"
                content += "-" * 20 + "\n"
            
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
    Export annotations as CSV for Leaf ID consistency.
    Cols: frame_index, filename, leaf_id, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, keypoints...
    """
    output = io.StringIO()
    # Header
    # Assuming 2 Keypoints (Base, Tip). If more, flexible?
    # User asked for "base_x, base_y, tip_x, tip_y".
    output.write("frame_index,filename,leaf_id,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,base_x,base_y,tip_x,tip_y\n")
    
    # Sort frames
    sorted_frames = sorted(tracking_results.keys())
    
    for f_idx in sorted_frames:
        res = tracking_results[f_idx]
        path = image_loader.get_image_path(f_idx)
        fname = os.path.basename(path) if path else f"frame_{f_idx}.jpg"
        
        if not res.leaves: continue
        
        for leaf in res.leaves:
            # BBox
            bx_min, by_min, bx_max, by_max = "", "", "", ""
            if leaf.bbox:
                bx_min, by_min, bx_max, by_max = leaf.bbox.x_min, leaf.bbox.y_min, leaf.bbox.x_max, leaf.bbox.y_max
            
            # Points (Base=0, Tip=1 assumption)
            base_x, base_y, tip_x, tip_y = "", "", "", ""
            if leaf.points:
                # Find by ID to be sure
                p0 = next((p for p in leaf.points if p.id == 0), None)
                p1 = next((p for p in leaf.points if p.id == 1), None)
                if p0: base_x, base_y = p0.x, p0.y
                if p1: tip_x, tip_y = p1.x, p1.y
                
            line = f"{f_idx},{fname},{leaf.id},{bx_min},{by_min},{bx_max},{by_max},{base_x},{base_y},{tip_x},{tip_y}\n"
            output.write(line)

    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotations.csv"}
    )

@router.get("/annotations")
def get_annotations():
    count = len(tracking_results)
    frames_with_leaves = sum(1 for res in tracking_results.values() if res.leaves)
    print(f"DEBUG: get_annotations returning {count} frames, {frames_with_leaves} with leaves.")
    return tracking_results



class SaveFrameRequest(BaseModel):
    frame_index: int
    leaves: List[LeafAnnotation]

@router.post("/save_frame")
def save_frame(req: SaveFrameRequest):
    """
    Save annotations for a single frame (manual annotation).
    """
    # Create or update result for this frame
    if req.frame_index not in tracking_results:
        tracking_results[req.frame_index] = TrackingResult(leaves=[])
    
    # Force Manual Flag
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
