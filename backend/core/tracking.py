import torch
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
from ..models.schemas import Point, BBox, TrackingResult, LeafAnnotation
from .model_loader import model_loader
import math
from ..core.image_loader import image_loader

class TrackingEngine:
    def __init__(self):
        # We access device via model_loader when needed to ensure latest state
        self.progress = 0.0
        self.status = "idle" # idle, running, error
        
    def generate_mask_and_support_points(self, frame_path: str, bbox: BBox, points: List[Point]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Uses SAM 2 to get mask from BBox.
        Samples support points inside mask using GoodFeaturesToTrack.
        """

        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not read {frame_path}")
            
        # Handle EXIF Orientation using PIL to match Browser display
        # Browser usually rotates images based on EXIF. cv2 does not.
        from PIL import Image, ImageOps
        try:
            pil_img = Image.open(frame_path)
            pil_img = ImageOps.exif_transpose(pil_img) # Auto-rotate
            img_rgb = np.array(pil_img) # RGB
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # BGR
        except Exception as e:
            print(f"PIL loading failed, fallback to cv2: {e}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            

        
        predictor = model_loader.sam2_image_predictor
        if not predictor:
            # Fallback if model not loaded (dev mode)
            print("SAM 2 not loaded, using fallback grid.")
            return self._fallback_grid_points(bbox)

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(img_rgb)
                
                print(f"DEBUG: Processing {frame_path}")
                print(f"DEBUG: Image shape mapped for SAM: {img.shape}")
                print(f"DEBUG: Input BBox: {bbox}")
                
                # Prompt with BBox only (Default usage)
                # box expects [x1, y1, x2, y2]
                box = np.array([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
                print(f"DEBUG: SAM Box Prompt: {box}")
                
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=True
                )
                
                # Default selection (Best Score)
                best_idx = np.argmax(scores)
                print(f"DEBUG: SAM Scores: {scores}, Best: {best_idx}")
                
                # mask is (3, H, W) -> Select best
                mask = masks[best_idx].astype(np.uint8) * 255

                # IMPROVEMENT: Erode mask to avoid edges
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=2)
                
                # Extract Contour (Polygon) for Visualization
                polygon_list = []
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Take largest contour for visualization
                    largest_cnt = max(contours, key=cv2.contourArea)
                    # Simplify
                    epsilon = 0.005 * cv2.arcLength(largest_cnt, True)
                    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
                    
                    polygon_list = [{"x": float(p[0][0]), "y": float(p[0][1]), "id": -2} for p in approx]
                    
                    # Debug Poly
                    xs = [p['x'] for p in polygon_list]
                    ys = [p['y'] for p in polygon_list]
                    print(f"DEBUG: Poly Bounds: x[{min(xs):.1f}, {max(xs):.1f}], y[{min(ys):.1f}, {max(ys):.1f}]")
                
                # Sample Grid Points inside Mask
                support_points_list = []
                
                # Revert to Dynamic step size (V33)
                # target_step = 15
                
                w_box = bbox.x_max - bbox.x_min
                h_box = bbox.y_max - bbox.y_min

                step = max(int(min(w_box, h_box) / 7), 5) # V35: Medium density (Target ~50 points)
                
                # Global Grid Alignment
                start_x = math.ceil(bbox.x_min / step) * step
                start_y = math.ceil(bbox.y_min / step) * step

                # Scan BBox area
                y_range = range(start_y, int(bbox.y_max), step)
                x_range = range(start_x, int(bbox.x_max), step)
                
                for y in y_range:
                    for x in x_range:
                        # Ensure within image bounds
                        if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
                            continue
                            
                        # Check mask
                        if mask[y, x] > 0:
                             support_points_list.append({"x": float(x), "y": float(y), "id": -1})
                

                
                
                # Check if we have points
                if not support_points_list:
                    print("SAM 2 mask found, but grid (step=15) missed it. Retrying with finer grid...")
                    
                    # Retry with finer grid (step=5)
                    fine_step = 5
                    start_x_f = math.ceil(bbox.x_min / fine_step) * fine_step
                    start_y_f = math.ceil(bbox.y_min / fine_step) * fine_step
                    for y in range(start_y_f, int(bbox.y_max), fine_step):
                        for x in range(start_x_f, int(bbox.x_max), fine_step):
                             if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]: continue
                             if mask[y, x] > 0:
                                 support_points_list.append({"x": float(x), "y": float(y), "id": -1})
                    
                    # If STILL empty, use Centroid
                    if not support_points_list:
                        print("Finer grid also empty. Using Centroid.")
                        M = cv2.moments(mask)
                        if M["m00"] != 0:
                            cX = M["m10"] / M["m00"]
                            cY = M["m01"] / M["m00"]
                            support_points_list.append({"x": float(cX), "y": float(cY), "id": -1})
                        else:
                            # This implies mask is empty? But contour check passed.
                            # Just in case, create one point at bbox center if mask area is > 0 (which it should be)
                            # If mask is truly empty, we probably shouldn't be here.
                             print("Mask moments 0. Mask might be empty or noise.")
                             pass

                # If we still have 0 points, it means mask is practically non-existent or noise.
                # In that case, returning empty list is better than points on background.
                # But to avoid crashing CoTracker which might expect points, we can return empty.
                if not support_points_list:
                     print("Warning: Could not generate ANY support points on mask. Returning empty points.")
                     return mask, [], polygon_list

                # V34: Cap max points to prevent overload
                if len(support_points_list) > 50:
                    import random
                    support_points_list = random.sample(support_points_list, 50)

                print(f"SAM 2 Success: {len(support_points_list)} points, {len(polygon_list)} polygon vertices.")
                return mask, support_points_list, polygon_list
                
        except Exception as e:
            import traceback
            print(f"SAM 2 Inference Error: {e}")
            traceback.print_exc()
            # V30: Do NOT use fallback grid on error. Return empty to avoid misleading points.
            # return self._fallback_grid_points(bbox)
            return None, [], []

    def _fallback_grid_points(self, bbox: BBox):
        support_points = []
        w = bbox.x_max - bbox.x_min
        h = bbox.y_max - bbox.y_min
        for i in range(1, 5):
            for j in range(1, 5):
                px = bbox.x_min + (w * i / 5)
                py = bbox.y_min + (h * j / 5)
                support_points.append({"x": px, "y": py, "id": -1})
        return None, support_points, []

    # Helper for unified CoTracker Execution
    def _run_cotracker(self, frames: List[str], input_leaves: List[LeafAnnotation], global_start_idx: int, is_reversed: bool = False) -> Dict[int, List[LeafAnnotation]]:
        if not input_leaves or not frames:
            return {}

        predictor = model_loader.cotracker_predictor
        if not predictor: 
            return {}
            
        torch.cuda.empty_cache()

        # Prepare First Frame (Reference for Scale)
        first_path = frames[0]
        orig_img = cv2.imread(first_path)
        if orig_img is None: return {}
        
        h_orig, w_orig = orig_img.shape[:2]
        target_w, target_h = 1024, 768
        scale = min(target_w / w_orig, target_h / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        pad_x, pad_y = int((target_w - new_w) / 2), int((target_h - new_h) / 2)
        pad_y_b, pad_x_r = target_h - new_h - pad_y, target_w - new_w - pad_x
        
        # Prepare Queries
        track_points = []
        point_metadata = []
        for leaf in input_leaves:
            for p in leaf.points:
                 tx, ty = float(p.x * scale) + pad_x, float(p.y * scale) + pad_y
                 track_points.append([0.0, tx, ty])
                 point_metadata.append({'leaf_id': leaf.id, 'type': 'main', 'obj': p})
            for p in leaf.support_points:
                 tx, ty = float(p.x * scale) + pad_x, float(p.y * scale) + pad_y
                 track_points.append([0.0, tx, ty])
                 point_metadata.append({'leaf_id': leaf.id, 'type': 'support', 'obj': p})
                 
        if not track_points: return {}
        queries = torch.tensor([track_points], device=model_loader.cotracker_device).float()
        
        # Process in Chunks (CoTracker Limit typically ~200-300 frames safely?)
        # For bidirectional segment, segment might be small. 
        # Assuming segment fits in memory or we chunk it?
        # If segment is huge (e.g. 0 to 1000), we probably shouldn't do single pass.
        # But for V8 prototype, let's process reasonable size.
        # If frames list is huge, we chunk it.
        
        results: Dict[int, List[LeafAnnotation]] = {}
        
        # Determine internal chunks
        # V46: Adjusted for RTX 3090 (24GB VRAM). 
        # 350 was risky. Reducing to 150 for safety.
        SUB_CHUNK = 150
        
        # Logic: If we chunk, we must propagate.
        # This function tracks Forward only (relative to 'frames' list).
        
        current_leaves_state = input_leaves
        current_frame_offset = 0
        
        while current_frame_offset < len(frames):
            remaining = len(frames) - current_frame_offset
            length = min(SUB_CHUNK, remaining)
            # Need at least 2 frames?
            if length < 2 and remaining < 2: 
                # If 1 frame left and it's the start, we have query.
                # If we tracked up to here, we have result.
                break
            
            chunk_slice = frames[current_frame_offset : current_frame_offset + length]
            
            # Prepare Video Tensor
            video_tensor_list = []
            last_valid = orig_img
            for i, path in enumerate(chunk_slice):
                img = orig_img if (i==0 and current_frame_offset==0) else cv2.imread(path)
                if img is None: img = last_valid
                else: last_valid = img
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img, (new_w, new_h))
                padded = cv2.copyMakeBorder(resized, pad_y, pad_y_b, pad_x, pad_x_r, cv2.BORDER_CONSTANT, value=(0,0,0))
                video_tensor_list.append(torch.from_numpy(padded).permute(2, 0, 1))

            video = torch.stack(video_tensor_list).unsqueeze(0).float().to(model_loader.cotracker_device)
            print(f"DEBUG: CoTracker Video Tensor Stats: Min={video.min().item()}, Max={video.max().item()}, Shape={video.shape}")
            
            # Inference
            # Queries must be updated if this is not the first chunk
            if current_frame_offset > 0:
                 # Regenerate queries from current_leaves_state
                 track_points = []
                 point_metadata = []
                 for leaf in current_leaves_state:
                    for p in leaf.points:
                         tx, ty = float(p.x * scale) + pad_x, float(p.y * scale) + pad_y
                         track_points.append([0.0, tx, ty])
                         point_metadata.append({'leaf_id': leaf.id, 'type': 'main', 'obj': p})
                    for p in leaf.support_points:
                         tx, ty = float(p.x * scale) + pad_x, float(p.y * scale) + pad_y
                         track_points.append([0.0, tx, ty])
                         point_metadata.append({'leaf_id': leaf.id, 'type': 'support', 'obj': p})
                 if track_points:
                     queries = torch.tensor([track_points], device=model_loader.cotracker_device).float()
                 else:
                     break

            print(f"DEBUG: CoTracker Queries: Shape={queries.shape}")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_tracks, _ = predictor(video, queries=queries)
            
            tracks_np = pred_tracks[0].cpu().numpy()
            
            # Debug Movement
            if len(tracks_np) > 1:
                diff = np.linalg.norm(tracks_np[-1] - tracks_np[0], axis=-1)
                avg_move = np.mean(diff)
                print(f"DEBUG: Average Point Movement over {len(tracks_np)} frames: {avg_move:.2f} px")
                
                # Visualization (Debug)
                # try:
                #     self._visualize_tracks(video, tracks_np, filename="debug_tracking.mp4")
                # except Exception as e:
                #     print(f"Visualization Failed: {e}")
            
            chunk_leaves_last = []
            
            for t in range(len(chunk_slice)):
                # Map to global Frame Index
                if is_reversed:
                    # frames is reversed list
                    # frames[0] is global_start_idx (which is actually End Frame of segment)
                    # frames[t] corresponds to global_start_idx - t (if we assume contiguous)
                    # Actually global_start_idx is the index of frames[0].
                    # Wait, caller passes global_start_idx = End Frame index?
                    # Let's simplify: caller manages mapping.
                    # We just return Dict[relative_index, List[Leaf]]?
                    # Or Dict[str(filename), ...]
                    # Let's return Dict[int, ...] mapped to global idx
                    
                    # Logic: 
                    # If is_reversed: frames = [F50, F49, F48...]
                    # chunk_slice[t] is F(50-offset-t)
                    # We need actual global index.
                    # This requires parsing filename or caller passing logic.
                    # Let's assume global_start_idx is the index of frames[0] in the GLOBAL sequence.
                    # And sequence is contiguous.
                    current_global_idx = global_start_idx - (current_frame_offset + t)
                else:
                    current_global_idx = global_start_idx + current_frame_offset + t

                # Construct Leaves
                leaf_map: Dict[int, LeafAnnotation] = {}
                for input_leaf in current_leaves_state:
                     # Propagate mask_polygon from source if available
                     # Note: This is a static copy, won't deform with tracking, but keeps visualization
                     leaf_map[input_leaf.id] = LeafAnnotation(
                         id=input_leaf.id,
                         bbox=None, points=[], support_points=[], 
                         mask_polygon=input_leaf.mask_polygon,
                         manual=False
                     )

                current_pts = tracks_np[t]
                for i, (x, y) in enumerate(current_pts):
                    meta = point_metadata[i]
                    lid = meta['leaf_id']
                    sx, sy = (x - pad_x) / scale, (y - pad_y) / scale
                    new_p = Point(x=sx, y=sy, id=meta['obj'].id)
                    target = leaf_map[lid].points if meta['type'] == 'main' else leaf_map[lid].support_points
                    target.append(new_p)
                
                final_leaves = []
                for leaf in leaf_map.values():
                    # Outlier Filtering (Support Points)
                    if len(leaf.support_points) > 5:
                         leaf.support_points = self._filter_outliers(leaf.support_points)

                    # BBox Check
                    all_pts = leaf.points + leaf.support_points
                    if all_pts:
                        xs, ys = [p.x for p in all_pts], [p.y for p in all_pts]
                        # Add Padding (V5: "Slightly Larger" request)
                        pad = 10
                        min_x = max(0, min(xs) - pad)
                        min_y = max(0, min(ys) - pad)
                        max_x = min(w_orig, max(xs) + pad)
                        max_y = min(h_orig, max(ys) + pad)
                        
                        leaf.bbox = BBox(x_min=float(min_x), y_min=float(min_y), x_max=float(max_x), y_max=float(max_y))
                        
                    final_leaves.append(leaf)
                
                results[current_global_idx] = final_leaves
                if t == len(chunk_slice) - 1:
                    chunk_leaves_last = final_leaves
            
            # Next Chunk
            current_leaves_state = chunk_leaves_last
            current_frame_offset += (len(chunk_slice) - 1) # Overlap 1 frame
            if len(chunk_slice) < SUB_CHUNK: break
            
        return results

    def _filter_outliers(self, points: List[Point]) -> List[Point]:
        """
        Filter out points that are > Mean + 2*StdDev from the centroid.
        """
        if len(points) < 4: return points
        
        # Centroid
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # Distances
        dists = [math.sqrt((p.x - cx)**2 + (p.y - cy)**2) for p in points]
        mean_dist = sum(dists) / len(dists)
        variance = sum([(d - mean_dist)**2 for d in dists]) / len(dists)
        std_dev = math.sqrt(variance)
        
        threshold = mean_dist + 2.0 * std_dev
        
        filtered = []
        for i, p in enumerate(points):
             if dists[i] <= threshold:
                 filtered.append(p)
             # else: print(f"DEBUG: Removed outlier point at dist {dists[i]:.2f} (Thresh: {threshold:.2f})")
             
        return filtered

    def _visualize_tracks(self, video_tensor, tracks, filename="debug_tracking.mp4"):
        """
        Save a video with tracking visualization.
        video_tensor: [1, T, 3, H, W] (Float 0-255? or 0-1? Check usage. It was 0-255 in _run_cotracker)
        tracks: [T, N, 2]
        """
        print(f"DEBUG: Generating visualization to {filename}...")
        
        # Tensor to Numpy [T, H, W, 3]
        vid = video_tensor[0].cpu().numpy().transpose(0, 2, 3, 1) # [T, H, W, 3]
        # It is RGB, 0-255 (if padded with 0-0-0 and read by cv2)
        # cv2.imread is 0-255.
        
        T, H, W, C = vid.shape
        fps = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (W, H))
        
        for t in range(T):
            frame = vid[t].astype(np.uint8).copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Back to BGR for Opencv
            
            pts = tracks[t] # [N, 2]
            for val in pts:
                 # Check visibility? CoTracker returns visibility in [T, N, 1]? 
                 # pred_tracks is [1, T, N, 2]. Wait, does it return visibility?
                 # pred_tracks, pred_visibility = predictor(...)
                 # We ignored visibility.
                 
                 x, y = int(val[0]), int(val[1])
                 cv2.circle(frame, (x, y), 3, (0, 0, 255), -1) # Red dot
            
            out.write(frame)
            
        out.release()
        print(f"DEBUG: Visualization saved to {os.path.abspath(filename)}")

    def execute_tracking(self, start_frame_idx: int, initial_leaves: List[LeafAnnotation], keyframes: Dict[int, List[LeafAnnotation]] = None, image_paths: List[str] = None) -> Dict[int, TrackingResult]:
        print(f"DEBUG: execute_tracking (V44 Dynamic Propagation) called from {start_frame_idx}")
        print(f"DEBUG: execute_tracking args: image_paths len={len(image_paths) if image_paths else 'None'}")
        self.status = "running"
        self.progress = 0.0
        
        try:
            # V49: Support Dense Tracking (Override logic)
            if image_paths:
                all_frames = image_paths
                print(f"DEBUG: Using provided dense image list ({len(all_frames)} frames)")
            else:
                all_frames = image_loader.images
            
            total_frames = len(all_frames)
            
            # 1. Prepare Keyframes
            # Ensure start_frame is in keyframes
            prompts = keyframes.copy() if keyframes else {}
            prompts[start_frame_idx] = initial_leaves
            
            # Sort Keyframe Indices
            sorted_keys = sorted(prompts.keys())
            print(f"DEBUG: Keyframes at {sorted_keys}")
            
            all_results: Dict[int, TrackingResult] = {}
            
            # Helper to merge point lists (Overwrite collision)
            def merge_into_global(frame_idx, leaves: List[LeafAnnotation]):
                if frame_idx not in all_results:
                    all_results[frame_idx] = TrackingResult(leaves=[])
                all_results[frame_idx].leaves = leaves

            # Dynamic Prompt Update Helper
            def get_propagated_prompt(k_idx, manual_leaves: List[LeafAnnotation]):
                # If we have tracked results for this frame (from previous segment), merge them with manual leaves.
                # Priority: Manual > Tracked
                if k_idx in all_results and all_results[k_idx].leaves:
                    tracked_leaves = all_results[k_idx].leaves
                    manual_ids = {l.id for l in manual_leaves}
                    
                    merged = list(manual_leaves)
                    for l in tracked_leaves:
                        if l.id not in manual_ids:
                            # Propagate: Set manual=False just in case
                            l_copy = LeafAnnotation(**l.dict())
                            l_copy.manual = False 
                            merged.append(l_copy)
                            print(f"DEBUG: Propagated Leaf {l.id} through Keyframe {k_idx}")
                    return merged
                return manual_leaves

            # Segment Processing
            
            # 2. Head Segment (Start -> First Keyframe)
            if sorted_keys[0] > 0:
                k0 = sorted_keys[0]
                print(f"DEBUG: Processing Head Segment (0 <- {k0})")
                frames_head = all_frames[:k0+1][::-1] 
                # Run Backward
                res_head = self._run_cotracker(frames_head, prompts[k0], k0, is_reversed=True)
                for f, leaves in res_head.items():
                    if f < k0:
                        merge_into_global(f, leaves)

            # 3. Middle Segments (Ki <-> Ki+1)
            for i in range(len(sorted_keys) - 1):
                k_start = sorted_keys[i]
                k_end = sorted_keys[i+1]
                print(f"DEBUG: Processing Segment {i}: [{k_start} <-> {k_end}]")
                
                # Get Source Prompts (Dynamic)
                # For Start of Segment, we use what's in prompts (Manual) OR what was merged previously.
                # Actually, prompts dict is static manual data. 
                # We need to construct the *effective* start leaves.
                
                # V44: The critical fix. 
                # For the START of this segment, we use the merged result if available (which carries history).
                # But wait, k_start IS the end of the previous segment.
                # So if i > 0, we can look at all_results[k_start].
                
                if i == 0 and sorted_keys[0] == 0:
                     # Very first frame, just manual
                     start_leaves = prompts[k_start]
                else:
                     # Attempt to pull from global results if they exist (contains propagation)
                     # Fallback to manual prompt
                     start_leaves = get_propagated_prompt(k_start, prompts[k_start])

                # End Prompt is strictly manual (Target Anchor) ... NO!
                # If we want Backward Tracking to also respect propagated leaves?
                # No, Backward Tracking from k_end relies on k_end's state.
                # If Leaf X is missing at k_end (user didn't annotate), we can't track it backward from there.
                # Leaf X is being tracked FORWARD from k_start.
                # So in Blending:
                # Leaf X exists in Fwd Result, but NOT in Bwd Result.
                # Our blending logic (Case 2) handles "Only Fwd" -> Keep it.
                # So the issue is: Does `res_fwd` actually contain Leaf X?
                # Yes, if we pass the propagated list to `run_cotracker`.
                
                # Forward Tracking Target (Source)
                leaves_for_fwd = start_leaves
                
                # Backward Tracking Target (Source)
                # Logic: We track BACKWARD from k_end.
                # If Leaf X is missing in k_end, we don't track it backward.
                # That is correct behavior (it might have disappeared).
                # BUT, if it shouldn't have disappeared, user should have annotated it at k_end?
                # The user says "Leaf 1 annotated at Frame 0, NOT at Frame N".
                # "Frame N has Leaf 0".
                # User expects Leaf 1 to appear at N and continue.
                # So Forward Tracking from 0 -> N works.
                # Backward Tracking from N -> 0: Leaf 1 is missing, so it returns nothing for Leaf 1.
                # Blend: Leaf 1 Fwd exists, Bwd missing -> "Only Fwd" case -> Result: Leaf 1 exists.
                # So Segment 0->N generates Leaf 1 at Frame N.
                # The result is stored in all_results[N].
                #
                # NEXT SEGMENT (N -> End):
                # We start Forward from N.
                # If we only use prompts[N] (Manual Leaf 0), Leaf 1 is dropped.
                # We MUST use the result we just generated (Leaf 0 + Leaf 1) as the source for N->End.
                
                # Fwd: k_start -> k_end
                frames_fwd = all_frames[k_start : k_end + 1]
                res_fwd = self._run_cotracker(frames_fwd, leaves_for_fwd, k_start, is_reversed=False)
                
                # Bwd: k_end -> k_start
                # Optimize: Limit backward tracking to max 60 frames
                start_limit = max(k_start, k_end - 60)
                frames_bwd = all_frames[start_limit : k_end + 1][::-1] 
                res_bwd = self._run_cotracker(frames_bwd, prompts[k_end], k_end, is_reversed=True)
                
                # Blend
                for t in range(k_start, k_end + 1):
                    alpha = (t - k_start) / (k_end - k_start)
                    leaves_fwd = {l.id: l for l in res_fwd.get(t, [])}
                    leaves_bwd = {l.id: l for l in res_bwd.get(t, [])}
                    
                    if alpha > 0.8 and not leaves_bwd:
                         print(f"WARNING: Frame {t} (Alpha={alpha:.2f}) has NO Backward Tracking results! This will cause a jump.")
                    
                    all_ids = set(leaves_fwd.keys()) | set(leaves_bwd.keys())
                    blended_leaves = []
                    
                    for lid in all_ids:
                        lf = leaves_fwd.get(lid)
                        lb = leaves_bwd.get(lid)
                        
                        if lf and lb: # Blend
                            # ... (Same blend logic)
                            pmap_f = {p.id: p for p in lf.points}
                            pmap_b = {p.id: p for p in lb.points}
                            final_points = []
                            final_support = []
                            
                            # Main Points
                            main_ids = set(pmap_f.keys()) | set(pmap_b.keys())
                            for pid in main_ids:
                                pf = pmap_f.get(pid)
                                pb = pmap_b.get(pid)
                                if pf and pb:
                                    nx = pf.x * (1 - alpha) + pb.x * alpha
                                    ny = pf.y * (1 - alpha) + pb.y * alpha
                                    final_points.append(Point(x=nx, y=ny, id=pid))
                                elif pf: final_points.append(pf)
                                elif pb: final_points.append(pb)
                                    
                            # Support Points
                            count_f = len(lf.support_points)
                            count_b = len(lb.support_points)
                            is_mismatch = abs(count_f - count_b) > max(count_f, count_b) * 0.2
                            
                            if is_mismatch:
                                final_support = lf.support_points if alpha < 0.5 else lb.support_points
                                final_mask = lf.mask_polygon if alpha < 0.5 else lb.mask_polygon
                            else:
                                max_idx = max(count_f, count_b)
                                for idx in range(max_idx):
                                    sp_f = lf.support_points[idx] if idx < count_f else None
                                    sp_b = lb.support_points[idx] if idx < count_b else None
                                    if sp_f and sp_b:
                                        nx = sp_f.x * (1 - alpha) + sp_b.x * alpha
                                        ny = sp_f.y * (1 - alpha) + sp_b.y * alpha
                                        final_support.append(Point(x=nx, y=ny, id=-1))
                                    elif sp_f:
                                        if alpha < 0.5: final_support.append(sp_f)
                                    elif sp_b:
                                        if alpha >= 0.5: final_support.append(sp_b)
                                final_mask = lf.mask_polygon if alpha < 0.5 else lb.mask_polygon # Fallback mask blend logic?

                            all_p = final_points + final_support
                            bbox = None
                            if all_p:
                                xs, ys = [p.x for p in all_p], [p.y for p in all_p]
                                bbox = BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))
                                
                            blended_leaves.append(LeafAnnotation(
                                id=lid, bbox=bbox, points=final_points, 
                                support_points=final_support, mask_polygon=final_mask, manual=False
                            ))
                            
                        elif lf: blended_leaves.append(lf)
                        elif lb: blended_leaves.append(lb)

                    merge_into_global(t, blended_leaves)
                self.progress = (i + 1) / len(sorted_keys) * 100

            # 4. Tail Segment (Last Keyframe -> End)
            k_last = sorted_keys[-1]
            if k_last < total_frames - 1:
                print(f"DEBUG: Processing Tail Segment ({k_last} -> End)")
                frames_tail = all_frames[k_last:]
                # V44: Use Propagated Prompt!
                tail_start_leaves = get_propagated_prompt(k_last, prompts[k_last])
                
                # DEBUG LOG
                print(f"DEBUG: Tail Segment Start Leaves: {[l.id for l in tail_start_leaves]}")
                
                res_tail = self._run_cotracker(frames_tail, tail_start_leaves, k_last, is_reversed=False)
                print(f"DEBUG: Tail Result Frames: {len(res_tail)}")
                if res_tail:
                     sample_f = list(res_tail.keys())[0]
                     print(f"DEBUG: Sample Frame {sample_f} Leaves: {[l.id for l in res_tail[sample_f]]}")

                for f, leaves in res_tail.items():
                    merge_into_global(f, leaves)

            # 5. Restore Keyframes (Ground Truth)
            # We must be careful. 
            # If we enforce prompts, we deleted the propagated leaves at the exact frame indices of keyframes.
            # But get_propagated_prompt logic suggests we want to KEEP the propagated ones combined with manual.
            # So we should update all_results[k] using get_propagated_prompt logic again.
            
            for k, manual_leaves in prompts.items():
                if k in all_results:
                     # Merge the just-calculated tracking results (which act as propagation) + manual
                     propagated = get_propagated_prompt(k, manual_leaves)
                     all_results[k] = TrackingResult(leaves=propagated)
                else:
                     all_results[k] = TrackingResult(leaves=manual_leaves)

            print(f"DEBUG: V44 Tracking Complete. Total Frames: {len(all_results)}")
            
            # Post-processing: Trajectory Smoothing (V42)
            print("DEBUG: Applying Smoothing...")
            # Same smoothing logic ...
            trajectories = {}
            fixed_frames_per_leaf = {} 
            for k_idx, leaves in prompts.items():
                for l in leaves:
                     if l.id not in fixed_frames_per_leaf: fixed_frames_per_leaf[l.id] = set()
                     fixed_frames_per_leaf[l.id].add(k_idx)

            for f_idx, res in all_results.items():
                for l in res.leaves:
                    if l.id not in trajectories: trajectories[l.id] = {}
                    for p in l.points:
                        if p.id not in trajectories[l.id]: trajectories[l.id][p.id] = {}
                        trajectories[l.id][p.id][f_idx] = p
                    for i, p in enumerate(l.support_points):
                        pid = f"S{i}"
                        if pid not in trajectories[l.id]: trajectories[l.id][pid] = {}
                        trajectories[l.id][pid][f_idx] = p

            window = 2
            for lid, leaf_tracks in trajectories.items():
                fixed_frames = fixed_frames_per_leaf.get(lid, set())
                for pid, time_series in leaf_tracks.items():
                    frames = sorted(time_series.keys())
                    if len(frames) < 3: continue
                    coords_map = {f: (time_series[f].x, time_series[f].y) for f in frames}
                    for i, f in enumerate(frames):
                        if f in fixed_frames: continue
                        
                        num_x, num_y, den = 0.0, 0.0, 0.0
                        for w in range(-window, window + 1):
                            neighbor_idx = i + w
                            if 0 <= neighbor_idx < len(frames):
                                nf = frames[neighbor_idx]
                                if abs(nf - f) <= window: 
                                    weight = 1.0
                                    nx, ny = coords_map[nf]
                                    num_x += nx * weight
                                    num_y += ny * weight
                                    den += weight
                        if den > 0:
                            time_series[f].x = num_x / den
                            time_series[f].y = num_y / den

            return all_results
            
        except Exception as e:
            print(f"Tracking Execution Failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            self.progress = 100.0



tracking_engine = TrackingEngine()
