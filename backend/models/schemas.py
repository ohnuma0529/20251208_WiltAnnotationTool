from pydantic import BaseModel
from typing import List, Optional, Tuple

class Point(BaseModel):
    x: float
    y: float
    id: int # 0: Base, 1: Tip, -1: Support
    
class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class LeafAnnotation(BaseModel):
    id: int
    bbox: Optional[BBox] = None
    points: List[Point] = []
    support_points: List[Point] = []
    mask_polygon: Optional[List[dict]] = []
    manual: bool = False
    
class InitTrackingRequest(BaseModel):
    frame_index: int
    leaves: List[LeafAnnotation]

class SaveFrameRequest(BaseModel):
    frame_index: int
    leaves: List[LeafAnnotation]

class DeleteLeafRequest(BaseModel):
    frame_index: int
    leaf_id: Optional[int] = None
    delete_all: bool = False
    delete_global: bool = False

class PreviewPointsRequest(BaseModel):
    frame_index: int
    bbox: BBox


class UpdateRegionRequest(BaseModel):
    frame_index: int
    leaf_id: int
    bbox: BBox # New bbox for face correction
    
class UpdatePointRequest(BaseModel):
    frame_index: int
    leaf_id: int
    point_id: int
    x: float
    y: float

class FrameData(BaseModel):
    filename: str
    frame_index: int
    timestamp: str # HH:MM

class TrackingResult(BaseModel):
    leaves: List[LeafAnnotation] # Reusing LeafAnnotation structure for result

class SetFilterRequest(BaseModel):
    unit: str
    date: str
    frequency: int

