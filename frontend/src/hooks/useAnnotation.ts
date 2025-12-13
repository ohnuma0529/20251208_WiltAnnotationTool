import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// API Base URL
const API_BASE = '/api';
const API_URL = API_BASE; // Alias for legacy if needed

export interface Point {
    x: number;
    y: number;
    id: number; // 0: Base, 1: Tip
}

export interface BBox {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
}

export interface FrameData {
    filename: string;
    frame_index: number;
    timestamp: string;
}

// Interface for Multi-Leaf
export interface Leaf {
    id: number;
    bbox: BBox | null;
    points: Point[]; // Base(0), Tip(1)
    supportPoints: Point[]; // ID -1
    maskPolygon?: Point[];
    color: string;
}

const LEAF_COLORS = [
    '#FF0000', // Red
    '#00FF00', // Green
    '#0000FF', // Blue
    '#FFFF00', // Yellow
    '#FF00FF', // Magenta
    '#00FFFF', // Cyan
    '#FFA500', // Orange
    '#800080'  // Purple
];

export const useAnnotation = () => {
    const [frames, setFrames] = useState<FrameData[]>([]);
    const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(0);
    const [loading, setLoading] = useState<boolean>(false);

    // Filter State
    const [units, setUnits] = useState<string[]>([]);
    const [selectedUnit, setSelectedUnit] = useState<string>('');
    const [dates, setDates] = useState<string[]>([]);
    const [selectedDate, setSelectedDate] = useState<string>('');
    const [frequency, setFrequency] = useState<number>(10);

    // Multi-Leaf State
    const [leaves, setLeaves] = useState<Leaf[]>([]);
    const [tempLeaf, setTempLeaf] = useState<Partial<Leaf>>({});
    const [trackingStarted, setTrackingStarted] = useState<boolean>(false);

    // Annotations Cache
    const [annotations, setAnnotations] = useState<Record<number, any>>({});

    // Load Units on mount
    useEffect(() => {
        let isMounted = true;
        axios.get(`${API_BASE}/units`)
            .then(res => {
                if (!isMounted) return;
                const availUnits = res.data.units;
                setUnits(availUnits);
                if (availUnits.length > 0) {
                    setSelectedUnit(availUnits[0]);
                }
            })
            .catch(err => console.error("Failed to load units", err));
        return () => { isMounted = false; };
    }, []);

    // Load Dates when Unit changes
    useEffect(() => {
        if (!selectedUnit) return;

        let isActive = true;
        axios.get(`${API_BASE}/dates?unit=${selectedUnit}`)
            .then(res => {
                if (!isActive) return;
                const availableDates = res.data.dates;
                setDates(availableDates);

                // Always default to the first date when Unit changes
                // preventing stale date issues or empty display
                if (availableDates.length > 0) {
                    // Check if current selectedDate is valid for this unit? 
                    // Usually units have distinct dates, so just switch to first.
                    // If we are strictly initializing, selectedDate might be empty.
                    // If switching units, we overwrite it.
                    changeDate(availableDates[0]);
                } else {
                    setFrames([]);
                    setSelectedDate('');
                }
            })
            .catch(err => console.error("Failed to load dates", err));

        return () => { isActive = false; };
    }, [selectedUnit]);


    // Fetch Annotations
    const fetchAnnotations = useCallback(async () => {
        try {
            const res = await axios.get(`${API_BASE}/annotations?t=${new Date().getTime()}`);
            setAnnotations(res.data);
        } catch (err) {
            console.error("Failed to fetch annotations", err);
        }
    }, []);

    // --- Core Data Fetching ---

    const fetchFrames = async (freqOverride?: number) => {
        setLoading(true);
        try {
            const f = freqOverride !== undefined ? freqOverride : frequency;
            const res = await axios.get(`${API_BASE}/images?frequency=${f}`);
            setFrames(res.data);
            // If current index is out of bounds, reset
            if (currentFrameIndex >= res.data.length && res.data.length > 0) {
                setCurrentFrameIndex(0);
            }
            // Fetch annotations whenever frames reload (context switch)
            await fetchAnnotations();
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };


    const changeUnit = (unit: string) => {
        setSelectedUnit(unit);
        // Date loading is handled by useEffect to ensure state consistency
    };

    const changeDate = async (date: string) => {
        setSelectedDate(date);
        try {
            // First set filter in backend (ensure Dense Loading)
            await axios.post(`${API_BASE}/set_filter`, { unit: selectedUnit, date: date, frequency: 1 });

            // Then fetch frames with current sparse frequency
            setLeaves([]);
            setTempLeaf({});
            setTrackingStarted(false);

            await fetchFrames(frequency);
        } catch (e) { console.error(e); }
    };

    const changeFrequency = async (f: number) => {
        setFrequency(f);
        // No need to reload backend images (already dense), just re-fetch sparse list
        fetchFrames(f);
    };

    const truncateFrames = async () => {
        if (!frames[currentFrameIndex]) return;
        try {
            // Use dense frame index
            const denseIdx = frames[currentFrameIndex].frame_index;
            const res = await axios.post(`${API_BASE}/truncate_frames`, { frame_index: denseIdx });
            if (res.data.status === 'success') {
                // alert(`Truncated ${res.data.deleted_frames} frames.`);
                // Refresh
                fetchFrames();
            }
        } catch (e) {
            console.error(e);
            alert("Failed to truncate.");
        }
    };

    // --- Interaction ---

    // Playback State
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    const nextFrame = () => {
        setCurrentFrameIndex(prev => (prev + 1) < frames.length ? prev + 1 : prev);
    };

    const prevFrame = () => {
        setCurrentFrameIndex(prev => (prev - 1) >= 0 ? prev - 1 : prev);
    };

    const togglePlay = () => {
        setIsPlaying(!isPlaying);
    };

    useEffect(() => {
        let interval: any;
        if (isPlaying) {
            interval = setInterval(() => {
                setCurrentFrameIndex(prev => {
                    if (prev + 1 >= frames.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 200); // 5fps
        }
        return () => clearInterval(interval);
    }, [isPlaying, frames.length]);

    // Preload Next Frames
    useEffect(() => {
        if (frames.length === 0) return;
        const PRELOAD_COUNT = 10;
        const baseUrl = ''; // Assuming API_BASE is /api

        for (let i = 1; i <= PRELOAD_COUNT; i++) {
            const nextIndex = currentFrameIndex + i;
            if (nextIndex < frames.length) {
                const img = new Image();
                img.src = `${baseUrl}/images/${frames[nextIndex].filename}`;
            }
        }
    }, [currentFrameIndex, frames]);

    // Sync current frame with annotations (Multi-Leaf)
    useEffect(() => {
        if (frames.length === 0) return;
        const frame = frames[currentFrameIndex];
        // Use Dense Index to look up annotation
        const denseIdx = frame ? frame.frame_index : -1;

        const ann = annotations[denseIdx];
        if (ann && ann.leaves) {
            const restoredLeaves = ann.leaves.map((l: any) => ({
                ...l,
                supportPoints: l.support_points || l.supportPoints || [],
                maskPolygon: l.mask_polygon || l.maskPolygon || [],
                color: LEAF_COLORS[l.id % LEAF_COLORS.length]
            }));
            setLeaves(restoredLeaves);
        } else {
            setLeaves([]);
        }
        setTempLeaf({});
    }, [currentFrameIndex, annotations, frames]); // Added frames dep

    // Helper: Calculate IoU
    const calculateIoU = (boxA: BBox, boxB: BBox) => {
        const xA = Math.max(boxA.x_min, boxB.x_min);
        const yA = Math.max(boxA.y_min, boxB.y_min);
        const xB = Math.min(boxA.x_max, boxB.x_max);
        const yB = Math.min(boxA.y_max, boxB.y_max);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const boxAArea = (boxA.x_max - boxA.x_min) * (boxA.y_max - boxA.y_min);
        const boxBArea = (boxB.x_max - boxB.x_min) * (boxB.y_max - boxB.y_min);

        return interArea / (boxAArea + boxBArea - interArea);
    };

    // Preview Support Points
    const previewPoints = async (bboxVal: BBox) => {
        // Check for overlap with existing leaves
        let overlappingLeafId: number | null = null;
        let maxIoU = 0;

        leaves.forEach(l => {
            if (l.bbox) {
                // Aggressive Matching (V37)
                // Calculate Intersection Area
                const xA = Math.max(bboxVal.x_min, l.bbox.x_min);
                const yA = Math.max(bboxVal.y_min, l.bbox.y_min);
                const xB = Math.min(bboxVal.x_max, l.bbox.x_max);
                const yB = Math.min(bboxVal.y_max, l.bbox.y_max);
                const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);

                const boxAArea = (bboxVal.x_max - bboxVal.x_min) * (bboxVal.y_max - bboxVal.y_min);
                const boxBArea = (l.bbox.x_max - l.bbox.x_min) * (l.bbox.y_max - l.bbox.y_min);
                const minArea = Math.min(boxAArea, boxBArea);
                const unionArea = boxAArea + boxBArea - interArea;

                const overlapRatio = minArea > 0 ? interArea / minArea : 0;
                const iou = unionArea > 0 ? interArea / unionArea : 0;

                // Combined Score: OverlapRatio (handling inclusions) + IoU (handling exact matches)
                const score = overlapRatio + iou;

                // Threshold: 0.5 (e.g., 30% overlap + 20% IoU is enough)
                if (score > 0.5 && score > maxIoU) {
                    maxIoU = score; // Use Score as metric
                    overlappingLeafId = l.id;
                }
            }
        });

        const denseIdx = frames[currentFrameIndex] ? frames[currentFrameIndex].frame_index : 0;

        try {
            const res = await axios.post(`${API_BASE}/preview_points`, {
                frame_index: denseIdx,
                bbox: bboxVal
            });

            if (overlappingLeafId !== null) {
                // Update existing leaf
                const updatedLeaves = leaves.map(l => {
                    if (l.id === overlappingLeafId) {
                        return {
                            ...l,
                            bbox: bboxVal,
                            supportPoints: res.data.points,
                            maskPolygon: res.data.polygon
                        };
                    }
                    return l;
                });
                setLeaves(updatedLeaves);
                // Persist
                const payload = updatedLeaves.map(l => ({
                    ...l,
                    support_points: l.supportPoints,
                    mask_polygon: l.maskPolygon
                }));
                await axios.post(`${API_BASE}/save_frame`, {
                    frame_index: denseIdx,
                    leaves: payload
                });

            } else {
                // Create new Temp Leaf
                const newId = leaves.length > 0 ? Math.max(...leaves.map(l => l.id)) + 1 : 0;
                setTempLeaf({
                    id: newId,
                    bbox: bboxVal,
                    points: [],
                    supportPoints: res.data.points,
                    maskPolygon: res.data.polygon,
                    color: LEAF_COLORS[newId % LEAF_COLORS.length]
                });
            }
        } catch (err) {
            console.error("Preview failed", err);
        }
    };

    const deleteLeaf = async (leafId: number, deleteAll: boolean) => {
        const denseIdx = frames[currentFrameIndex] ? frames[currentFrameIndex].frame_index : 0;
        try {
            await axios.post(`${API_BASE}/delete_leaf`, {
                frame_index: denseIdx,
                leaf_id: leafId,
                delete_all: deleteAll
            });
            await fetchAnnotations();
        } catch (err) {
            console.error("Delete failed", err);
            alert("Failed to delete leaf.");
        }
    };

    // Correction Logic
    const updateLeafPoint = (leafId: number, pointIndex: number, newPos: { x: number, y: number }) => {
        setLeaves(prev => prev.map(l => {
            if (l.id !== leafId) return l;
            const newPoints = [...l.points];
            if (newPoints[pointIndex]) {
                newPoints[pointIndex] = { ...newPoints[pointIndex], ...newPos };
            }

            // Recalculate BBox to ensure inclusion
            const allPts = [...newPoints, ...(l.supportPoints || [])];
            let newBBox = l.bbox;
            if (allPts.length > 0) {
                const xs = allPts.map(p => p.x);
                const ys = allPts.map(p => p.y);
                const minX = Math.min(...xs); const maxX = Math.max(...xs);
                const minY = Math.min(...ys); const maxY = Math.max(...ys);
                const padX = (maxX - minX) * 0.05;
                const padY = (maxY - minY) * 0.05;

                newBBox = {
                    x_min: Math.max(0, minX - padX),
                    y_min: Math.max(0, minY - padY),
                    x_max: maxX + padX,
                    y_max: maxY + padY
                };
            }

            return {
                ...l,
                points: newPoints,
                bbox: newBBox
            };
        }));
    };

    const saveLeafCorrection = async (leafId: number) => {
        const denseIdx = frames[currentFrameIndex] ? frames[currentFrameIndex].frame_index : 0;
        const leaf = leaves.find(l => l.id === leafId);
        if (!leaf) return;

        try {
            const payload = leaves.map(l => ({
                ...l,
                support_points: l.supportPoints,
                mask_polygon: l.maskPolygon
            }));

            await axios.post(`${API_BASE}/save_frame`, {
                frame_index: denseIdx,
                leaves: payload
            });
            console.log(`Leaf ${leafId} correction saved.`);
        } catch (err) {
            console.error("Correction save failed", err);
        }
    };



    const addTempPoint = (pt: Point) => {
        if (!tempLeaf.bbox) return;
        const denseIdx = frames[currentFrameIndex] ? frames[currentFrameIndex].frame_index : 0;

        const currentPoints = tempLeaf.points || [];
        if (currentPoints.length < 2) {
            const newPoints = [...currentPoints, pt];
            setTempLeaf(prev => ({ ...prev, points: newPoints }));

            if (newPoints.length === 2) {
                const completedLeaf: Leaf = {
                    id: tempLeaf.id!,
                    bbox: tempLeaf.bbox!,
                    points: newPoints,
                    supportPoints: tempLeaf.supportPoints || [],
                    maskPolygon: tempLeaf.maskPolygon,
                    color: tempLeaf.color!
                };

                const newLeaves = [...leaves, completedLeaf];
                setLeaves(newLeaves);
                setTempLeaf({});

                const payload = newLeaves.map(l => ({
                    ...l,
                    support_points: l.supportPoints,
                    mask_polygon: l.maskPolygon
                }));

                axios.post(`${API_BASE}/save_frame`, {
                    frame_index: denseIdx,
                    leaves: payload
                })
                    .then(() => {
                        console.log("Leaf saved to backend.");
                        return fetchAnnotations();
                    })
                    .catch(err => console.error("Failed to save leaf", err));
            }
        }
    };

    const [progress, setProgress] = useState<number>(0);
    const [isPolling, setIsPolling] = useState<boolean>(false);

    // Polling
    useEffect(() => {
        let interval: any;
        if (isPolling) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${API_BASE}/tracking_status`);
                    const { status, progress: currentProgress } = res.data;

                    setProgress(currentProgress);

                    if (status === 'idle' || status === 'error') {
                        setIsPolling(false);
                        setLoading(false);
                        if (status === 'error') {
                            alert("Tracking failed.");
                        } else {
                            await fetchAnnotations();
                            setTrackingStarted(true);
                        }
                    }
                } catch (e) {
                    console.error("Polling error", e);
                }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isPolling, fetchAnnotations]);

    const startTracking = async () => {
        if (leaves.length === 0) {
            alert("Please annotate at least one leaf.");
            return;
        }

        setLoading(true);
        setProgress(0);
        const denseIdx = frames[currentFrameIndex] ? frames[currentFrameIndex].frame_index : 0;
        try {
            const payloadLeaves = leaves.map(l => ({
                ...l,
                support_points: l.supportPoints,
                mask_polygon: l.maskPolygon
            }));

            await axios.post(`${API_BASE}/init_tracking`, {
                frame_index: denseIdx,
                leaves: payloadLeaves
            });
            setIsPolling(true);
        } catch (err) {
            console.error(err);
            alert("Failed to start tracking.");
            setLoading(false);
        }
    };

    const exportDataset = () => {
        window.open(`${API_BASE}/export`, '_blank');
    };

    const exportCSV = () => {
        window.open(`${API_BASE}/export_csv`, '_blank');
    };

    return {
        frames,
        currentFrameIndex,
        setCurrentFrameIndex,
        loading,
        leaves,
        tempLeaf,
        startTracking,
        exportDataset,
        exportCSV,
        trackingStarted,
        trackingStatus: isPolling ? 'running' : 'idle',
        units,
        selectedUnit,
        changeUnit,
        dates,
        selectedDate,
        frequency,
        changeDate,
        changeFrequency,
        isPlaying,
        nextFrame,
        prevFrame,
        togglePlay,
        previewPoints,
        addTempPoint,
        progress,
        deleteLeaf,
        updateLeafPoint,
        saveLeafCorrection,

        truncateFrames, // Exported
        annotations // Exported for Timeline marks
    };
};
