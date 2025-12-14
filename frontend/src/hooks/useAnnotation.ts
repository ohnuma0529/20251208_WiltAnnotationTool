import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// API Base URL
const API_URL = '/api';

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
    const [frequency, setFrequency] = useState<number>(30);

    // Multi-Leaf State
    const [leaves, setLeaves] = useState<Leaf[]>([]);
    const [tempLeaf, setTempLeaf] = useState<Partial<Leaf>>({});
    const [trackingStarted, setTrackingStarted] = useState<boolean>(false);

    // API & Loading

    // Load Units on mount
    useEffect(() => {
        let isMounted = true;

        const fetchUnits = () => {
            axios.get(`${API_URL}/units`)
                .then(res => {
                    if (!isMounted) return;
                    const availUnits = res.data.units;
                    setUnits(availUnits);
                    if (availUnits.length > 0) {
                        setSelectedUnit(availUnits[0]);
                    }
                })
                .catch(err => console.error("Failed to load units", err));
        };
        fetchUnits();
        return () => { isMounted = false; };
    }, []);

    // Load Dates when Unit changes
    useEffect(() => {
        if (!selectedUnit) return;

        let isMounted = true;
        const fetchDates = () => {
            axios.get(`${API_URL}/dates?unit=${selectedUnit}`)
                .then(res => {
                    if (!isMounted) return;
                    const availableDates = res.data.dates;
                    setDates(availableDates);
                    if (availableDates.length > 0) {
                        setSelectedDate(availableDates[0]);
                    } else {
                        setSelectedDate('');
                        setFrames([]);
                    }
                })
                .catch(err => {
                    if (!isMounted) return;
                    console.error("Failed to load dates, retrying...", err);
                    setTimeout(fetchDates, 3000);
                });
        };
        fetchDates();
        return () => { isMounted = false; };
    }, [selectedUnit]);

    // Annotations Cache
    const [annotations, setAnnotations] = useState<Record<number, any>>({});

    // Load Annotations
    const fetchAnnotations = useCallback(async () => {
        try {
            const res = await axios.get(`${API_URL}/annotations?t=${new Date().getTime()}`);
            setAnnotations(res.data);
            console.log("Fetched annotations:", Object.keys(res.data).length);
        } catch (err) {
            console.error("Failed to fetch annotations", err);
        }
    }, []);

    // Fetch images when unit/date/freq changes
    useEffect(() => {
        if (!selectedUnit || !selectedDate) return;

        setLoading(true);
        axios.post(`${API_URL}/set_filter`, { unit: selectedUnit, date: selectedDate, frequency: frequency })
            .then(() => {
                // After filter is set, get images
                return axios.get(`${API_URL}/images`);
            })
            .then(res => {
                setFrames(res.data);
                setCurrentFrameIndex(0);
                setLeaves([]);
                setTempLeaf({});
                setTrackingStarted(false);
                // Important: Fetch annotations AFTER images are set to ensure sync
                return fetchAnnotations();
            })
            .catch(err => console.error("Failed to load images/annotations", err))
            .finally(() => setLoading(false));
    }, [selectedUnit, selectedDate, frequency, fetchAnnotations]);

    const changeUnit = (unit: string) => setSelectedUnit(unit);
    const changeDate = (date: string) => setSelectedDate(date);
    const changeFrequency = (freq: number) => setFrequency(freq);

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

    // Preload Next Frames ... (Same)
    useEffect(() => {
        if (frames.length === 0) return;
        const PRELOAD_COUNT = 10;
        const baseUrl = API_URL.replace('/api', '');

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
        const ann = annotations[currentFrameIndex];
        if (ann && ann.leaves) {
            // Restore leaves from backend result
            // Note: Backend result leaves might lack 'color'. We need to re-assign or persist color.
            // Ideally color should be deterministic based on ID.
            const restoredLeaves = ann.leaves.map((l: any) => ({
                ...l,
                supportPoints: l.support_points || l.supportPoints || [],
                color: LEAF_COLORS[l.id % LEAF_COLORS.length]
            }));
            setLeaves(restoredLeaves);
        } else {
            // Only clear if no annotation found?
            // If manual drawing has not started?
            setLeaves([]);
        }
        setTempLeaf({});
    }, [currentFrameIndex, annotations]);

    // Initial load handled by chained effect
    // useEffect(() => {
    //     fetchAnnotations();
    // }, [fetchAnnotations]);


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
        // Check for overlap with existing leaves
        let overlappingLeafId: number | null = null;
        let maxIoU = 0;

        // Find leaves essentially "inside" the new BBox
        // Or if the new BBox is a refinement of an existing one.
        const enclosedLeaves = leaves.filter(l => {
            if (!l.bbox) return false;
            // Check if leaf center is inside new BBox or significant IoU
            // Center Check
            const cx = (l.bbox.x_min + l.bbox.x_max) / 2;
            const cy = (l.bbox.y_min + l.bbox.y_max) / 2;
            const isCenterInside = cx >= bboxVal.x_min && cx <= bboxVal.x_max &&
                cy >= bboxVal.y_min && cy <= bboxVal.y_max;

            // Also check IoU
            const iou = calculateIoU(bboxVal, l.bbox);
            return isCenterInside || iou > 0.5;
        });

        if (enclosedLeaves.length === 1) {
            // Exactly one candidate -> Update it
            overlappingLeafId = enclosedLeaves[0].id;
            console.log(`Single Enclosed Leaf Found: ${overlappingLeafId}. Updating...`);
        } else {
            // Fallback to max IoU if no clear single containment, or multiple
            leaves.forEach(l => {
                if (l.bbox) {
                    const iou = calculateIoU(bboxVal, l.bbox);
                    if (iou > 0.3 && iou > maxIoU) {
                        maxIoU = iou;
                        overlappingLeafId = l.id;
                    }
                }
            });
        }

        try {
            const res = await axios.post(`${API_URL}/preview_points`, {
                frame_index: currentFrameIndex,
                bbox: bboxVal
            });

            if (overlappingLeafId !== null) {
                // Update existing leaf
                console.log(`Updating overlapping leaf ${overlappingLeafId} (IoU: ${maxIoU.toFixed(2)})`);

                const updatedLeaves = leaves.map(l => {
                    if (l.id === overlappingLeafId) {
                        return {
                            ...l,
                            bbox: res.data.new_bbox || bboxVal,
                            supportPoints: res.data.points,
                            maskPolygon: res.data.polygon
                            // Keep existing points (Base/Tip) and color
                        };
                    }
                    return l;
                });

                setLeaves(updatedLeaves); // Optimistic

                // Persist
                const payload = updatedLeaves.map(l => ({
                    ...l,
                    support_points: l.supportPoints
                }));
                await axios.post(`${API_URL}/save_frame`, {
                    frame_index: currentFrameIndex,
                    leaves: payload
                });

            } else {
                // Create new Temp Leaf
                const newId = leaves.length > 0 ? Math.max(...leaves.map(l => l.id)) + 1 : 0;
                setTempLeaf({
                    id: newId,
                    bbox: res.data.new_bbox || bboxVal,
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

    const deleteLeaf = async (leafId: number) => {
        if (!selectedUnit || !selectedDate) return;

        try {
            await axios.delete(`${API_URL}/delete_leaf`, {
                data: {
                    unit: selectedUnit,
                    date: selectedDate,
                    leaf_id: leafId,
                    frame_index: currentFrameIndex,
                    delete_global: true // Always global delete
                }
            });
            // Update local state by removing just that leaf to be responsive
            setAnnotations(prev => {
                // Deep copy isn't easy for Record, but we can just invalidate.
                return { ...prev };
            });
            await fetchAnnotations();
        } catch (error) {
            console.error('Error deleting leaf:', error);
        }
    };

    const deleteAllLeaves = async () => {
        if (!selectedUnit || !selectedDate) return;

        setLoading(true);
        try {
            await axios.delete(`${API_URL}/delete_leaf`, {
                data: {
                    unit: selectedUnit,
                    date: selectedDate,
                    frame_index: currentFrameIndex,
                    delete_all: true,
                    delete_global: true
                }
            });
            setAnnotations({});
            setLeaves([]);
            await fetchAnnotations();
        } catch (error) {
            console.error('Error deleting all leaves:', error);
        } finally {
            setLoading(false);
        }
    };

    const deleteFutureFrames = async () => {
        try {
            await axios.post(`${API_URL}/delete_frames`, {
                frame_index: currentFrameIndex
            });

            setLoading(true);
            // Force reload images to reflect truncation
            const res = await axios.get(`${API_URL}/images`);
            setFrames(res.data);

            // Adjust index if necessary
            if (currentFrameIndex > 0) {
                setCurrentFrameIndex(currentFrameIndex - 1);
            } else {
                setCurrentFrameIndex(0);
            }

            // Refresh annotations
            await fetchAnnotations();

            setLoading(false);
            // alert("Future frames deleted."); // Removed per user request
        } catch (err) {
            console.error("Delete frames failed", err);
            alert("Failed to delete frames.");
            setLoading(false);
        }
    };

    // Correction Logic
    const updateLeafPoint = (leafId: number, pointIndex: number, newPos: { x: number, y: number }) => {
        // Update local state without persisting yet (persisted on release or re-track)
        setLeaves(prev => prev.map(l => {
            if (l.id !== leafId) return l;
            const newPoints = [...l.points];
            if (newPoints[pointIndex]) {
                newPoints[pointIndex] = { ...newPoints[pointIndex], ...newPos };
            }

            // Recalculate BBox (Frontend Optimistic)
            const allPoints = [...(l.supportPoints || []), ...newPoints];
            const allXs = allPoints.map(p => p.x);
            const allYs = allPoints.map(p => p.y);

            const minPtX = Math.min(...allXs);
            const minPtY = Math.min(...allYs);
            const maxPtX = Math.max(...allXs);
            const maxPtY = Math.max(...allYs);

            let newBBox = l.bbox;
            let needsResize = true;

            // Containment Check
            if (l.bbox) {
                const isContained = (
                    l.bbox.x_min <= minPtX &&
                    l.bbox.y_min <= minPtY &&
                    l.bbox.x_max >= maxPtX &&
                    l.bbox.y_max >= maxPtY
                );
                if (isContained) {
                    needsResize = false;
                    // Keep existing BBox
                }
            }

            if (needsResize) {
                const pad = 10;
                newBBox = {
                    x_min: Math.max(0, minPtX - pad),
                    y_min: Math.max(0, minPtY - pad),
                    x_max: maxPtX + pad,
                    y_max: maxPtY + pad
                };
            }

            return {
                ...l,
                bbox: newBBox,
                points: newPoints,
            };
        }));
    };

    const saveLeafCorrection = async (leafId: number) => {
        // Save the modified leaf logic to backend (same as new leaf save, but overwrite)
        // Uses current leaves state
        const leaf = leaves.find(l => l.id === leafId);
        if (!leaf) return;

        try {
            // We need to send ALL leaves for this frame to be safe, or just update one?
            // save_frame updates the whole list for that frame index?
            // backend/api/endpoints.py: save_frame -> tracking_results[idx].leaves = req.leaves
            // So we must send ALL leaves.
            const payload = leaves.map(l => ({
                ...l,
                support_points: l.supportPoints
            }));

            await axios.post(`${API_URL}/save_frame`, {
                frame_index: currentFrameIndex,
                leaves: payload
            });
            console.log(`Leaf ${leafId} correction saved.`);
            fetchAnnotations(); // Refresh to update Timeline stars
        } catch (err) {
            console.error("Correction save failed", err);
        }
    };

    const regenerateSupportPoints = async (leafId: number) => {
        const leaf = leaves.find(l => l.id === leafId);
        if (!leaf || !leaf.points || leaf.points.length !== 2) return;

        const xs = leaf.points.map(p => p.x);
        const ys = leaf.points.map(p => p.y);
        const minX = Math.min(...xs); const maxX = Math.max(...xs);
        const minY = Math.min(...ys); const maxY = Math.max(...ys);
        const w = maxX - minX; const h = maxY - minY;
        // Strict Tight Fit (Bitabita) - No Padding
        const newBBox: BBox = {
            x_min: minX,
            y_min: minY,
            x_max: maxX,
            y_max: maxY
        };

        try {
            // Call update_region endpoint (New V43 requirement)
            // Or preview_points logic? 
            // V43 requires update_region to regenerate points.
            // Let's call update_region directly which saves to backend immediately.
            const res = await axios.post(`${API_URL}/update_region`, {
                frame_index: currentFrameIndex,
                leaf_id: leafId,
                bbox: newBBox
            });

            if (res.data.status === 'updated') {
                const updatedLeaf = res.data.leaf;
                // Update local state
                setLeaves(prev => prev.map(l => l.id === leafId ? { ...l, ...updatedLeaf, color: l.color } : l));
                fetchAnnotations(); // Refresh stars
            }
        } catch (err) {
            console.error("Regenerate (Update Region) failed", err);
        }
    };

    const addTempPoint = (pt: Point) => {
        if (!tempLeaf.bbox) return; // Must have bbox first

        const currentPoints = tempLeaf.points || [];
        if (currentPoints.length < 2) {
            const newPoints = [...currentPoints, pt];
            setTempLeaf(prev => ({ ...prev, points: newPoints }));

            // If 2 points, Leaf is complete. Move to main list.
            if (newPoints.length === 2) {

                // Recalculate BBox to include Base/Tip (in case they are outside)
                // We start with the SAM-tightened bbox
                let finalLeafBBox = tempLeaf.bbox!;
                const xs = [newPoints[0].x, newPoints[1].x, finalLeafBBox.x_min, finalLeafBBox.x_max];
                const ys = [newPoints[0].y, newPoints[1].y, finalLeafBBox.y_min, finalLeafBBox.y_max];

                // If Base/Tip are outside, expand. If inside, keep tight SAM bbox.
                // Actually, strict "Bitabita" means MIN/MAX of ALL Points.
                // The SAM points + Base + Tip.
                // But SAM points are in `tempLeaf.supportPoints`.
                // So we should recalculate from ALL points relative to support + main.

                const allLeafPoints = [...(tempLeaf.supportPoints || []), ...newPoints];
                if (allLeafPoints.length > 0) {
                    const allXs = allLeafPoints.map(p => p.x);
                    const allYs = allLeafPoints.map(p => p.y);
                    const pad = 10;
                    finalLeafBBox = {
                        x_min: Math.max(0, Math.min(...allXs) - pad),
                        y_min: Math.max(0, Math.min(...allYs) - pad),
                        x_max: Math.max(...allXs) + pad, // Upper bound not clipped here, safe within Canvas
                        y_max: Math.max(...allYs) + pad
                    };
                }

                const completedLeaf: Leaf = {
                    id: tempLeaf.id!,
                    bbox: finalLeafBBox,
                    points: newPoints,
                    supportPoints: tempLeaf.supportPoints || [],
                    maskPolygon: tempLeaf.maskPolygon,
                    color: tempLeaf.color!
                };

                // Optimistic Update
                const newLeaves = [...leaves, completedLeaf];
                setLeaves(newLeaves);
                setTempLeaf({}); // Reset for next leaf

                // Persist to Backend Immediately
                const payload = newLeaves.map(l => ({
                    ...l,
                    support_points: l.supportPoints
                }));

                axios.post(`${API_URL}/save_frame`, {
                    frame_index: currentFrameIndex,
                    leaves: payload
                })
                    .then(() => {
                        console.log("Leaf saved to backend.");
                        return fetchAnnotations(); // Sync source of truth
                    })
                    .catch(err => console.error("Failed to save leaf", err));
            }
        }
    };

    const [progress, setProgress] = useState<number>(0);
    const [isPolling, setIsPolling] = useState<boolean>(false);

    // Polling Logic
    useEffect(() => {
        let interval: any;
        if (isPolling) {
            interval = setInterval(async () => {
                try {
                    const res = await axios.get(`${API_URL}/tracking_status`);
                    const { status, progress: currentProgress } = res.data;

                    setProgress(currentProgress);

                    if (status === 'idle' || status === 'error') {
                        setIsPolling(false);
                        setLoading(false);
                        if (status === 'error') {
                            alert("Tracking failed (check backend logs).");
                        } else {
                            // Sync completed. Small delay to ensure file write.
                            setTimeout(async () => {
                                await fetchAnnotations();
                                setTrackingStarted(true);
                            }, 500);
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
        try {
            // Map frontend camelCase to backend snake_case
            const payloadLeaves = leaves.map(l => ({
                ...l,
                support_points: l.supportPoints
            }));

            await axios.post(`${API_URL}/init_tracking`, {
                frame_index: currentFrameIndex,
                leaves: payloadLeaves
            });
            // Start polling
            setIsPolling(true);
        } catch (err) {
            console.error(err);
            alert("Failed to start tracking.");
            setLoading(false);
        }
    };

    const exportYolo = () => {
        window.open(`${API_URL}/export_yolo`, '_blank');
    };

    const exportCSV = () => {
        window.open(`${API_URL}/export_csv`, '_blank');
    };

    const annotatedIndices = Object.entries(annotations)
        .filter(([_, ann]: [string, any]) => ann.leaves && ann.leaves.some((l: any) => l.manual))
        .map(([k, _]) => Number(k))
        .sort((a, b) => a - b);

    return {
        frames,
        currentFrameIndex,
        setCurrentFrameIndex,
        loading,
        leaves,
        tempLeaf,
        startTracking,
        exportYolo,
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
        regenerateSupportPoints,
        annotatedIndices,
        deleteFutureFrames,
        deleteAllLeaves
    };
};
