import React, { useRef, useEffect, useState } from 'react';
import { Point, BBox, Leaf } from '../hooks/useAnnotation';

interface ViewerProps {
    imageUrl: string;
    leaves: Leaf[];
    tempLeaf: Partial<Leaf>;
    onBBoxComplete: (bbox: BBox) => void;
    onPointAdd: (pt: Point) => void;

    // V6 Props
    isAnnotationMode: boolean;
    updateLeafPoint: (leafId: number, pointIndex: number, newPos: { x: number, y: number }) => void;
    saveLeafCorrection: (leafId: number) => void;
    regenerateSupportPoints: (leafId: number) => void;
    opacity: number;
}

export const Viewer: React.FC<ViewerProps> = ({
    imageUrl, leaves, tempLeaf, onBBoxComplete, onPointAdd,
    isAnnotationMode, updateLeafPoint, saveLeafCorrection, regenerateSupportPoints,
    opacity
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const [isDrawingBBox, setIsDrawingBBox] = useState(false);
    const [startPos, setStartPos] = useState<{ x: number, y: number } | null>(null);
    const [currentBBox, setCurrentBBox] = useState<BBox | null>(null);
    const [displayedUrl, setDisplayedUrl] = useState<string>(imageUrl);
    const [imgDimensions, setImgDimensions] = useState<{ width: number, height: number } | null>(null);

    // Dragging State
    const [draggingPoint, setDraggingPoint] = useState<{ leafId: number, pointIndex: number } | null>(null);

    // Buffering Logic
    useEffect(() => {
        if (!imageUrl) return;
        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            setDisplayedUrl(imageUrl);
            setImgDimensions(null); // Trigger resize check
        };
    }, [imageUrl]);

    // Redraw
    useEffect(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Helper to draw a leaf
        const drawLeaf = (leaf: Partial<Leaf>, isTemp: boolean = false) => {
            const color = leaf.color || '#FFFF00';

            // Draw Mask Polygon
            if (leaf.maskPolygon && leaf.maskPolygon.length > 0) {
                ctx.beginPath();
                leaf.maskPolygon.forEach((p, i) => {
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                });
                ctx.closePath();
                ctx.globalAlpha = opacity * 0.5; // Mask slightly more transparent
                ctx.fillStyle = color;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            // Draw BBox
            if (leaf.bbox) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(leaf.bbox.x_min, leaf.bbox.y_min, leaf.bbox.x_max - leaf.bbox.x_min, leaf.bbox.y_max - leaf.bbox.y_min);

                if (!isTemp || leaf.id !== undefined) {
                    ctx.fillStyle = color;
                    ctx.font = 'bold 16px Arial';
                    ctx.fillText(`Leaf ${leaf.id ?? '?'}`, leaf.bbox.x_min, leaf.bbox.y_min - 5);
                }
            }

            // Draw Points (Base, Tip)
            if (leaf.points) {
                // Connection
                if (leaf.points.length === 2) {
                    ctx.beginPath();
                    ctx.moveTo(leaf.points[0].x, leaf.points[0].y);
                    ctx.lineTo(leaf.points[1].x, leaf.points[1].y);
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                leaf.points.forEach((p, idx) => {
                    ctx.beginPath();
                    // Larger radius if annotating to easier click
                    const r = isAnnotationMode ? 8 : 5;
                    ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
                    ctx.fillStyle = p.id === 0 ? 'red' : 'blue';
                    ctx.fill();

                    if (isAnnotationMode) {
                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }

                    ctx.fillStyle = 'white';
                    ctx.font = '12px Arial';
                    ctx.fillText(p.id === 0 ? "Base" : "Tip", p.x + 10, p.y + 4);
                });
            }

            // Draw Support Points
            if (leaf.supportPoints) {
                ctx.globalAlpha = opacity;
                leaf.supportPoints.forEach(p => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                });
                ctx.globalAlpha = 1.0;
            }
        };

        // Draw Completed Leaves
        leaves.forEach(l => drawLeaf(l));

        // Draw Temp Leaf (in progress)
        if (currentBBox) {
            ctx.strokeStyle = '#FFFF00';
            ctx.strokeRect(currentBBox.x_min, currentBBox.y_min, currentBBox.x_max - currentBBox.x_min, currentBBox.y_max - currentBBox.y_min);
        } else {
            drawLeaf(tempLeaf, true);
        }

    }, [leaves, tempLeaf, currentBBox, displayedUrl, imgDimensions, isAnnotationMode, opacity]);

    const getMousePos = (e: React.MouseEvent) => {
        const rect = canvasRef.current!.getBoundingClientRect();
        const scaleX = canvasRef.current!.width / rect.width;
        const scaleY = canvasRef.current!.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        const pos = getMousePos(e);

        if (isAnnotationMode) {
            // Check for hit on existing points
            for (const leaf of leaves) {
                if (leaf.points) {
                    for (let i = 0; i < leaf.points.length; i++) {
                        const p = leaf.points[i];
                        const dist = Math.sqrt((p.x - pos.x) ** 2 + (p.y - pos.y) ** 2);
                        if (dist < 15) { // Hit Radius
                            setDraggingPoint({ leafId: leaf.id, pointIndex: i });
                            return; // Start Drag
                        }
                    }
                }
            }
        }

        // If not hitting point, and valid mode for drawing
        if (isAnnotationMode && !tempLeaf.bbox) {
            setIsDrawingBBox(true);
            setStartPos(pos);
            setCurrentBBox(null);
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        const currentPos = getMousePos(e);

        if (draggingPoint) {
            // Update Point Position
            updateLeafPoint(draggingPoint.leafId, draggingPoint.pointIndex, currentPos);
            return;
        }

        // Cursor Logic for Keypoints
        if (isAnnotationMode && canvasRef.current) {
            let hovering = false;
            for (const leaf of leaves) {
                if (leaf.points) {
                    for (const p of leaf.points) {
                        const dist = Math.sqrt((p.x - currentPos.x) ** 2 + (p.y - currentPos.y) ** 2);
                        if (dist < 10) { // Hover Radius
                            hovering = true;
                            break;
                        }
                    }
                }
                if (hovering) break;
            }
            canvasRef.current.style.cursor = hovering ? 'pointer' : 'crosshair';
        }

        if (isDrawingBBox && startPos) {
            const newBBox = {
                x_min: Math.min(startPos.x, currentPos.x),
                y_min: Math.min(startPos.y, currentPos.y),
                x_max: Math.max(startPos.x, currentPos.x),
                y_max: Math.max(startPos.y, currentPos.y)
            };
            setCurrentBBox(newBBox);
        }
    };

    const handleMouseUp = (e: React.MouseEvent) => {
        if (draggingPoint) {
            // End Drag
            saveLeafCorrection(draggingPoint.leafId);
            setDraggingPoint(null);
            return;
        }

        if (isDrawingBBox && startPos) {
            setIsDrawingBBox(false);
            if (currentBBox) {
                onBBoxComplete(currentBBox);
            }
            setCurrentBBox(null);
            setStartPos(null);
        } else {
            // Click to Add Point
            if (isAnnotationMode && tempLeaf.bbox) {
                const pos = getMousePos(e);
                const currentLen = tempLeaf.points?.length || 0;
                if (currentLen < 2) {
                    onPointAdd({ ...pos, id: currentLen });
                }
            }
        }
    };

    const handleContextMenu = (e: React.MouseEvent) => {
        e.preventDefault();
        if (!isAnnotationMode) return;

        const pos = getMousePos(e);
        // Check fit on any leaf to regenerate
        for (const leaf of leaves) {
            // Simple distance check to BBox center or points?
            // Let's check if inside BBox
            if (leaf.bbox) {
                if (pos.x >= leaf.bbox.x_min && pos.x <= leaf.bbox.x_max &&
                    pos.y >= leaf.bbox.y_min && pos.y <= leaf.bbox.y_max) {
                    if (confirm(`Regenerate support points for Leaf ${leaf.id}?`)) {
                        regenerateSupportPoints(leaf.id);
                    }
                    return;
                }
            }
        }
    };

    return (
        <div className="relative border border-gray-700 bg-black inline-flex">
            <img
                ref={imgRef}
                src={displayedUrl}
                className="block max-w-full h-auto"
                alt="Frame"
                onLoad={() => {
                    if (canvasRef.current && imgRef.current) {
                        const w = imgRef.current.naturalWidth;
                        const h = imgRef.current.naturalHeight;
                        canvasRef.current.width = w;
                        canvasRef.current.height = h;
                        setImgDimensions({ width: w, height: h });
                    }
                }}
            />
            <canvas
                ref={canvasRef}
                className={`absolute top-0 left-0 w-full h-full ${isAnnotationMode ? 'cursor-crosshair' : 'cursor-default'}`}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onContextMenu={handleContextMenu}
            />
        </div>
    );
};
