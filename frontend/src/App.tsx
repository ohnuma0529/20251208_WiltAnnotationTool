import React, { useState, useEffect } from 'react';
import { Viewer } from './components/Viewer';
import { Controls } from './components/Controls';
import { Timeline } from './components/Timeline';

import { useAnnotation } from './hooks/useAnnotation';

function App() {
    const [opacity, setOpacity] = useState<number>(0.6); // Default 60%
    const {
        frames,
        currentFrameIndex,
        setCurrentFrameIndex,
        loading,
        leaves,
        tempLeaf,
        addTempPoint,
        startTracking,
        exportYolo,
        exportCSV,
        trackingStarted,
        previewPoints,
        dates,
        selectedDate,
        changeDate,
        frequency,
        changeFrequency,
        isPlaying,
        nextFrame,
        prevFrame,
        togglePlay,
        progress,
        deleteLeaf,
        units,
        selectedUnit,
        changeUnit,
        // V6 Props
        updateLeafPoint,
        saveLeafCorrection,
        regenerateSupportPoints,
        annotatedIndices,
        deleteFutureFrames,
        deleteAllLeaves
    } = useAnnotation();

    // V6 State
    // const [isAnnotationMode, setAnnotationMode] = useState<boolean>(true); // Removed - Always ON

    // Derived state
    const currentFrame = frames[currentFrameIndex];
    const imageUrl = currentFrame ? `/images/${currentFrame.filename}` : '';

    const handleTrack = () => {
        startTracking();
    };

    return (
        <div className="h-screen flex flex-col bg-gray-900 text-white">
            {/* Header */}
            <header className="bg-gray-800 p-4 border-b border-gray-700">
                <h1 className="text-2xl font-bold text-green-500">Tomato Wilt Annotation Tool</h1>
            </header>

            {/* Main Content */}
            <div className="flex-grow flex overflow-hidden">
                {/* Canvas Area */}
                <div className="flex-grow flex items-center justify-center p-4 bg-gray-900 relative">
                    {loading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                            <div className="text-white text-xl">Processing...</div>
                        </div>
                    )}
                    <div className="flex-1 flex flex-col relative items-center justify-center">
                        {/* Header */}
                        <div className="bg-gray-800 p-2 border-b border-gray-700 flex justify-between items-center text-xs">
                            <div>
                                <span className="text-gray-400">Time:</span> {currentFrameIndex > 0 && frames[currentFrameIndex]?.timestamp}
                                <span className="mx-2">|</span>
                                <span className="text-gray-400">File:</span> {frames[currentFrameIndex]?.filename}
                            </div>
                            <div className="flex gap-2">
                                <span className="text-gray-400">Leaves:</span> {leaves.length}
                            </div>
                        </div>

                        {imageUrl ? (
                            <Viewer
                                imageUrl={imageUrl}
                                leaves={leaves}
                                tempLeaf={tempLeaf}
                                onBBoxComplete={previewPoints}
                                onPointAdd={addTempPoint}
                                isAnnotationMode={true} // Always ON
                                updateLeafPoint={updateLeafPoint}
                                saveLeafCorrection={saveLeafCorrection}
                                regenerateSupportPoints={regenerateSupportPoints}
                                opacity={opacity}
                            />
                        ) : (
                            <div className="text-gray-500">No images loaded</div>
                        )}
                    </div>
                </div>

                {/* Sidebar Controls */}
                <div className="w-80 border-l border-gray-700 bg-gray-800 overflow-y-auto">
                    <Controls
                        onTrack={handleTrack}
                        onExportYolo={exportYolo}
                        onExportCSV={exportCSV}
                        trackingStarted={trackingStarted}
                        loading={loading}
                        dates={dates}
                        selectedDate={selectedDate}
                        changeDate={changeDate}
                        frequency={frequency}
                        changeFrequency={changeFrequency}
                        progress={progress || 0}
                        leaves={leaves}
                        onDeleteLeaf={deleteLeaf}
                        onDeleteAllLeaves={deleteAllLeaves}
                        onDeleteFutureFrames={deleteFutureFrames}
                        units={units}
                        selectedUnit={selectedUnit}
                        changeUnit={changeUnit}
                        opacity={opacity}
                        setOpacity={setOpacity}
                        currentFilename={currentFrame?.filename || 'No Image'}
                        currentTime={currentFrame?.timestamp || '--:--'}
                    />
                </div>
            </div>

            {/* Timeline Footer */}
            <div className="border-t border-gray-700">
                <Timeline
                    min={0}
                    max={frames.length - 1}
                    value={currentFrameIndex}
                    onChange={setCurrentFrameIndex}
                    timestamp={currentFrame ? currentFrame.timestamp : "--:--"}
                    onPrev={prevFrame}
                    onNext={nextFrame}
                    onPlayToggle={togglePlay}
                    isPlaying={isPlaying}
                    annotatedIndices={annotatedIndices}
                />
            </div>
        </div>
    );
}

export default App;
