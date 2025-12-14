import React from 'react';
import { Leaf } from '../hooks/useAnnotation';

export interface ControlsProps {
    opacity: number;
    setOpacity: (val: number) => void;
    onTrack: () => void;
    onExportYolo: () => void;
    onExportCSV: () => void;

    trackingStarted: boolean;
    loading: boolean;
    leaves: Leaf[];
    onDeleteLeaf: (id: number) => void;
    onDeleteAllLeaves: () => void;
    onDeleteFutureFrames: () => void;    // New props
    dates: string[];
    selectedDate: string;

    changeDate: (d: string) => void;
    frequency: number;
    changeFrequency: (f: number) => void;
    progress: number;
    units: string[];
    selectedUnit: string;
    changeUnit: (u: string) => void;

    // Info Props
    currentFilename: string;
    currentTime: string;
}

export const Controls: React.FC<ControlsProps> = ({
    onTrack, onExportYolo, onExportCSV, trackingStarted, loading, leaves, onDeleteLeaf, onDeleteAllLeaves,
    dates, selectedDate, changeDate, frequency, changeFrequency, progress,
    units, selectedUnit, changeUnit,
    opacity, setOpacity, onDeleteFutureFrames,
    currentFilename, currentTime
}) => {

    // Calculate stats for Info section
    const leafCount = leaves.length;
    const supportPointCount = leaves.reduce((acc, l) => acc + (l.supportPoints?.length || 0), 0);

    return (
        <div className="flex flex-col gap-4 p-4 bg-gray-800 text-white rounded-lg h-full">

            {/* Info Section (Moved to Top & Merged) */}
            <div className="bg-gray-700 p-3 rounded-lg space-y-2">
                <h2 className="text-lg font-bold text-blue-400">Info</h2>

                <div className="text-sm space-y-1 border-b border-gray-600 pb-2">
                    <div className="flex justify-between">
                        <span className="text-gray-400">File:</span>
                        <span className="font-bold truncate ml-2" title={currentFilename}>{currentFilename}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Time:</span>
                        <span className="font-bold">{currentTime}</span>
                    </div>
                </div>

                <div className="text-sm space-y-1">
                    <div className="flex justify-between">
                        <span className="text-gray-400">Leaves:</span>
                        <span className="font-bold">{leafCount}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-gray-400">Support Pts:</span>
                        <span className="font-bold">{supportPointCount}</span>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-600 my-1"></div>

            {/* Opacity Slider */}
            <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-400 flex justify-between">
                    Opacity <span>{Math.round(opacity * 100)}%</span>
                </label>
                <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            {/* Data Selection */}
            <div>
                <h2 className="text-xl font-bold mb-2">Data Selection</h2>
                <div className="flex flex-col gap-2">
                    <div>
                        <label className="block text-sm text-gray-400">Unit</label>
                        <select
                            value={selectedUnit}
                            onChange={(e) => changeUnit(e.target.value)}
                            className="w-full bg-gray-700 p-2 rounded text-white"
                        >
                            {units.map(u => <option key={u} value={u}>{u}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400">Date</label>
                        <select
                            value={selectedDate}
                            onChange={(e) => changeDate(e.target.value)}
                            className="w-full bg-gray-700 p-2 rounded text-white"
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400">Frequency (min)</label>
                        <select
                            value={frequency}
                            onChange={(e) => changeFrequency(Number(e.target.value))}
                            className="w-full bg-gray-700 p-2 rounded text-white"
                        >
                            <option value={1}>1 min</option>
                            <option value={5}>5 min</option>
                            <option value={10}>10 min</option>
                            <option value={30}>30 min</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-600 my-2"></div>

            {/* Tracking & Export */}
            <h2 className="text-xl font-bold">Tracking</h2>

            <div className="flex gap-2 flex-col">
                <div className="flex gap-2">
                    <button
                        onClick={onTrack}
                        disabled={loading}
                        className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 p-2 rounded font-bold"
                    >
                        {loading ? `Tracking... ${progress.toFixed(0)}% ` : 'Run Tracking'}
                    </button>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={onExportYolo}
                        disabled={loading}
                        className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 p-2 rounded font-bold"
                    >
                        Export YOLO
                    </button>
                    <button
                        onClick={onExportCSV}
                        disabled={loading}
                        className="flex-1 bg-teal-600 hover:bg-teal-500 disabled:bg-gray-600 p-2 rounded font-bold"
                    >
                        Export CSV
                    </button>
                </div>
                {loading && (
                    <div className="w-full bg-gray-700 rounded-full h-2.5 dark:bg-gray-700">
                        <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                    </div>
                )}
            </div>

            <div className="border-t border-gray-600 my-2"></div>

            {/* Delete Annotation */}
            <div className="mt-2">
                <label className="text-xs text-gray-400 block mb-1">Delete Annotation (Global)</label>
                <select
                    className="w-full bg-gray-700 text-white p-2 rounded"
                    onChange={(e) => {
                        const val = e.target.value;
                        if (val === "all") {
                            onDeleteAllLeaves();
                        } else if (val !== "") {
                            onDeleteLeaf(parseInt(val));
                        }
                        e.target.value = ""; // Reset
                    }}
                    defaultValue=""
                >
                    <option value="" disabled>Select to delete...</option>
                    <option value="all" className="text-red-400 font-bold">All Leaves (Global)</option>
                    {leaves.map(l => (
                        <option key={l.id} value={l.id}>Leaf {l.id}</option>
                    ))}
                </select>
                <div className="text-xs text-gray-400 mt-1">
                    * Deletes from ALL frames.
                </div>
            </div>

            <div className="mt-auto border-t border-gray-700 pt-4">
                {/* Delete Future Images Button */}
                <button
                    onClick={() => onDeleteFutureFrames()}
                    className="w-full bg-red-900 hover:bg-red-800 text-white font-bold py-2 px-4 rounded mb-4 border border-red-700 text-xs"
                >
                    DELETE FUTURE IMAGES (Files)
                </button>
            </div>
        </div >
    );
};
