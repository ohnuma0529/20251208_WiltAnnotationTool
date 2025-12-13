import React from 'react';
import { Leaf } from '../hooks/useAnnotation';

interface ControlsProps {
    onTrack: () => void;
    onExport: () => void;
    onExportCSV: () => void;
    // V6 Props
    isAnnotationMode: boolean;
    setAnnotationMode: (mode: boolean) => void;

    trackingStarted: boolean;
    loading: boolean;
    leaves: Leaf[];
    onDeleteLeaf: (id: number, deleteAll: boolean) => void;
    // New props
    dates: string[];
    selectedDate: string;
    changeDate: (d: string) => void;
    frequency: number;
    changeFrequency: (f: number) => void;
    progress: number;
    units: string[];
    selectedUnit: string;
    changeUnit: (u: string) => void;
    // Visibility
    visibility: { bbox: boolean; keypoints: boolean; supportPoints: boolean; mask: boolean; };
    opacity: { bbox: number; keypoints: number; supportPoints: number; mask: number; };
    setVisibility: (v: { bbox: boolean; keypoints: boolean; supportPoints: boolean; mask: boolean; }) => void;
    setOpacity: (o: { bbox: number; keypoints: number; supportPoints: number; mask: number; }) => void;

    // V11
    truncateFrames: () => void;
}

export const Controls: React.FC<ControlsProps> = ({
    onTrack, onExport, onExportCSV, isAnnotationMode, setAnnotationMode, trackingStarted, loading, leaves, onDeleteLeaf,
    dates, selectedDate, changeDate, frequency, changeFrequency, progress,
    units, selectedUnit, changeUnit,
    visibility, opacity, setVisibility, setOpacity,
    truncateFrames
}) => {
    return (
        <div className="w-64 bg-gray-200 p-2 flex flex-col h-full text-xs">
            <h1 className="text-base font-bold mb-2 text-gray-700">Setting</h1>

            {/* Display Settings */}
            {/* Display Settings */}
            <div className="bg-white p-2 rounded shadow mb-2">
                <h3 className="font-semibold text-xs mb-1 text-gray-600">Display Settings</h3>
                {[
                    { key: 'bbox', label: 'BBox' },
                    { key: 'keypoints', label: 'Keypoints' },
                    { key: 'supportPoints', label: 'Support Pts' }
                ].map(({ key, label }) => (
                    <div key={key} className="flex items-center justify-between mb-2 text-sm">
                        <label className="flex items-center cursor-pointer select-none text-gray-700">
                            <input
                                type="checkbox"
                                checked={(visibility as any)[key]}
                                onChange={e => setVisibility({ ...visibility, [key]: e.target.checked })}
                                className="mr-2 h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
                            />
                            {label}
                        </label>
                        <input
                            type="range" min="0" max="1" step="0.1"
                            value={(opacity as any)[key]}
                            onChange={e => setOpacity({ ...opacity, [key]: parseFloat(e.target.value) })}
                            className="w-16 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>
                ))}
            </div>

            {/* Filter Section */}
            {/* Filter Section */}
            <h2 className="text-sm font-bold mb-1 text-gray-700">Data Selection</h2>
            <div className="flex flex-col gap-2 mb-4">
                <div>
                    <label className="block text-sm text-gray-500">Unit</label>
                    <select
                        value={selectedUnit}
                        onChange={(e) => changeUnit(e.target.value)}
                        className="w-full bg-white p-2 rounded text-gray-800 border"
                    >
                        {units.map(u => <option key={u} value={u}>{u}</option>)}
                    </select>
                </div>
                <div>
                    <label className="block text-sm text-gray-500">Date</label>
                    <select
                        value={selectedDate}
                        onChange={(e) => changeDate(e.target.value)}
                        className="w-full bg-white p-2 rounded text-gray-800 border"
                    >
                        {dates.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                </div>
                <div>
                    <label className="block text-sm text-gray-500">Display Freq (min)</label>
                    <select
                        value={frequency}
                        onChange={(e) => changeFrequency(parseInt(e.target.value))}
                        className="w-full bg-white p-2 rounded text-gray-800 border"
                    >
                        {[1, 5, 10, 30].map(f => <option key={f} value={f}>{f}m</option>)}
                    </select>
                </div>
            </div>

            <div className="border-t border-gray-400 my-2"></div>

            <h2 className="text-sm font-bold mb-1 text-gray-700">Annotation</h2>

            <div className="flex items-center gap-2 mb-2">
                <button
                    onClick={() => setAnnotationMode(!isAnnotationMode)}
                    className={`w-full p-2 rounded font-bold text-white shadow ${isAnnotationMode ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-gray-600 hover:bg-gray-500'}`}
                >
                    {isAnnotationMode ? 'Mode: ON' : 'Mode: OFF'}
                </button>
            </div>
            <p className="text-xs text-gray-500 mb-2">
                {isAnnotationMode
                    ? "Drag Base/Tip to correct. Draw BBox for new leaves."
                    : "View only."}
            </p>

            <div className="border-t border-gray-400 my-2"></div>

            <h2 className="text-sm font-bold mb-1 text-gray-700">Tracking</h2>

            <div className="flex gap-2 flex-col">
                <div className="flex gap-2">
                    <button
                        onClick={onTrack}
                        disabled={loading}
                        className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-400 text-white p-2 rounded font-bold shadow"
                    >
                        {loading ? `${progress.toFixed(0)}%` : 'Track'}
                    </button>
                    <button
                        onClick={onExport}
                        disabled={loading}
                        className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-gray-400 text-white p-2 rounded font-bold shadow"
                    >
                        Export
                    </button>
                    <button
                        onClick={onExportCSV}
                        disabled={loading}
                        className="flex-1 bg-teal-600 hover:bg-teal-500 disabled:bg-gray-400 text-white p-2 rounded font-bold shadow"
                    >
                        CSV
                    </button>
                </div>
                {loading && (
                    <div className="w-full bg-gray-300 rounded-full h-2.5 mt-2">
                        <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
                    </div>
                )}
            </div>

            <div className="border-t border-gray-400 my-4"></div>

            {/* Delete Dropdown */}
            <div>
                <label className="text-xs text-gray-500 block mb-1">Delete Leaf</label>
                <select
                    className="w-full bg-white text-gray-800 p-2 rounded border"
                    onChange={(e) => {
                        const val = e.target.value;
                        if (val === "all") {
                            onDeleteLeaf(0, true);
                        }
                        else if (val !== "") {
                            onDeleteLeaf(parseInt(val), false);
                        }
                        e.target.value = ""; // Reset
                    }}
                    defaultValue=""
                >
                    <option value="" disabled>Select to delete...</option>
                    <option value="all">All Leaves</option>
                    {leaves.map(l => (
                        <option key={l.id} value={l.id}>Leaf {l.id}</option>
                    ))}
                </select>
                <p className="text-xs text-red-500 mt-1">*Deletes immediately (no confirm)</p>
            </div>

            <div className="border-t border-gray-400 my-4"></div>

            <div className="bg-red-100 p-2 rounded border border-red-300">
                <h3 className="text-red-700 font-bold text-xs mb-2">Danger Zone</h3>
                <button
                    onClick={() => {
                        if (confirm("DELETE all subsequent frames (on disk/cache) starting from current? This cannot be undone.")) {
                            truncateFrames();
                        }
                    }}
                    className="w-full bg-red-600 hover:bg-red-500 text-white p-2 rounded text-xs font-bold shadow"
                >
                    Truncate Future Frames
                </button>
            </div>
        </div>
    );
};
