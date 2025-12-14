import React from 'react';

interface TimelineProps {
    min: number;
    max: number;
    value: number;
    onChange: (val: number) => void;
    timestamp: string;
    // New props
    onPrev: () => void;
    onNext: () => void;
    onPlayToggle: () => void;
    isPlaying: boolean;
    annotatedIndices: number[];
}

export const Timeline: React.FC<TimelineProps> = ({
    min,
    max,
    value,
    onChange,
    timestamp,
    onPrev,
    onNext,
    onPlayToggle,
    isPlaying,
    annotatedIndices
}) => {
    return (
        <div className="flex items-center gap-4 p-4 bg-gray-900 text-white w-full">
            <div className="flex gap-2">
                <button onClick={onPrev} className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">
                    &lt;&lt;
                </button>
                <button onClick={onPlayToggle} className="px-3 py-1 bg-green-700 rounded hover:bg-green-600 w-16 text-center">
                    {isPlaying ? "Stop" : "Play"}
                </button>
                <button onClick={onNext} className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">
                    &gt;&gt;
                </button>
            </div>

            <div className="font-mono text-xl w-16 text-center">{timestamp}</div>
            <div className="relative flex-grow h-6 flex items-center">
                <input
                    type="range"
                    min={min}
                    max={max}
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer z-10"
                />
                {annotatedIndices.map(idx => (
                    <div
                        key={idx}
                        className="absolute text-yellow-400 pointer-events-none"
                        style={{ left: `${(idx / (max || 1)) * 100}%`, top: '-10px', fontSize: '10px' }}
                    >
                        ★
                    </div>
                ))}
            </div>
            <div className="font-mono text-sm text-gray-400 w-24 text-right">
                Frame: {value}
            </div>
        </div>
    );
};
