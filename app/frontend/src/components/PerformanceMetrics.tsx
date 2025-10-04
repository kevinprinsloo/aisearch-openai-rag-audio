import { Activity } from "lucide-react";

interface PerformanceMetricsProps {
    transcriptionMs?: number;
    llmMs?: number;
    ttsMs?: number;
    totalMs?: number;
}

export const PerformanceMetrics = ({ transcriptionMs, llmMs, ttsMs, totalMs }: PerformanceMetricsProps) => {
    const hasMetrics = transcriptionMs !== undefined || llmMs !== undefined || ttsMs !== undefined;

    if (!hasMetrics) {
        return null;
    }

    const formatTime = (ms: number | undefined) => {
        if (ms === undefined) return "-";
        if (ms < 1000) return `${ms.toFixed(0)}ms`;
        return `${(ms / 1000).toFixed(2)}s`;
    };

    const getLatencyColor = (ms: number | undefined) => {
        if (ms === undefined) return "text-gray-500";
        if (ms < 1500) return "text-yellow-600";
        return "text-red-600";
    };

    return (
        <div className="fixed top-4 left-4 rounded-xl border border-gray-300/50 bg-white/80 backdrop-blur-md p-4 shadow-lg min-w-[280px] z-50">
            <div className="mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4 text-blue-600" />
                <h3 className="text-sm font-semibold text-gray-700">Performance Metrics</h3>
            </div>
            
            <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                    <span className="text-gray-600">Transcription:</span>
                    <span className={`font-mono font-semibold ${getLatencyColor(transcriptionMs)}`}>
                        {formatTime(transcriptionMs)}
                    </span>
                </div>
                
                <div className="flex items-center justify-between">
                    <span className="text-gray-600">LLM Response:</span>
                    <span className={`font-mono font-semibold ${getLatencyColor(llmMs)}`}>
                        {formatTime(llmMs)}
                    </span>
                </div>
                
                <div className="flex items-center justify-between">
                    <span className="text-gray-600">TTS Onset:</span>
                    <span className={`font-mono font-semibold ${getLatencyColor(ttsMs)}`}>
                        {formatTime(ttsMs)}
                    </span>
                </div>
                
                <div className="mt-3 border-t border-gray-200 pt-2">
                    <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-700">Total:</span>
                        <span className={`font-mono font-bold ${getLatencyColor(totalMs)}`}>
                            {formatTime(totalMs)}
                        </span>
                    </div>
                </div>
            </div>
            
            <div className="mt-3 text-xs text-gray-500">
                <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                        <div className="h-2 w-2 rounded-full bg-green-500"></div>
                        &lt;500ms
                    </span>
                    <span className="flex items-center gap-1">
                        <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
                        &lt;1.5s
                    </span>
                    <span className="flex items-center gap-1">
                        <div className="h-2 w-2 rounded-full bg-red-500"></div>
                        &gt;1.5s
                    </span>
                </div>
            </div>
        </div>
    );
};
