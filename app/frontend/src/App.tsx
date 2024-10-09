import { useState } from "react";
import { Mic, MicOff } from "lucide-react";

import { Button } from "@/components/ui/button";
import { GroundingFiles } from "@/components/ui/grounding-files";
import GroundingFileView from "@/components/ui/grounding-file-view";
import StatusMessage from "@/components/ui/status-message";

import useRealTime from "@/hooks/useRealtime";
import useAudioRecorder from "@/hooks/useAudioRecorder";
import useAudioPlayer from "@/hooks/useAudioPlayer";

import { GroundingFile, ToolResult } from "./types";

import logo from "./assets/logo.svg";

function App() {
    const [isRecording, setIsRecording] = useState(false);
    const [groundingFiles, setGroundingFiles] = useState<GroundingFile[]>([]);
    const [selectedFile, setSelectedFile] = useState<GroundingFile | null>(null);

    const { startSession, addUserAudio, inputAudioBufferClear } = useRealTime({
        onWebSocketOpen: () => console.log("WebSocket connection opened"),
        onWebSocketClose: () => console.log("WebSocket connection closed"),
        onWebSocketError: event => console.error("WebSocket error:", event),
        onReceivedError: message => console.error("error", message),
        onReceivedResponseAudioDelta: message => {
            isRecording && playAudio(message.delta);
        },
        onReceivedInputAudioBufferSpeechStarted: () => {
            stopAudioPlayer();
        },
        onReceivedExtensionMiddleTierToolResponse: message => {
            const result: ToolResult = JSON.parse(message.tool_result);

            const files: GroundingFile[] = result.sources.map(x => {
                const match = x.chunk_id.match(/_pages_(\d+)$/);
                const name = match ? `${x.title}   (Page=${match[1]})` : x.title;
                return { id: x.chunk_id, name: name, content: x.chunk };
            });

            setGroundingFiles(prev => [...prev, ...files]);
        }
    });

    const { reset: resetAudioPlayer, play: playAudio, stop: stopAudioPlayer } = useAudioPlayer();
    const { start: startAudioRecording, stop: stopAudioRecording } = useAudioRecorder({ onAudioRecorded: addUserAudio });

    const onToggleListening = async () => {
        if (!isRecording) {
            startSession();
            await startAudioRecording();
            resetAudioPlayer();

            setIsRecording(true);
        } else {
            await stopAudioRecording();
            stopAudioPlayer();
            inputAudioBufferClear();

            setIsRecording(false);
        }
    };

    return (
        <div className="flex min-h-screen flex-col bg-gray-800 text-gray-100">
            <div className="flex justify-center items-center p-4 sm:absolute sm:left-1/2 sm:transform sm:-translate-x-1/2 sm:top-25">
                <img src={logo} alt="Azure logo" className="h-40 w-36" />
            </div>
            <main className="flex flex-grow flex-col items-center justify-center">
                <h1 className="mb-8 bg-gradient-to-r from-orange-400 via-pink-500 to-purple-700 bg-clip-text text-4xl font-bold text-transparent md:text-7xl">
                    AI Voice Agent
                </h1>
                <div className="mb-4 flex flex-col items-center justify-center">
                <Button
                    onClick={onToggleListening}
                    className={`h-12 w-60 bg-gradient-to-r ${isRecording ? "from-red-500 via-red-600 to-red-700" : "from-pink-500 via-purple-600 to-blue-700"} hover:opacity-70`}
                    aria-label={isRecording ? "Stop recording" : "Start recording"}
                >
                    {isRecording ? (
                        <>
                            <MicOff className="mr-2 h-4 w-4" />
                            Stop conversation
                        </>
                    ) : (
                        <>
                            <Mic className="mr-2 h-6 w-6" />
                            Start conversation
                        </>
                    )}
                </Button>
                    <StatusMessage isRecording={isRecording} />
                </div>
                <GroundingFiles files={groundingFiles} onSelected={setSelectedFile} />
            </main>

            <footer className="py-4 text-center">
                <p>Sky Prototype</p>
            </footer>
            <GroundingFileView groundingFile={selectedFile} onClosed={() => setSelectedFile(null)} />
        </div>
    );
}

export default App;
