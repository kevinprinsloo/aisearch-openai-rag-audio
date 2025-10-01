import { useState, useRef } from "react";
import { Mic, MicOff } from "lucide-react";
import { useTranslation } from "react-i18next";

import "./styles.css";
import AnimatedCircle from "./AnimatedCircle";
import useAudioAmplitude from "./hooks/useAudioAmplitude";
import useAgentAudioAmplitude from "./hooks/useAgentAudioAmplitude";

import { Button } from "@/components/ui/button";
import { GroundingFiles } from "@/components/ui/grounding-files";
import GroundingFileView from "@/components/ui/grounding-file-view";
import StatusMessage from "@/components/ui/status-message";

import useRealTime from "@/hooks/useRealtime";
import useAudioRecorder from "@/hooks/useAudioRecorder";
import useAudioPlayer from "@/hooks/useAudioPlayer";

import {
  GroundingFile,
  ToolResult,
  ResponseInputAudioTranscriptionCompleted,
  ResponseAudioTranscriptDelta,
} from "./types";

import logo from "./assets/logo.svg";
import { useNavigate } from "react-router-dom";

type TranscriptEntry = {
  speaker: "Agent" | "Customer";
  text: string;
  timestamp: string;
};

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [groundingFiles, setGroundingFiles] = useState<GroundingFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<GroundingFile | null>(null);
  const [transcriptEntries, setTranscriptEntries] = useState<TranscriptEntry[]>(
    []
  );
  const [agentTranscriptBuffer, setAgentTranscriptBuffer] =
    useState<string>("");

  const [loadingAnalytics, setLoadingAnalytics] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<string>("");

  const [showUploadModal, setShowUploadModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const navigate = useNavigate();
  const { t } = useTranslation();

  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);

  const addTranscriptEntry = (speaker: "Agent" | "Customer", text: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setTranscriptEntries((prev) => [...prev, { speaker, text, timestamp }]);
  };

  const onReceivedInputAudioTranscriptionCompleted = (
    message: ResponseInputAudioTranscriptionCompleted
  ) => {
    // Customer speech is provided here. Append as a single entry.
    if (message.transcript && message.transcript.trim().length > 0) {
      addTranscriptEntry("Customer", message.transcript);
    }
  };

  const onReceivedResponseAudioTranscriptDelta = (
    message: ResponseAudioTranscriptDelta
  ) => {
    if (message.delta && message.delta.trim().length > 0) {
      setAgentTranscriptBuffer((prev) =>
        prev ? prev + " " + message.delta : message.delta
      );
    }
    // If we are receiving agent audio transcript deltas, the agent is "speaking"
    if (isRecording && message.delta && message.delta.trim().length > 0) {
      setIsAgentSpeaking(true);
    }
  };

  const onReceivedResponseDone = () => {
    // Agent final transcript should now be stable. Add the buffered transcript as one entry if it's not empty.
    if (agentTranscriptBuffer.trim().length > 0) {
      addTranscriptEntry("Agent", agentTranscriptBuffer.trim());
    }
    // Clear the buffer for next turn.
    setAgentTranscriptBuffer("");
  };

  const { startSession, addUserAudio, inputAudioBufferClear } = useRealTime({
    enableInputAudioTranscription: true, // ensure we get customer transcripts
    onWebSocketOpen: () => console.log("WebSocket opened"),
    onWebSocketClose: () => console.log("WebSocket closed"),
    onWebSocketError: (event) => console.error("WebSocket error:", event),
    onReceivedError: (message) => console.error("error", message),
    onReceivedResponseAudioDelta: (message) => {
      // Play agent audio, but don't add transcript here.
      isRecording && playAudio(message.delta);
    },
    onReceivedInputAudioBufferSpeechStarted: () => {
      stopAudioPlayer();
    },
    onReceivedExtensionMiddleTierToolResponse: (message) => {
      const result: ToolResult = JSON.parse(message.tool_result);
      const files: GroundingFile[] = result.sources.map((x) => {
        return { id: x.chunk_id, name: x.title, content: x.chunk };
      });
      setGroundingFiles((prev) => [...prev, ...files]);
    },
    onReceivedInputAudioTranscriptionCompleted,
    onReceivedResponseDone,
    onReceivedResponseAudioTranscriptDelta,
  });

  const {
    reset: resetAudioPlayer,
    play: playAudio,
    stop: stopAudioPlayer,
    getAnalyser,
  } = useAudioPlayer();
  const { start: startAudioRecording, stop: stopAudioRecording } =
    useAudioRecorder({ onAudioRecorded: addUserAudio });

  const onToggleListening = async () => {
    if (!isRecording) {
      startSession();
      await startAudioRecording();
      await resetAudioPlayer(); // ensure audio player is initialized before playing
      setIsRecording(true);
    } else {
      await stopAudioRecording();
      stopAudioPlayer();
      inputAudioBufferClear();
      setIsRecording(false);
      setIsAgentSpeaking(false); // stop agent amplitude once call ends
      fetchAnalysis();
    }
  };

  const fetchAnalysis = async () => {
    setLoadingAnalytics(true);
    try {
      const response = await fetch("/api/analyzeTranscript", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: transcriptEntries }),
      });
      const data = await response.json();
      setAnalysisResult(data.analysis);
    } catch (e) {
      console.error("Failed to fetch analysis:", e);
    } finally {
      setLoadingAnalytics(false);
    }
  };

  const showTranscriptButton = !isRecording && transcriptEntries.length > 0;

  const onViewTranscriptClicked = () => {
    navigate("/callAnalytics", {
      state: { transcriptEntries, analysisResult, loadingAnalytics },
    });
  };

  // Synthetic call upload flow
  const onUploadSyntheticCall = () => {
    setShowUploadModal(true);
  };

  // Get user amplitude (microphone) and agent amplitude (played audio)
  const userAmplitude = useAudioAmplitude(isRecording);
  const agentAmplitude = useAgentAudioAmplitude(isAgentSpeaking, getAnalyser);

  // Decide which amplitude to show
  // For a simple approach, show the louder of the two:
  const currentAmplitude = Math.max(userAmplitude, agentAmplitude);

  const onFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const syntheticTranscript = JSON.parse(text);

      // Assume syntheticTranscript is in the correct format (array of { speaker, text, timestamp })
      // Set these as the transcriptEntries
      setTranscriptEntries(syntheticTranscript);

      // Simulate the end of a call by fetching analysis immediately
      await fetchAnalysis();
    } catch (err) {
      console.error("Failed to parse the transcript JSON:", err);
    } finally {
      setShowUploadModal(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-gray-800 text-gray-100 relative">
      <div className="p-4 sm:absolute sm:left-4 sm:top-4">
        <img src={logo} alt="Azure logo" className="h-16 w-16" />
      </div>

      {/* Synthetic call upload button positioned in the top-right corner */}
      <div className="absolute top-4 right-4">
        <Button
          onClick={onUploadSyntheticCall}
          className="mb-8 bg-gradient-to-r from-pink-500 via-purple-600 to-blue-700 hover:bg-pink-600"
        >
          {t("app.UploadSyntheticCall")}
        </Button>
      </div>

      <main className="flex flex-grow flex-col items-center justify-center">
        <h1 className="mb-8 bg-gradient-to-r from-orange-400 via-pink-500 to-purple-700 bg-clip-text text-4xl font-bold text-transparent md:text-7xl animate-gradient">
          Sky Voice Intelligence
        </h1>
        <div className="mb-4 flex flex-col items-center justify-center animate-gradient-circle">
          {isRecording && (
            <div className="mb-8">
              <AnimatedCircle
                amplitude={currentAmplitude}
                isRecording={isRecording}
              />
            </div>
          )}

          <Button
            onClick={onToggleListening}
            className={`h-12 w-60 bg-gradient-to-r ${
              isRecording
                ? "from-red-500 via-pink-600 to-red-700 gradient-x"
                : "from-pink-500 via-purple-600 to-blue-700"
            } hover:opacity-70`}
            aria-label={isRecording ? "Stop Recording" : "Start Recording"}
          >
            {isRecording ? (
              <>
                <MicOff className="mr-2 h-4 w-4" />
                Stop Conversation
              </>
            ) : (
              <>
                <Mic className="mr-2 h-6 w-6" />
                Start Conversation
              </>
            )}
          </Button>

          <StatusMessage isRecording={isRecording} />
        </div>

        <GroundingFiles files={groundingFiles} onSelected={setSelectedFile} />

        {showTranscriptButton && (
          <div className="mt-4">
            <Button
              onClick={onViewTranscriptClicked}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {loadingAnalytics
                ? t("app.loadingAnalysis")
                : t("app.viewCallTranscript")}
            </Button>
          </div>
        )}
      </main>

      <footer className="py-4 text-center">
        <p>{t("app.footer")}</p>
      </footer>

      <GroundingFileView
        groundingFile={selectedFile}
        onClosed={() => setSelectedFile(null)}
      />

      {/* Modal for file upload */}
      {showUploadModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 z-50">
          <div className="bg-white p-6 rounded shadow-lg">
            <h2 className="mb-4 text-xl font-bold">Upload JSON Transcript</h2>
            <label htmlFor="file-upload" className="sr-only">
              Upload JSON Transcript
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".json"
              ref={fileInputRef}
              onChange={onFileSelected}
              title="Upload JSON Transcript"
            />
            <div className="mt-4 flex justify-end">
              <Button
                onClick={() => setShowUploadModal(false)}
                className="bg-gray-500 hover:bg-gray-600 mr-2"
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
