import { useState, useEffect } from "react";
import { Mic, MicOff, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { GroundingFiles } from "@/components/ui/grounding-files";
import GroundingFileView from "@/components/ui/grounding-file-view";
import { PerformanceMetrics } from "@/components/PerformanceMetrics";

import useLocalVoice from "@/hooks/useLocalVoice";
import useLocalAudioRecorder from "@/hooks/useLocalAudioRecorder";
import useLocalAudioPlayer from "@/hooks/useLocalAudioPlayer";

import { GroundingFile, ToolResult } from "./types";

import logo from "./assets/logo.svg";

interface WaveTextProps {
    text: string;
}

const WaveText = ({ text }: WaveTextProps) => {
    return (
        <span className="animate-gradient-sweep inline-block bg-gradient-to-r from-orange-500 via-pink-500 to-purple-500 bg-[length:200%_100%] bg-clip-text text-transparent">
            {text}
        </span>
    );
};

interface AIVisualizerProps {
    size?: number;
    isVisible: boolean;
    mode: "thinking" | "speaking" | "listening";
}

const AIVisualizer = ({ size = 120, isVisible, mode }: AIVisualizerProps) => {
    if (!isVisible) return null;

    const getColorScheme = (mode: string) => {
        switch (mode) {
            case "thinking":
                return {
                    primary: "#6366f1",
                    secondary: "#8b5cf6",
                    accent: "#a855f7",
                    glow: "rgba(139, 92, 246, 0.3)"
                };
            case "speaking":
                return {
                    primary: "#10b981",
                    secondary: "#059669",
                    accent: "#34d399",
                    glow: "rgba(16, 185, 129, 0.3)"
                };
            default: // listening
                return {
                    primary: "#3b82f6",
                    secondary: "#2563eb",
                    accent: "#60a5fa",
                    glow: "rgba(59, 130, 246, 0.3)"
                };
        }
    };

    const colors = getColorScheme(mode);

    return (
        <div className="mb-6 flex items-center justify-center">
            <div className="relative" style={{ width: size, height: size }}>
                {/* Outer glow ring */}
                <div
                    className="absolute inset-0 rounded-full opacity-40 blur-2xl"
                    style={{
                        transform: "scale(1.6)",
                        background: `radial-gradient(circle, ${colors.glow}, transparent 70%)`,
                        animation:
                            mode === "speaking"
                                ? "gentlePulse 1.2s ease-in-out infinite"
                                : mode === "thinking"
                                  ? "gentlePulse 2s ease-in-out infinite"
                                  : "gentlePulse 3s ease-in-out infinite"
                    }}
                />

                {/* Main circle with animated gradient */}
                <div
                    className="absolute inset-0 rounded-full"
                    style={{
                        background: `
                            radial-gradient(circle at 30% 30%, rgba(255,255,255,0.2) 0%, transparent 50%),
                            linear-gradient(135deg, 
                                ${
                                    mode === "thinking"
                                        ? "#8b5cf6, #ec4899, #f59e0b"
                                        : mode === "speaking"
                                          ? "#10b981, #3b82f6, #8b5cf6"
                                          : "#3b82f6, #06b6d4, #14b8a6"
                                })
                        `,
                        backgroundSize: "200% 100%",
                        boxShadow: `
                            0 0 ${size / 4}px ${colors.glow},
                            inset 0 0 ${size / 8}px rgba(255,255,255,0.1)
                        `,
                        animation: `
                            ${
                                mode === "thinking"
                                    ? "subtleMorph 4s ease-in-out infinite"
                                    : mode === "speaking"
                                      ? "speechMorph 0.8s ease-in-out infinite"
                                      : "idleMorph 6s ease-in-out infinite"
                            },
                            gradientSweep 3s ease-in-out infinite
                        `
                    }}
                />

                {/* Inner highlight */}
                <div
                    className="absolute rounded-full opacity-60"
                    style={{
                        top: "15%",
                        left: "15%",
                        width: "30%",
                        height: "30%",
                        background: "radial-gradient(circle, rgba(255,255,255,0.8), transparent 70%)",
                        filter: "blur(2px)"
                    }}
                />

                {/* Speech wave rings for speaking mode */}
                {mode === "speaking" && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        {[...Array(3)].map((_, i) => (
                            <div
                                key={i}
                                className="absolute rounded-full border"
                                style={{
                                    width: `${80 + i * 30}%`,
                                    height: `${80 + i * 30}%`,
                                    borderColor: colors.accent,
                                    borderWidth: "1px",
                                    opacity: 0.4 - i * 0.1,
                                    animation: `speechRings ${0.8 + i * 0.2}s ease-out infinite`,
                                    animationDelay: `${i * 0.15}s`
                                }}
                            />
                        ))}
                    </div>
                )}

                {/* Thinking particles */}
                {mode === "thinking" && (
                    <div className="absolute inset-0">
                        {[...Array(8)].map((_, i) => {
                            const angle = (i * 45 * Math.PI) / 180;
                            const radius = size * 0.4;
                            const x = 50 + ((Math.cos(angle) * radius) / size) * 100;
                            const y = 50 + ((Math.sin(angle) * radius) / size) * 100;

                            return (
                                <div
                                    key={i}
                                    className="absolute rounded-full"
                                    style={{
                                        left: `${x}%`,
                                        top: `${y}%`,
                                        width: "3px",
                                        height: "3px",
                                        background: colors.accent,
                                        transform: "translate(-50%, -50%)",
                                        animation: `thinkingOrbit ${3 + i * 0.2}s linear infinite`,
                                        animationDelay: `${i * 0.2}s`,
                                        opacity: 0.7
                                    }}
                                />
                            );
                        })}
                    </div>
                )}

                {/* Listening pulse dots */}
                {mode === "listening" && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        {[...Array(4)].map((_, i) => (
                            <div
                                key={i}
                                className="absolute rounded-full"
                                style={{
                                    width: "4px",
                                    height: "4px",
                                    background: colors.accent,
                                    top: `${45 + i * 2}%`,
                                    left: `${45 + i * 2}%`,
                                    animation: `listeningPulse ${2 + i * 0.3}s ease-in-out infinite`,
                                    animationDelay: `${i * 0.4}s`,
                                    opacity: 0.6
                                }}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

type ModelStatus = "initializing" | "ready" | "error" | "not-initialized";
type ProcessingStatus = "idle" | "recording" | "transcribing" | "thinking" | "speaking" | "generating";

interface PerformanceMetrics {
    transcription_ms?: number;
    llm_ms?: number;
    tts_ms?: number;
    total_ms?: number;
}

function LocalVoiceRAG() {
    const [isRecording, setIsRecording] = useState(false);
    const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({});

    // Add custom CSS animations for the AI visualizer
    const customStyles = `
        @keyframes gentlePulse {
            0%, 100% { 
                opacity: 0.3;
                transform: scale(1.6);
            }
            50% { 
                opacity: 0.5;
                transform: scale(1.8);
            }
        }
        
        @keyframes subtleMorph {
            0%, 100% { 
                border-radius: 50%;
                transform: scale(1);
            }
            25% { 
                border-radius: 48% 52% 50% 50% / 50% 48% 52% 50%;
                transform: scale(1.02);
            }
            50% { 
                border-radius: 52% 48% 48% 52% / 48% 52% 52% 48%;
                transform: scale(0.98);
            }
            75% { 
                border-radius: 50% 50% 52% 48% / 52% 48% 50% 52%;
                transform: scale(1.01);
            }
        }
        
        @keyframes speechMorph {
            0%, 100% { 
                border-radius: 50%;
                transform: scale(1);
            }
            12.5% { 
                border-radius: 42% 58% 48% 52% / 52% 42% 58% 48%;
                transform: scale(1.12);
            }
            25% { 
                border-radius: 58% 42% 55% 45% / 45% 58% 42% 55%;
                transform: scale(0.92);
            }
            37.5% { 
                border-radius: 48% 52% 42% 58% / 58% 48% 52% 42%;
                transform: scale(1.08);
            }
            50% { 
                border-radius: 52% 48% 58% 42% / 42% 52% 48% 58%;
                transform: scale(0.96);
            }
            62.5% { 
                border-radius: 45% 55% 52% 48% / 48% 45% 55% 52%;
                transform: scale(1.06);
            }
            75% { 
                border-radius: 55% 45% 48% 52% / 52% 55% 45% 48%;
                transform: scale(0.98);
            }
            87.5% { 
                border-radius: 50% 50% 45% 55% / 55% 50% 50% 45%;
                transform: scale(1.04);
            }
        }
        
        @keyframes idleMorph {
            0%, 100% { 
                border-radius: 50%;
                transform: scale(1);
            }
            50% { 
                border-radius: 48% 52% 48% 52% / 52% 48% 52% 48%;
                transform: scale(1.03);
            }
        }
        
        @keyframes speechRings {
            0% { 
                transform: scale(0.8);
                opacity: 0.6;
            }
            100% { 
                transform: scale(1.4);
                opacity: 0;
            }
        }
        
        @keyframes thinkingOrbit {
            0% { 
                transform: translate(-50%, -50%) rotate(0deg) translateX(20px) rotate(0deg);
                opacity: 0.5;
            }
            50% {
                opacity: 1;
            }
            100% { 
                transform: translate(-50%, -50%) rotate(360deg) translateX(20px) rotate(-360deg);
                opacity: 0.5;
            }
        }
        
        @keyframes listeningPulse {
            0%, 100% { 
                transform: scale(1);
                opacity: 0.4;
            }
            50% { 
                transform: scale(1.5);
                opacity: 0.8;
            }
        }
        
        @keyframes gradientSweep {
            0%, 100% { 
                background-position: 0% 50%;
            }
            50% { 
                background-position: 100% 50%;
            }
        }
    `;
    const [groundingFiles, setGroundingFiles] = useState<GroundingFile[]>([]);
    const [selectedFile, setSelectedFile] = useState<GroundingFile | null>(null);
    const [modelStatus, setModelStatus] = useState<ModelStatus>("not-initialized");
    const [processingStatus, setProcessingStatus] = useState<ProcessingStatus>("idle");
    const [statusMessage, setStatusMessage] = useState<string>("");
    const [transcribedText, setTranscribedText] = useState<string>("");
    const [responseText, setResponseText] = useState<string>("");
    const [isListening, setIsListening] = useState(false);
    const [bufferMessage, setBufferMessage] = useState<string>("");
    const [processingStartTime, setProcessingStartTime] = useState<number | null>(null);
    const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<number | null>(null);

    // Buffer messages for different processing stages
    const bufferMessages = {
        transcribing: ["Converting your speech to text...", "Processing audio with local models...", "Almost done transcribing..."],
        thinking: [
            "Let me think about that...",
            "Analyzing your question...",
            "Searching through knowledge base...",
            "Generating a thoughtful response...",
            "Almost ready with an answer...",
            "Just a moment while I process this...",
            "Working on the best response for you..."
        ],
        generating: ["Converting response to speech...", "Preparing audio response...", "Almost ready to speak..."]
    };

    // Cycle through buffer messages during processing
    useEffect(() => {
        if (processingStatus === "thinking" || processingStatus === "transcribing" || processingStatus === "generating") {
            const messages = bufferMessages[processingStatus];
            let messageIndex = 0;

            const interval = setInterval(() => {
                setBufferMessage(messages[messageIndex]);
                messageIndex = (messageIndex + 1) % messages.length;
            }, 2000); // Change message every 2 seconds

            return () => clearInterval(interval);
        } else {
            setBufferMessage("");
        }
    }, [processingStatus]);

    // Track processing time and show estimates
    useEffect(() => {
        if (processingStatus === "thinking") {
            setProcessingStartTime(Date.now());

            const estimateInterval = setInterval(() => {
                const elapsed = Date.now() - (processingStartTime || Date.now());

                // Estimate based on typical local LLM response times
                if (elapsed < 5000) {
                    setEstimatedTimeRemaining(Math.max(0, 8000 - elapsed));
                } else if (elapsed < 10000) {
                    setEstimatedTimeRemaining(Math.max(0, 15000 - elapsed));
                } else {
                    setEstimatedTimeRemaining(null); // Stop showing estimates after 10s
                }
            }, 1000);

            return () => {
                clearInterval(estimateInterval);
                setEstimatedTimeRemaining(null);
            };
        } else {
            setProcessingStartTime(null);
            setEstimatedTimeRemaining(null);
        }
    }, [processingStatus, processingStartTime]);

    // Warmup models on component mount
    useEffect(() => {
        const warmupModels = async () => {
            try {
                setModelStatus("initializing");
                setStatusMessage("Initializing AI models... This may take a minute on first run.");

                const response = await fetch("/api/local-voice/warmup", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    }
                });

                const result = await response.json();

                if (result.success) {
                    setModelStatus("ready");
                    setStatusMessage("AI models ready! Click the microphone to start.");
                } else {
                    setModelStatus("error");
                    setStatusMessage("Failed to initialize models. Some features may not work.");
                }
            } catch (error) {
                console.error("Warmup error:", error);
                setModelStatus("error");
                setStatusMessage("Error initializing models. Check connection.");
            }
        };

        warmupModels();
    }, []);

    const { startSession, addUserAudio, processCurrentAudio, clearTTSFlag } = useLocalVoice({
        onWebSocketOpen: () => console.log("Local voice connection opened"),
        onWebSocketError: (event: Event) => console.error("Local voice error:", event),
        onReceivedError: (message: { error: string }) => {
            console.error("error", message);

            // Resume microphone in case error occurred during TTS playback
            if (isListening) {
                resumeAudioRecording();
            }

            setProcessingStatus("idle");
            setStatusMessage("Error processing request. Please try again.");
            setIsListening(false);
            setIsRecording(false);
        },
        onProcessingStarted: () => {
            console.log("🚀 UI: Audio processing started");
            setProcessingStatus("transcribing");
            setStatusMessage("Processing your voice... please wait");
        },
        onTTSPlaybackFinished: () => {
            console.log("🔇 UI: TTS playback finished, ready for next question");

            // Resume microphone recording after TTS finishes to prevent feedback loop
            if (isListening) {
                resumeAudioRecording();
                setProcessingStatus("idle");
                setStatusMessage("Listening... ask your next question");
            } else {
                setProcessingStatus("idle");
                setStatusMessage("Click the microphone to start a new conversation");
            }
            // Clear transcribed text and response text after TTS finishes and user has had time to read it
            setTimeout(() => {
                setTranscribedText("");
                setResponseText("");
            }, 3000); // Give user 3 seconds to see their question and response after AI finishes speaking
        },
        onReceivedResponseAudioDelta: async (message: { delta: string }) => {
            console.log("🔊 LocalVoiceRAG: Received combined audio, playing...");

            // Pause microphone recording during TTS playback to prevent feedback loop
            pauseAudioRecording();

            setProcessingStatus("speaking");
            setStatusMessage("Listen to the AI response...");

            // Play the complete combined audio (no duration calculation needed - handled by useLocalVoice)
            await playAudio(message.delta);
        },
        onReceivedInputAudioBufferSpeechStarted: () => {
            console.log("🎤 UI: Speech started callback triggered");
            stopAudioPlayer();
            setIsRecording(true);
            setProcessingStatus("recording");
            setStatusMessage("Listening... speak your question now");
        },
        onReceivedInputAudioBufferSpeechStopped: () => {
            console.log("🛑 UI: Speech stopped callback triggered");
            setIsRecording(false);
            setProcessingStatus("transcribing");
            setStatusMessage("Processing speech... please wait");
        },
        onReceivedTranscription: (message: { transcription: string }) => {
            console.log("🎯 UI: Transcription received:", message.transcription);
            setTranscribedText(message.transcription);
            setProcessingStatus("thinking");
            setStatusMessage("AI is analyzing your question and generating a response...");
        },
        onReceivedResponseText: (message: { response_text: string }) => {
            console.log("📝 UI: Response text received:", message.response_text);
            setResponseText(message.response_text);
            setProcessingStatus("generating");
            setStatusMessage("Converting response to speech...");
        },
        onReceivedExtensionMiddleTierToolResponse: (message: any) => {
            const result: ToolResult = JSON.parse(message.tool_result);

            const files: GroundingFile[] = result.sources.map(x => {
                return { id: x.chunk_id, name: x.title, content: x.chunk };
            });

            setGroundingFiles(prev => [...prev, ...files]);
        },
        onMetricsReceived: (metrics: PerformanceMetrics) => {
            console.log("📊 Performance metrics received:", metrics);
            setPerformanceMetrics(metrics);
        }
    });

    const { play: playAudio, stop: stopAudioPlayer, cleanup: cleanupAudioPlayer, initialize: initializeAudioPlayer } = useLocalAudioPlayer();

    const {
        start: startAudioRecording,
        stop: stopAudioRecording,
        pause: pauseAudioRecording,
        resume: resumeAudioRecording
    } = useLocalAudioRecorder({
        onAudioRecorded: addUserAudio
    });

    // Only clear transcribed text after TTS finishes (handled by onTTSPlaybackFinished)
    // Removed the automatic timeout that was causing UI resets during speaking

    const onToggleListening = async () => {
        if (modelStatus !== "ready") {
            console.log("Models not ready, cannot start recording");
            return;
        }

        if (!isListening && !isRecording) {
            // Start listening mode
            console.log("LocalVoiceRAG: Starting voice session...");
            setIsListening(true);
            setProcessingStatus("idle");
            setStatusMessage("Ready to listen. Ask your question and I'll process it automatically when you pause.");
            setTranscribedText("");
            setResponseText("");

            // Initialize audio player with user gesture
            await initializeAudioPlayer();

            // Start the session and recording immediately
            startSession();
            await startAudioRecording();
            stopAudioPlayer();

            console.log("LocalVoiceRAG: Voice session started, now listening for speech");
        } else {
            // Stop listening mode
            console.log("LocalVoiceRAG: Stopping voice session...");
            setIsListening(false);
            setIsRecording(false);

            await stopAudioRecording();
            stopAudioPlayer();

            // If we're interrupting during TTS, clear the flag immediately
            if (processingStatus === "speaking") {
                console.log("LocalVoiceRAG: Interrupting TTS playback...");
                clearTTSFlag();
                setProcessingStatus("idle");
                setStatusMessage("Click the microphone to start a new conversation");
            } else {
                // Process any current audio before stopping (in case user spoke and then clicked stop)
                console.log("LocalVoiceRAG: Processing any current audio before stopping...");
                await processCurrentAudio();
            }

            console.log("LocalVoiceRAG: Voice session stopped");
        }
    };

    const { t } = useTranslation();

    // Cleanup audio player when component unmounts
    useEffect(() => {
        return () => {
            cleanupAudioPlayer();
        };
    }, [cleanupAudioPlayer]);

    return (
        <div className="flex min-h-screen flex-col bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 text-gray-900">
            {/* Custom CSS for AI visualizer animations */}
            <style dangerouslySetInnerHTML={{ __html: customStyles }} />

            {/* Navigation Header */}
            <header className="flex w-full items-center justify-between border-b border-gray-200 bg-white/80 p-4 shadow-sm backdrop-blur-md">
                <div className="flex items-center">
                    <img src={logo} alt="Azure logo" className="mr-4 h-12 w-12" />
                    <h2 className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-xl font-semibold text-transparent">Local Voice RAG</h2>
                </div>
                <Link to="/">
                    <Button variant="outline" className="border-blue-500 bg-blue-500 text-white transition-all duration-200 hover:bg-blue-600">
                        Back to Azure OpenAI
                    </Button>
                </Link>
            </header>

            <main className="flex flex-grow flex-col items-center justify-center px-4">
                <h1 className="mb-8 text-4xl font-bold md:text-7xl">
                    <WaveText text="Local Voice RAG" />
                </h1>

                {/* Model Status Indicator */}
                <div className="mb-6 flex items-center gap-2 rounded-full bg-white/60 px-4 py-2 shadow-sm backdrop-blur-sm">
                    {modelStatus === "initializing" && (
                        <>
                            <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                            <span className="font-medium text-blue-600">Initializing AI Models...</span>
                        </>
                    )}
                    {modelStatus === "ready" && (
                        <>
                            <CheckCircle className="h-5 w-5 text-green-500" />
                            <span className="font-medium text-green-600">AI Models Ready</span>
                        </>
                    )}
                    {modelStatus === "error" && (
                        <>
                            <AlertCircle className="h-5 w-5 text-red-500" />
                            <span className="font-medium text-red-600">Model Initialization Failed</span>
                        </>
                    )}
                </div>

                {/* AI Visualizer - shows different modes */}
                <AIVisualizer
                    isVisible={
                        processingStatus === "thinking" ||
                        processingStatus === "transcribing" ||
                        processingStatus === "speaking" ||
                        processingStatus === "generating" ||
                        (isListening && processingStatus === "idle")
                    }
                    size={160}
                    mode={
                        processingStatus === "thinking" || processingStatus === "transcribing" || processingStatus === "generating"
                            ? "thinking"
                            : processingStatus === "speaking"
                              ? "speaking"
                              : isListening && processingStatus === "idle"
                                ? "listening"
                                : "thinking"
                    }
                />

                <div className="mb-4 flex flex-col items-center justify-center">
                    <Button
                        onClick={onToggleListening}
                        disabled={modelStatus !== "ready"}
                        className={`h-14 w-64 text-base font-semibold shadow-lg transition-all duration-200 ${
                            modelStatus !== "ready"
                                ? "cursor-not-allowed bg-gray-400"
                                : isListening
                                  ? processingStatus === "speaking"
                                      ? "bg-orange-600 hover:bg-orange-700 hover:shadow-xl"
                                      : "bg-red-600 hover:bg-red-700 hover:shadow-xl"
                                  : "bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 hover:shadow-xl"
                        }`}
                        aria-label={isListening ? t("app.stopRecording") : t("app.startRecording")}
                    >
                        {modelStatus === "initializing" ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Initializing...
                            </>
                        ) : isListening ? (
                            <>
                                <MicOff className="mr-2 h-4 w-4" />
                                {processingStatus === "speaking" ? "Interrupt" : "Stop Voice Chat"}
                            </>
                        ) : (
                            <>
                                <Mic className="mr-2 h-6 w-6" />
                                Start Voice Chat
                            </>
                        )}
                    </Button>

                    {/* Processing Status */}
                    <div className="mt-3 text-center">
                        {processingStatus === "recording" && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-2 text-red-600">
                                    <div className="h-3 w-3 animate-pulse rounded-full bg-red-500"></div>
                                    <span className="font-medium">Recording your question...</span>
                                </div>
                                <p className="text-sm text-gray-500">Speak naturally, I'll process automatically when you pause</p>
                            </div>
                        )}
                        {processingStatus === "transcribing" && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-2 text-blue-600">
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                    <span className="font-medium">{bufferMessage || "Converting speech to text..."}</span>
                                </div>
                                <p className="text-sm text-gray-500">Processing your voice with local models</p>
                            </div>
                        )}
                        {processingStatus === "thinking" && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-2 text-purple-600">
                                    <span className="font-medium">{bufferMessage || "AI is analyzing your question..."}</span>
                                </div>
                                <div className="flex flex-col items-center gap-1">
                                    <p className="text-sm text-gray-500">Searching knowledge base and generating response</p>
                                    {estimatedTimeRemaining && estimatedTimeRemaining > 1000 && (
                                        <p className="text-xs font-medium text-blue-500">Estimated time: ~{Math.ceil(estimatedTimeRemaining / 1000)}s</p>
                                    )}
                                </div>
                            </div>
                        )}
                        {processingStatus === "speaking" && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-2 text-green-600">
                                    <div className="h-3 w-3 animate-pulse rounded-full bg-green-500"></div>
                                    <span className="font-medium">AI is speaking...</span>
                                </div>
                                <p className="text-sm text-gray-500">
                                    {isListening ? "Click 'Interrupt' button to stop and ask a new question" : "Listen to the complete response"}
                                </p>
                            </div>
                        )}
                        {processingStatus === "generating" && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-2 text-orange-600">
                                    <Loader2 className="h-5 w-5 animate-spin" />
                                    <span className="font-medium">{bufferMessage || "Converting response to speech..."}</span>
                                </div>
                                <p className="text-sm text-gray-500">Preparing audio response with local TTS</p>
                            </div>
                        )}
                        {processingStatus === "idle" && statusMessage && (
                            <div className="flex flex-col items-center gap-2">
                                <div className="font-medium text-gray-700">{statusMessage}</div>
                                {!isListening ? (
                                    <p className="text-sm text-gray-500">Click the microphone to start a conversation</p>
                                ) : (
                                    <p className="text-sm font-medium text-blue-600">
                                        🎤 Ask a question... I'll automatically process when you finish speaking
                                    </p>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Transcribed Text Display */}
                    {transcribedText && (
                        <div className="mt-8 max-w-2xl transform transition-all duration-500 ease-in-out animate-in fade-in slide-in-from-bottom-4">
                            <div className="rounded-2xl border border-blue-300/50 bg-white/70 p-6 shadow-xl backdrop-blur-sm">
                                <div className="mb-3 flex items-center">
                                    <div className="mr-2 h-2 w-2 animate-pulse rounded-full bg-blue-500"></div>
                                    <h3 className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-lg font-semibold text-transparent">
                                        Your Question
                                    </h3>
                                </div>
                                <p className="text-lg italic leading-relaxed text-gray-700">"{transcribedText}"</p>
                            </div>
                        </div>
                    )}

                    {/* AI Response Text Display */}
                    {responseText && (
                        <div className="mt-6 max-w-2xl transform transition-all duration-500 ease-in-out animate-in fade-in slide-in-from-bottom-4">
                            <div className="rounded-2xl border border-green-300/50 bg-white/70 p-6 shadow-xl backdrop-blur-sm">
                                <div className="mb-3 flex items-center">
                                    <div className="mr-2 h-2 w-2 animate-pulse rounded-full bg-green-500"></div>
                                    <h3 className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-lg font-semibold text-transparent">
                                        AI Response
                                    </h3>
                                </div>
                                <div className="prose prose-sm max-w-none text-lg leading-relaxed text-gray-700">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{responseText}</ReactMarkdown>
                                </div>
                                {processingStatus === "generating" && (
                                    <div className="mt-3 flex items-center text-sm text-orange-600">
                                        <div className="mr-2 h-1 w-1 animate-pulse rounded-full bg-orange-500"></div>
                                        <span>Converting to speech...</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
                <GroundingFiles files={groundingFiles} onSelected={setSelectedFile} />
            </main>

            <footer className="border-t border-gray-200 bg-white/40 py-4 text-center backdrop-blur-sm">
                <p className="text-gray-600">{t("app.footer")}</p>
            </footer>

            <GroundingFileView groundingFile={selectedFile} onClosed={() => setSelectedFile(null)} />

            {/* Performance Metrics Display */}
            <PerformanceMetrics
                transcriptionMs={performanceMetrics.transcription_ms}
                llmMs={performanceMetrics.llm_ms}
                ttsMs={performanceMetrics.tts_ms}
                totalMs={performanceMetrics.total_ms}
            />
        </div>
    );
}

export default LocalVoiceRAG;
