/*
Local Voice Hook for Frontend

A custom hook that handles local voice communication instead of Azure OpenAI.
This hook provides the same interface as useRealTime but connects to local models.
*/

import { useState, useCallback, useRef, useEffect } from "react";

interface PerformanceMetrics {
    transcription_ms?: number;
    llm_ms?: number;
    tts_ms?: number;
    total_ms?: number;
    mode?: string;
    error?: string;
}

interface LocalVoiceParameters {
    onWebSocketOpen?: () => void;
    onWebSocketError?: (event: Event) => void;
    onReceivedResponseAudioDelta?: (message: { delta: string }) => void;
    onReceivedInputAudioBufferSpeechStarted?: () => void;
    onReceivedInputAudioBufferSpeechStopped?: () => void;
    onReceivedTranscription?: (message: { transcription: string }) => void;
    onReceivedResponseText?: (message: { response_text: string }) => void;
    onReceivedExtensionMiddleTierToolResponse?: (message: any) => void;
    onReceivedError?: (message: { error: string }) => void;
    onProcessingStarted?: () => void;
    onTTSPlaybackFinished?: () => void;
    onMetricsReceived?: (metrics: PerformanceMetrics) => void;
}

export default function useLocalVoice({
    onWebSocketOpen,
    onWebSocketError,
    onReceivedResponseAudioDelta,
    onReceivedInputAudioBufferSpeechStarted,
    onReceivedInputAudioBufferSpeechStopped,
    onReceivedTranscription,
    onReceivedResponseText,
    onReceivedExtensionMiddleTierToolResponse,
    onReceivedError,
    onProcessingStarted,
    onTTSPlaybackFinished,
    onMetricsReceived
}: LocalVoiceParameters) {
    const [isConnected, setIsConnected] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const audioChunksRef = useRef<string[]>([]);
    const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const isRecordingRef = useRef(false);
    const isManuallyStoppedRef = useRef(false);
    const isProcessingAudioRef = useRef(false);
    const lastAudioTimeRef = useRef<number>(Date.now());
    const silenceCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const isTTSPlayingRef = useRef(false);
    const speechDetectedRef = useRef(false);
    const speechChunksCountRef = useRef(0);
    const consecutiveSilenceCountRef = useRef(0);
    const totalSentencesRef = useRef(0);
    const receivedSentencesRef = useRef(0);
    const audioChunksBufferRef = useRef<string[]>([]);

    const connectWebSocket = useCallback(() => {
        try {
            // Connect to local voice backend (we'll create a simple HTTP endpoint instead)
            // For now, we'll simulate the WebSocket behavior with HTTP requests
            setIsConnected(true);
            onWebSocketOpen?.();
        } catch (error) {
            console.error("Failed to connect to local voice service:", error);
            onWebSocketError?.(error as Event);
        }
    }, [onWebSocketOpen, onWebSocketError]);

    const startSession = useCallback(() => {
        if (!isConnected) {
            connectWebSocket();
        }
        // Reset all flags when starting a new session
        isManuallyStoppedRef.current = false;
        isProcessingAudioRef.current = false;
        speechDetectedRef.current = false;
        speechChunksCountRef.current = 0;
        consecutiveSilenceCountRef.current = 0;
        console.log("Local voice session started");
    }, [isConnected, connectWebSocket]);

    const sendAudioToBackend = useCallback(
        async (audioData: string) => {
            try {
                console.log("📤 Local voice: Sending request to streaming backend...");
                const response = await fetch("/api/local-voice/process-audio-streaming", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        audio: audioData,
                        format: "webm"
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                // Handle Server-Sent Events stream
                const reader = response.body?.getReader();
                const decoder = new TextDecoder();

                if (!reader) {
                    throw new Error("No response body reader available");
                }

                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        console.log("📥 Local voice: Stream complete");
                        break;
                    }

                    // Decode the chunk and add to buffer
                    buffer += decoder.decode(value, { stream: true });

                    // Process complete events (SSE format: "data: {...}\n\n")
                    const events = buffer.split("\n\n");
                    buffer = events.pop() || ""; // Keep incomplete event in buffer

                    for (const event of events) {
                        if (!event.trim() || !event.startsWith("data: ")) continue;

                        try {
                            const jsonData = event.substring(6); // Remove "data: " prefix
                            const chunk = JSON.parse(jsonData);

                            console.log("📦 Received chunk:", chunk.type);

                            // Handle different chunk types
                            switch (chunk.type) {
                                case "transcription":
                                    console.log("🎯 TRANSCRIPTION:", chunk.transcription);
                                    onReceivedTranscription?.({ transcription: chunk.transcription });
                                    break;

                                case "response_text":
                                    console.log("🤖 AI RESPONSE TEXT:", chunk.response_text);
                                    onReceivedResponseText?.({ response_text: chunk.response_text });
                                    break;

                                case "audio_delta":
                                    console.log(`🔊 Audio chunk ${chunk.sentence_index + 1}/${chunk.total_sentences}`);

                                    // Set TTS playing flag on first audio chunk
                                    if (chunk.sentence_index === 0) {
                                        isTTSPlayingRef.current = true;
                                        totalSentencesRef.current = chunk.total_sentences;
                                        receivedSentencesRef.current = 0;
                                        audioChunksBufferRef.current = []; // Clear buffer for new response
                                        console.log("🎵 Starting to accumulate audio chunks...");
                                    }

                                    // Accumulate chunks instead of playing immediately
                                    audioChunksBufferRef.current.push(chunk.delta);
                                    receivedSentencesRef.current++;

                                    // When all chunks received, combine and play as one
                                    if (receivedSentencesRef.current >= totalSentencesRef.current) {
                                        console.log("🎵 All audio chunks received, combining and playing...");

                                        // Combine all chunks into one complete audio buffer
                                        const combinedAudio = audioChunksBufferRef.current.join("");
                                        console.log(`🔊 Playing combined audio: ${combinedAudio.length} chars`);

                                        // Play the complete combined audio
                                        onReceivedResponseAudioDelta?.({ delta: combinedAudio });

                                        // Calculate total duration for completion callback
                                        try {
                                            const binary = atob(combinedAudio);
                                            const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
                                            const pcmData = new Int16Array(bytes.buffer);
                                            const durationMs = (pcmData.length / 24000) * 1000;

                                            // Schedule completion callback
                                            setTimeout(() => {
                                                isTTSPlayingRef.current = false;
                                                onTTSPlaybackFinished?.();
                                                console.log("✅ TTS playback completed");
                                            }, durationMs + 500);
                                        } catch (error) {
                                            console.error("Error calculating combined audio duration:", error);
                                            // Fallback timeout
                                            setTimeout(() => {
                                                isTTSPlayingRef.current = false;
                                                onTTSPlaybackFinished?.();
                                            }, 10000);
                                        }
                                    }
                                    break;

                                case "metrics":
                                    console.log("📊 Performance metrics:", chunk.metrics);
                                    onMetricsReceived?.(chunk.metrics);
                                    break;

                                case "error":
                                    console.error("❌ Backend error:", chunk.error);
                                    onReceivedError?.({ error: chunk.error });
                                    break;

                                case "done":
                                    console.log("✅ Processing complete");
                                    break;
                            }
                        } catch (parseError) {
                            console.error("Error parsing event:", parseError, event);
                        }
                    }
                }
            } catch (error) {
                console.error("Local voice: Error sending audio to backend:", error);
                onReceivedError?.({ error: "Failed to process audio. Please try again." });
                throw error;
            }
        },
        [
            onReceivedError,
            onReceivedResponseAudioDelta,
            onReceivedExtensionMiddleTierToolResponse,
            onReceivedTranscription,
            onReceivedResponseText,
            onMetricsReceived
        ]
    );

    const processAccumulatedAudio = useCallback(async () => {
        if (audioChunksRef.current.length === 0) {
            console.log("Local voice: No audio chunks to process");
            return;
        }

        // Prevent duplicate processing
        if (isProcessingAudioRef.current) {
            console.log("🚫 Local voice: Already processing audio, skipping duplicate request");
            return;
        }

        console.log("🚀 Local voice: === STARTING AUDIO PROCESSING ===");
        console.log(`📊 Local voice: Processing ${audioChunksRef.current.length} audio chunks...`);
        isProcessingAudioRef.current = true;
        setIsProcessing(true);

        // Notify UI that processing has started
        onProcessingStarted?.();

        try {
            // Combine all audio chunks into one
            const combinedAudio = audioChunksRef.current.join("");
            console.log(`🔗 Local voice: Combined audio length: ${combinedAudio.length}`);

            // Clear the accumulated chunks
            audioChunksRef.current = [];

            // Check if audio is too large (over 400KB base64) and truncate if needed
            const maxChunkSize = 400000; // 400KB base64 (roughly 300KB binary)

            if (combinedAudio.length > maxChunkSize) {
                console.log(`⚠️ Audio too large (${combinedAudio.length}), truncating...`);
                // Take the last part (most recent audio)
                const truncatedAudio = combinedAudio.slice(-maxChunkSize);
                console.log(`📏 Truncated audio to: ${truncatedAudio.length}`);
                await sendAudioToBackend(truncatedAudio);
            } else {
                await sendAudioToBackend(combinedAudio);
            }
        } catch (error) {
            console.error("Local voice: Error processing audio:", error);
            onReceivedError?.({ error: "Failed to process audio" });
        } finally {
            setIsProcessing(false);
            isProcessingAudioRef.current = false; // Reset processing flag
            isRecordingRef.current = false; // Stop recording after processing
            console.log("Local voice: Finished processing audio");
        }
    }, [sendAudioToBackend, onReceivedError, onProcessingStarted]);

    const addUserAudio = useCallback(
        (base64Audio: string) => {
            // If manually stopped, processing, or TTS is playing, ignore new audio chunks
            if (isManuallyStoppedRef.current || isProcessing || isProcessingAudioRef.current || isTTSPlayingRef.current) {
                console.log("🚫 Local voice: Ignoring audio - stopped, processing, or TTS playing");
                return;
            }

            // Calculate audio energy to detect actual speech vs background noise
            const audioData = atob(base64Audio);
            const samples = new Int16Array(audioData.length / 2);
            for (let i = 0; i < audioData.length; i += 2) {
                samples[i / 2] = (audioData.charCodeAt(i + 1) << 8) | audioData.charCodeAt(i);
            }
            const rms = Math.sqrt(samples.reduce((sum, val) => sum + val * val, 0) / samples.length) / 32768;

            // Much higher threshold to detect actual speech vs background noise
            const SPEECH_THRESHOLD = 0.02; // Increased significantly to avoid false positives
            const hasSpeech = rms > SPEECH_THRESHOLD;

            console.log(`🎵 Audio RMS: ${rms.toFixed(4)}, Speech detected: ${hasSpeech}`);

            // Always add audio chunks for potential processing
            audioChunksRef.current.push(base64Audio);

            if (hasSpeech) {
                speechChunksCountRef.current++;
                consecutiveSilenceCountRef.current = 0;
                lastAudioTimeRef.current = Date.now();

                // Only start "recording" state after we detect actual speech
                if (!speechDetectedRef.current) {
                    speechDetectedRef.current = true;
                    console.log("🎤 Local voice: First speech detected, starting recording state");
                }

                // Only trigger UI recording state after we have enough speech
                if (!isRecordingRef.current && speechChunksCountRef.current >= 3) {
                    isRecordingRef.current = true;
                    onReceivedInputAudioBufferSpeechStarted?.();
                    console.log("🎤 Local voice: Recording started after detecting sufficient speech");

                    // Start periodic silence checking only after speech is confirmed
                    silenceCheckIntervalRef.current = setInterval(() => {
                        const timeSinceLastSpeech = Date.now() - lastAudioTimeRef.current;
                        // Only process if we have actual speech and sufficient silence
                        if (
                            timeSinceLastSpeech > 3000 &&
                            isRecordingRef.current &&
                            speechChunksCountRef.current >= 5 && // Minimum 5 speech chunks
                            !isManuallyStoppedRef.current &&
                            !isProcessingAudioRef.current
                        ) {
                            console.log(
                                `🔇 Silence after speech detected: Processing ${audioChunksRef.current.length} chunks (${speechChunksCountRef.current} speech chunks, ${timeSinceLastSpeech}ms since last speech)`
                            );
                            isRecordingRef.current = false;

                            // Clear the interval
                            if (silenceCheckIntervalRef.current) {
                                clearInterval(silenceCheckIntervalRef.current);
                                silenceCheckIntervalRef.current = null;
                            }

                            // Call speech stopped callback FIRST to update UI
                            onReceivedInputAudioBufferSpeechStopped?.();
                            // Small delay to ensure UI updates before processing starts
                            setTimeout(async () => {
                                await processAccumulatedAudio();
                            }, 100);
                        }
                    }, 500); // Check every 500ms
                }
            } else {
                // Track consecutive silence to avoid processing noise
                consecutiveSilenceCountRef.current++;
            }

            // Prevent excessive audio accumulation - but only if we have actual speech
            if (audioChunksRef.current.length > 300 && speechChunksCountRef.current >= 10) {
                console.log(`⚠️ Too many audio chunks (${audioChunksRef.current.length}) with speech (${speechChunksCountRef.current}), processing early...`);
                isRecordingRef.current = false;

                // Clear the interval
                if (silenceCheckIntervalRef.current) {
                    clearInterval(silenceCheckIntervalRef.current);
                    silenceCheckIntervalRef.current = null;
                }

                onReceivedInputAudioBufferSpeechStopped?.();
                setTimeout(async () => {
                    await processAccumulatedAudio();
                }, 100);
            }

            // Clear accumulated noise if we have too much silence without speech
            if (consecutiveSilenceCountRef.current > 50 && speechChunksCountRef.current < 3) {
                console.log(`🧹 Clearing ${audioChunksRef.current.length} noise chunks (${consecutiveSilenceCountRef.current} consecutive silence)`);
                audioChunksRef.current = [];
                speechChunksCountRef.current = 0;
                consecutiveSilenceCountRef.current = 0;
                speechDetectedRef.current = false;
            }
        },
        [onReceivedInputAudioBufferSpeechStarted, onReceivedInputAudioBufferSpeechStopped, processAccumulatedAudio, isProcessing]
    );

    const inputAudioBufferClear = useCallback(() => {
        // Clear accumulated audio chunks
        audioChunksRef.current = [];

        // Clear silence timeout
        if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }

        // Clear silence check interval
        if (silenceCheckIntervalRef.current) {
            clearInterval(silenceCheckIntervalRef.current);
            silenceCheckIntervalRef.current = null;
        }

        // Reset all recording and speech detection state
        isRecordingRef.current = false;
        speechDetectedRef.current = false;
        speechChunksCountRef.current = 0;
        consecutiveSilenceCountRef.current = 0;

        // Set manual stop flag
        isManuallyStoppedRef.current = true;

        // Reset processing flag
        isProcessingAudioRef.current = false;

        console.log("Local voice audio buffer cleared - manually stopped");
    }, []);

    // Function to manually process current audio (when user stops recording)
    const processCurrentAudio = useCallback(async () => {
        if (audioChunksRef.current.length > 0) {
            console.log("🛑 Local voice: Manual stop - processing current audio");
            isRecordingRef.current = false;

            // Clear any pending timeouts and intervals
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }
            if (silenceCheckIntervalRef.current) {
                clearInterval(silenceCheckIntervalRef.current);
                silenceCheckIntervalRef.current = null;
            }

            // Trigger speech stopped callback for UI feedback
            onReceivedInputAudioBufferSpeechStopped?.();

            // Process the audio
            await processAccumulatedAudio();

            // Set manual stop flag AFTER processing
            isManuallyStoppedRef.current = true;
        } else {
            console.log("⚠️ Local voice: No audio to process on manual stop");
            isManuallyStoppedRef.current = true; // Set manual stop flag even if no audio
        }
    }, [processAccumulatedAudio, onReceivedInputAudioBufferSpeechStopped]);

    // Cleanup effect
    useEffect(() => {
        return () => {
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
            }
            if (silenceCheckIntervalRef.current) {
                clearInterval(silenceCheckIntervalRef.current);
            }
        };
    }, []);

    const clearTTSFlag = useCallback(() => {
        console.log("🔇 Local voice: TTS playback finished, re-enabling microphone");
        isTTSPlayingRef.current = false;
        onTTSPlaybackFinished?.();
    }, [onTTSPlaybackFinished]);

    return {
        startSession,
        addUserAudio,
        inputAudioBufferClear,
        processCurrentAudio,
        clearTTSFlag,
        isConnected,
        isProcessing
    };
}
