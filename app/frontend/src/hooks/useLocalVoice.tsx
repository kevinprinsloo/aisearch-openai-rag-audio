/*
Local Voice Hook for Frontend

A custom hook that handles local voice communication instead of Azure OpenAI.
This hook provides the same interface as useRealTime but connects to local models.
*/

import { useState, useCallback, useRef, useEffect } from "react";

interface LocalVoiceParameters {
    onWebSocketOpen?: () => void;
    onWebSocketError?: (event: Event) => void;
    onReceivedResponseAudioDelta?: (message: { delta: string }) => void;
    onReceivedInputAudioBufferSpeechStarted?: () => void;
    onReceivedExtensionMiddleTierToolResponse?: (message: any) => void;
    onReceivedError?: (message: { error: string }) => void;
}

export default function useLocalVoice({
    onWebSocketOpen,
    onWebSocketError,
    onReceivedResponseAudioDelta,
    onReceivedInputAudioBufferSpeechStarted,
    onReceivedExtensionMiddleTierToolResponse,
    onReceivedError
}: LocalVoiceParameters) {
    const [isConnected, setIsConnected] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const audioChunksRef = useRef<string[]>([]);
    const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const isRecordingRef = useRef(false);
    const isManuallyStoppedRef = useRef(false);

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
        // Reset manual stop flag when starting a new session
        isManuallyStoppedRef.current = false;
        console.log("Local voice session started");
    }, [isConnected, connectWebSocket]);

    const sendAudioToBackend = useCallback(
        async (audioData: string) => {
            try {
                console.log("📤 Local voice: Sending request to backend...");
                const response = await fetch("/api/local-voice/process-audio", {
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

                const result = await response.json();
                console.log("📥 Local voice: Received response from backend:", result);

                if (result.error) {
                    console.error("❌ Local voice: Backend returned error:", result.error);
                    onReceivedError?.({ error: result.error });
                    return;
                }

                // Display transcription and response prominently
                if (result.transcription) {
                    console.log("🎯 TRANSCRIPTION:", result.transcription);
                }

                if (result.response) {
                    console.log("🤖 AI RESPONSE:", result.response);
                }

                // Simulate receiving audio response in chunks
                if (result.audio) {
                    console.log("🔊 Local voice: Playing back audio response...");
                    const audioData = result.audio;
                    const chunkSize = 1024;

                    for (let i = 0; i < audioData.length; i += chunkSize) {
                        const chunk = audioData.slice(i, i + chunkSize);
                        onReceivedResponseAudioDelta?.({ delta: chunk });
                        // Small delay to simulate streaming
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }
                }

                // Send grounding information if available
                if (result.sources) {
                    onReceivedExtensionMiddleTierToolResponse?.({
                        tool_result: JSON.stringify({
                            sources: result.sources
                        })
                    });
                }
            } catch (error) {
                console.error("Local voice: Error sending audio to backend:", error);
                throw error; // Re-throw to be caught by processAccumulatedAudio
            }
        },
        [onReceivedError, onReceivedResponseAudioDelta, onReceivedExtensionMiddleTierToolResponse]
    );

    const processAccumulatedAudio = useCallback(async () => {
        if (audioChunksRef.current.length === 0) {
            console.log("Local voice: No audio chunks to process");
            return;
        }

        console.log("🚀 Local voice: === STARTING AUDIO PROCESSING ===");
        console.log(`📊 Local voice: Processing ${audioChunksRef.current.length} audio chunks...`);
        setIsProcessing(true);

        try {
            // Combine all audio chunks into one
            const combinedAudio = audioChunksRef.current.join("");
            console.log(`🔗 Local voice: Combined audio length: ${combinedAudio.length}`);

            // Clear the accumulated chunks
            audioChunksRef.current = [];

            // Check if audio is too large (over 800KB base64) and truncate if needed
            const maxChunkSize = 800000; // 800KB base64 (roughly 600KB binary)

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
            isRecordingRef.current = false; // Stop recording after processing
            console.log("Local voice: Finished processing audio");
        }
    }, [sendAudioToBackend, onReceivedError]);

    const addUserAudio = useCallback(
        (base64Audio: string) => {
            // If manually stopped or processing, ignore new audio chunks
            if (isManuallyStoppedRef.current || isProcessing) {
                console.log("🚫 Local voice: Ignoring audio - stopped or processing");
                return;
            }

            audioChunksRef.current.push(base64Audio);

            // Start recording if this is the first chunk
            if (!isRecordingRef.current) {
                isRecordingRef.current = true;
                onReceivedInputAudioBufferSpeechStarted?.();
                console.log("🎤 Local voice: Started recording");
            }

            // Reset silence timeout on each new audio chunk
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
            }

            // Silence detection timeout - triggers if no new audio comes for a while
            silenceTimeoutRef.current = setTimeout(async () => {
                if (isRecordingRef.current && audioChunksRef.current.length > 0 && !isManuallyStoppedRef.current) {
                    console.log(`🔇 Silence detected: Processing ${audioChunksRef.current.length} chunks`);
                    isRecordingRef.current = false;
                    await processAccumulatedAudio();
                }
            }, 3000); // 3 second silence timeout
        },
        [onReceivedInputAudioBufferSpeechStarted, processAccumulatedAudio, isProcessing]
    );

    const inputAudioBufferClear = useCallback(() => {
        // Clear accumulated audio chunks
        audioChunksRef.current = [];

        // Clear silence timeout
        if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
        }

        // Reset recording state
        isRecordingRef.current = false;

        // Set manual stop flag
        isManuallyStoppedRef.current = true;

        console.log("Local voice audio buffer cleared - manually stopped");
    }, []);

    // Function to manually process current audio (when user stops recording)
    const processCurrentAudio = useCallback(async () => {
        if (audioChunksRef.current.length > 0) {
            console.log("🛑 Local voice: Manual stop - processing current audio");
            isRecordingRef.current = false;
            isManuallyStoppedRef.current = true; // Set manual stop flag

            // Clear any pending timeouts
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
                silenceTimeoutRef.current = null;
            }

            await processAccumulatedAudio();
        } else {
            console.log("⚠️ Local voice: No audio to process on manual stop");
            isManuallyStoppedRef.current = true; // Set manual stop flag even if no audio
        }
    }, [processAccumulatedAudio]);

    // Cleanup effect
    useEffect(() => {
        return () => {
            if (silenceTimeoutRef.current) {
                clearTimeout(silenceTimeoutRef.current);
            }
        };
    }, []);

    return {
        startSession,
        addUserAudio,
        inputAudioBufferClear,
        processCurrentAudio,
        isConnected,
        isProcessing
    };
}
