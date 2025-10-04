import { useRef } from "react";
import { Recorder } from "@/components/audio/recorder";

// Larger buffer size for local voice processing - allows for better speech detection
const BUFFER_SIZE = 4800; // 4x larger than real-time API to accumulate more coherent audio chunks

type Parameters = {
    onAudioRecorded: (base64: string) => void;
};

export default function useLocalAudioRecorder({ onAudioRecorded }: Parameters) {
    const audioRecorder = useRef<Recorder>();
    const isPausedRef = useRef(false);

    let buffer = new Uint8Array();

    const appendToBuffer = (newData: Uint8Array) => {
        const newBuffer = new Uint8Array(buffer.length + newData.length);
        newBuffer.set(buffer);
        newBuffer.set(newData, buffer.length);
        buffer = newBuffer;
    };

    const handleAudioData = (data: Iterable<number>) => {
        // Skip processing audio data if paused (during TTS playback)
        if (isPausedRef.current) {
            console.log("🔇 Local audio recorder: Ignoring audio data during TTS playback");
            return;
        }

        const uint8Array = new Uint8Array(data);
        appendToBuffer(uint8Array);

        if (buffer.length >= BUFFER_SIZE) {
            const toSend = new Uint8Array(buffer.slice(0, BUFFER_SIZE));
            buffer = new Uint8Array(buffer.slice(BUFFER_SIZE));

            const regularArray = String.fromCharCode(...toSend);
            const base64 = btoa(regularArray);

            onAudioRecorded(base64);
        }
    };

    const start = async () => {
        try {
            console.log("useLocalAudioRecorder: Starting local audio recording...");
            if (!audioRecorder.current) {
                audioRecorder.current = new Recorder(handleAudioData);
                console.log("useLocalAudioRecorder: Created new Recorder instance for local voice");
            }
            console.log("useLocalAudioRecorder: Requesting microphone access...");
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log("useLocalAudioRecorder: Microphone access granted, starting local recorder...");
            audioRecorder.current.start(stream);
            console.log("useLocalAudioRecorder: Local audio recording started successfully");
        } catch (error) {
            console.error("useLocalAudioRecorder: Error starting local recording:", error);
            throw error;
        }
    };

    const stop = async () => {
        console.log("useLocalAudioRecorder: Stopping local audio recording...");
        await audioRecorder.current?.stop();
        isPausedRef.current = false; // Reset pause state when stopping
        console.log("useLocalAudioRecorder: Local audio recording stopped");
    };

    const pause = () => {
        console.log("🔇 Local audio recorder: Pausing microphone during TTS playback");
        isPausedRef.current = true;
        // Clear any accumulated buffer to prevent processing old audio after resume
        buffer = new Uint8Array();
    };

    const resume = () => {
        console.log("🎤 Local audio recorder: Resuming microphone after TTS playback");
        isPausedRef.current = false;
        // Clear buffer on resume to start fresh
        buffer = new Uint8Array();
    };

    return { start, stop, pause, resume };
}
