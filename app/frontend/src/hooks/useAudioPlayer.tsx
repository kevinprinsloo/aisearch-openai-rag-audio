import { useRef, useEffect } from "react";

import { Player } from "@/components/audio/player";

const SAMPLE_RATE = 24000;

export default function useAudioPlayer() {
    const audioPlayer = useRef<Player>();
    const isInitialized = useRef(false);

    const reset = async () => {
        audioPlayer.current = new Player();
        await audioPlayer.current.init(SAMPLE_RATE);
        isInitialized.current = true;
        console.log("🎵 Audio player initialized for streaming");
    };

    const play = (base64Audio: string) => {
        if (!isInitialized.current) {
            console.warn("⚠️ Audio player not initialized, initializing now...");
            reset().then(() => {
                const binary = atob(base64Audio);
                const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
                const pcmData = new Int16Array(bytes.buffer);
                audioPlayer.current?.play(pcmData);
            });
            return;
        }

        const binary = atob(base64Audio);
        const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
        const pcmData = new Int16Array(bytes.buffer);

        audioPlayer.current?.play(pcmData);
    };

    const stop = () => {
        audioPlayer.current?.stop();
    };

    // Initialize on mount
    useEffect(() => {
        reset();
    }, []);

    return { reset, play, stop };
}
