import { useRef, useCallback } from "react";

const SAMPLE_RATE = 24000;

export default function useLocalAudioPlayer() {
    const audioContextRef = useRef<AudioContext | null>(null);
    const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);

    const init = async () => {
        // Check if context exists and is not closed
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
            audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
            console.log(`🎵 Created NEW audio context with state: ${audioContextRef.current.state}`);
        } else {
            console.log(`🎵 Reusing existing audio context with state: ${audioContextRef.current.state}`);
        }
        
        // Always try to resume if suspended
        if (audioContextRef.current.state === 'suspended') {
            console.log('🔊 Resuming suspended audio context during init...');
            await audioContextRef.current.resume();
            console.log(`🔊 Audio context state after resume: ${audioContextRef.current.state}`);
        }
    };

    const play = async (base64Audio: string) => {
        try {
            await init();
            
            // Ensure audio context is running
            if (audioContextRef.current!.state === 'suspended') {
                console.log('🔊 Resuming suspended audio context...');
                await audioContextRef.current!.resume();
                console.log(`🔊 Audio context state after resume: ${audioContextRef.current!.state}`);
            }
            
            // Stop any currently playing audio
            stop();

            const binary = atob(base64Audio);
            const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
            const pcmData = new Int16Array(bytes.buffer);

            console.log(`🎵 Playing local audio: ${pcmData.length} samples (${(pcmData.length / SAMPLE_RATE).toFixed(1)}s)`);
            console.log(`🎵 Audio context state: ${audioContextRef.current!.state}`);
            console.log(`🎵 Audio context sample rate: ${audioContextRef.current!.sampleRate}`);
            console.log(`🎵 Base64 audio length: ${base64Audio.length}`);
            console.log(`🎵 Binary data length: ${binary.length}`);
            console.log(`🎵 PCM data length: ${pcmData.length}`);

            // Check for audio data validity (process in chunks to avoid stack overflow)
            let maxSample = -32768;
            let minSample = 32767;
            let sumSample = 0;
            
            for (let i = 0; i < pcmData.length; i++) {
                const sample = pcmData[i];
                if (sample > maxSample) maxSample = sample;
                if (sample < minSample) minSample = sample;
                sumSample += Math.abs(sample);
            }
            
            const avgSample = sumSample / pcmData.length;
            console.log(`🎵 Audio data stats - Max: ${maxSample}, Min: ${minSample}, Avg: ${avgSample.toFixed(2)}`);

            if (avgSample < 100) {
                console.warn('⚠️ Audio data seems very quiet or empty!');
            }

            // Create audio buffer
            const audioBuffer = audioContextRef.current!.createBuffer(1, pcmData.length, SAMPLE_RATE);
            const channelData = audioBuffer.getChannelData(0);

            // Convert PCM data to float32 and copy to buffer
            for (let i = 0; i < pcmData.length; i++) {
                channelData[i] = pcmData[i] / 32768; // Convert from int16 to float32
            }

            // Create and configure source node with gain for volume control
            const source = audioContextRef.current!.createBufferSource();
            const gainNode = audioContextRef.current!.createGain();
            
            source.buffer = audioBuffer;
            gainNode.gain.value = 1.0; // Full volume
            
            // Connect: source -> gain -> destination
            source.connect(gainNode);
            gainNode.connect(audioContextRef.current!.destination);

            // Store reference for stopping
            currentSourceRef.current = source;

            console.log('🚀 Starting audio playback with gain node...');
            console.log(`🔊 Gain value: ${gainNode.gain.value}`);
            console.log(`🔊 Audio context current time: ${audioContextRef.current!.currentTime}`);
            console.log(`🔊 Audio context destination: ${audioContextRef.current!.destination}`);
            
            // Start playback immediately
            source.start(0);
            console.log('🎵 Audio source started!');
            console.log(`🎵 Expected playback duration: ${(pcmData.length / SAMPLE_RATE).toFixed(2)}s`);
            console.log(`🎵 Will finish at audio context time: ${(audioContextRef.current!.currentTime + (pcmData.length / SAMPLE_RATE)).toFixed(2)}`);

            // Clean up reference when playback ends
            source.onended = () => {
                console.log('✅ Audio playback finished naturally');
                console.log(`✅ Audio context time when ended: ${audioContextRef.current?.currentTime.toFixed(2)}`);
                currentSourceRef.current = null;
            };

        } catch (error) {
            console.error("❌ Error playing local audio:", error);
            if (error instanceof Error) {
                console.error("❌ Error stack:", error.stack);
            }
        }
    };

    const stop = useCallback(() => {
        if (currentSourceRef.current) {
            console.log('⏹️ Stopping audio playback...');
            try {
                currentSourceRef.current.stop();
                console.log('⏹️ Audio source stopped');
            } catch (error) {
                console.log('⏹️ Audio source already stopped or error:', error);
            }
            currentSourceRef.current = null;
        }
    }, []);

    const cleanup = useCallback(() => {
        console.log('🧹 Cleaning up audio player...');
        stop();
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            console.log('🔒 Closing audio context...');
            audioContextRef.current.close();
        }
        audioContextRef.current = null;
    }, [stop]);

    const initialize = useCallback(async () => {
        await init();
        
        // Force audio context to start with user interaction
        if (audioContextRef.current && audioContextRef.current.state !== 'running') {
            console.log('🔊 Forcing audio context to resume with user interaction...');
            await audioContextRef.current.resume();
            console.log(`🔊 Audio context state after user interaction: ${audioContextRef.current.state}`);
        }
        
        // Test audio output immediately after user interaction
        if (audioContextRef.current && audioContextRef.current.state === 'running') {
            console.log('🔔 Testing audio output with user interaction...');
            const testOscillator = audioContextRef.current.createOscillator();
            const testGain = audioContextRef.current.createGain();
            
            testOscillator.connect(testGain);
            testGain.connect(audioContextRef.current.destination);
            
            testOscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime);
            testGain.gain.setValueAtTime(0.1, audioContextRef.current.currentTime);
            testGain.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + 0.1);
            
            testOscillator.start(audioContextRef.current.currentTime);
            testOscillator.stop(audioContextRef.current.currentTime + 0.1);
            
            console.log('🔔 Initialization test beep should be audible now...');
        }
    }, []);

    return { play, stop, cleanup, initialize };
}
