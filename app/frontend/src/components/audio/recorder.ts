export class Recorder {
    onDataAvailable: (buffer: Iterable<number>) => void;
    private audioContext: AudioContext | null = null;
    private mediaStream: MediaStream | null = null;
    private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
    private workletNode: AudioWorkletNode | null = null;

    public constructor(onDataAvailable: (buffer: Iterable<number>) => void) {
        this.onDataAvailable = onDataAvailable;
    }

    async start(stream: MediaStream) {
        try {
            // Clean up any existing audio context properly
            if (this.audioContext && this.audioContext.state !== 'closed') {
                await this.audioContext.close();
            }

            // Create a fresh AudioContext
            this.audioContext = new AudioContext({ sampleRate: 24000 });

            // Load the audio worklet module
            await this.audioContext.audioWorklet.addModule("./audio-processor-worklet.js");

            // Store the media stream
            this.mediaStream = stream;
            
            // Create the audio processing chain
            this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.mediaStream);

            this.workletNode = new AudioWorkletNode(this.audioContext, "audio-processor-worklet");
            this.workletNode.port.onmessage = event => {
                this.onDataAvailable(event.data.buffer);
            };

            // Connect the audio nodes
            this.mediaStreamSource.connect(this.workletNode);
            this.workletNode.connect(this.audioContext.destination);
        } catch (error) {
            console.error("Error starting recorder:", error);
            await this.stop();
        }
    }

    async stop() {
        // Stop all media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Disconnect and clean up worklet node
        if (this.workletNode) {
            try {
                this.workletNode.disconnect();
            } catch (e) {
                // Ignore if already disconnected
            }
            this.workletNode = null;
        }

        // Disconnect media stream source
        if (this.mediaStreamSource) {
            try {
                this.mediaStreamSource.disconnect();
            } catch (e) {
                // Ignore if already disconnected
            }
            this.mediaStreamSource = null;
        }

        // Close audio context if it's not already closed
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
        }
        this.audioContext = null;
    }
}
