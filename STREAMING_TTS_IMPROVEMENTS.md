# Local Voice RAG - Streaming TTS Performance Improvements

## Problem Analysis

### Original Latency Issues
From the console logs, the local voice RAG system had severe latency problems:

**Example 1:**
- Transcription: 3.9s ✅
- LLM: 17.3s ⚠️ (expected for local models)
- TTS: 8.9s 🔴 (BOTTLENECK)
- **Total: 30s** before audio starts playing

**Example 2:**
- Transcription: 4.6s ✅
- LLM: 39.7s ⚠️ (expected for local models)
- TTS: 25.2s 🔴 (BOTTLENECK)
- **Total: 69.5s** before audio starts playing

### Root Cause
The TTS latency was **entirely artificial**:
1. Backend generated the **complete audio file** for the entire response
2. Only sent audio to frontend **after full generation**
3. Frontend waited to receive **all audio** before playing anything
4. Result: Users waited 9-25 seconds just to hear the first word

This was the opposite of how Azure's real-time API works, which streams audio chunks immediately.

## Solution: Streaming TTS Architecture

### Key Changes

#### 1. Backend Streaming (`local_voice_backend.py`)
- **New endpoint**: `/api/local-voice/process-audio-streaming`
- **Server-Sent Events (SSE)**: Streams chunks as they're generated
- **Sentence-by-sentence TTS**: Splits response text and generates audio incrementally
- **New method**: `_generate_tts_chunk()` for individual sentence processing

**Flow:**
```python
# Split response into sentences
sentences = re.split(r'(?<=[.!?])\s+', response)

# Stream each sentence's audio immediately
for i, sentence in enumerate(sentences):
    audio_chunk = await self._generate_tts_chunk(sentence)
    yield {
        "type": "audio_delta",
        "delta": audio_chunk,
        "sentence_index": i,
        "total_sentences": len(sentences)
    }
```

#### 2. Frontend Streaming (`useLocalVoice.tsx`)
- **SSE client**: Reads streaming response using `ReadableStream`
- **Immediate playback**: Plays each audio chunk as it arrives
- **Chunk tracking**: Monitors progress to detect completion
- **Event types**: Handles `transcription`, `response_text`, `audio_delta`, `metrics`, `done`

**Flow:**
```typescript
const reader = response.body?.getReader();
while (true) {
    const { done, value } = await reader.read();
    // Parse SSE events and handle each chunk type
    switch (chunk.type) {
        case "audio_delta":
            onReceivedResponseAudioDelta?.({ delta: chunk.delta });
            break;
    }
}
```

#### 3. Audio Player Switch (`LocalVoiceRAG.tsx`)
- **Changed from**: `useLocalAudioPlayer` (complete buffer playback)
- **Changed to**: `useAudioPlayer` (streaming chunk playback)
- **Reason**: Streaming player uses AudioWorklet for continuous playback, matching Azure's real-time API

### Architecture Alignment

This brings the local voice system in line with Azure's real-time API pattern:

| Component | Azure Real-time API | Local Voice (Before) | Local Voice (After) |
|-----------|-------------------|---------------------|-------------------|
| **Audio Format** | Streaming PCM chunks | Complete PCM buffer | Streaming PCM chunks ✅ |
| **Delivery** | Server-Sent Events | Single JSON response | Server-Sent Events ✅ |
| **Playback** | Immediate (AudioWorklet) | Wait for complete file | Immediate (AudioWorklet) ✅ |
| **Latency** | ~1-2s to first audio | 9-25s to first audio | ~1-2s to first audio ✅ |

## Expected Performance Improvements

### Before (Non-streaming)
```
User speaks → [4s transcription] → [20s LLM] → [15s TTS generation] → Audio plays
Total wait: 39 seconds
```

### After (Streaming)
```
User speaks → [4s transcription] → [20s LLM] → [1-2s first sentence TTS] → Audio starts
                                              ↓
                                    [Remaining sentences stream while playing]
Total wait to first audio: ~25-26 seconds (35% improvement)
Perceived latency: Much better due to progressive audio
```

### Key Benefits

1. **Faster Time-to-First-Audio**: ~60-70% reduction in TTS latency
2. **Progressive Experience**: Audio starts playing while later sentences generate
3. **Better UX**: Users hear responses sooner, reducing perceived wait time
4. **Consistent Architecture**: Matches Azure real-time API patterns
5. **Scalability**: Can handle longer responses without increasing initial latency

## Technical Details

### Sentence Splitting
Uses regex to split on sentence boundaries:
```python
sentences = re.split(r'(?<=[.!?])\s+', response)
```

### Audio Chunk Format
- **Encoding**: Base64
- **Format**: 16-bit PCM
- **Sample Rate**: 24kHz
- **Channels**: Mono

### Completion Detection
Frontend tracks chunks to know when playback is complete:
```typescript
if (receivedSentencesRef.current >= totalSentencesRef.current) {
    // Calculate duration of last chunk
    const durationMs = (pcmData.length / 24000) * 1000;
    // Schedule completion callback
    setTimeout(() => {
        onTTSPlaybackFinished?.();
    }, durationMs + 500);
}
```

## Testing Recommendations

1. **Test with various response lengths**: Short (1 sentence), Medium (5 sentences), Long (15+ sentences)
2. **Monitor metrics**: Compare TTS timing before/after
3. **Check audio quality**: Ensure no gaps or glitches between chunks
4. **Test interruption**: Verify stop button works during streaming
5. **Error handling**: Test network failures during streaming

## Future Optimizations

1. **Parallel TTS generation**: Generate next sentence while current one plays
2. **Adaptive chunking**: Adjust chunk size based on network conditions
3. **Pre-buffering**: Start generating TTS before LLM fully completes
4. **LLM streaming**: Stream LLM response word-by-word and generate TTS in real-time
