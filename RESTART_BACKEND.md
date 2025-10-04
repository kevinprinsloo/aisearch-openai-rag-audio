# How to Restart Backend with Streaming TTS

## The Issue
The streaming TTS endpoint was added but the backend server needs to be restarted to pick up the new route.

## Quick Fix

### Option 1: Restart the Backend Server
If you're running the backend with `start.sh` or similar:

```bash
# Stop the current backend (Ctrl+C or kill the process)
# Then restart it:
cd /Users/kpl415/Documents/aisearch-openai-rag-audio
./start.sh
```

### Option 2: Using the Dev Container
If running in the dev container:

```bash
# In the terminal where the backend is running, press Ctrl+C
# Then restart:
python -m quart --app app/backend/app:app run --port 8767 --reload
```

### Option 3: Manual Python Run
```bash
cd /Users/kpl415/Documents/aisearch-openai-rag-audio
python -m aiohttp.web -H localhost -P 8767 app.backend.app:create_app
```

## Verify It's Working

After restarting, you should see the new streaming endpoint registered:
- `/api/local-voice/process-audio-streaming` (POST)

Test it by:
1. Opening the Local Voice RAG interface
2. Speaking a question
3. Checking the browser console for:
   - `📤 Local voice: Sending request to streaming backend...`
   - `📦 Received chunk: transcription`
   - `📦 Received chunk: response_text`
   - `📦 Received chunk: audio_delta`
   - `🔊 Audio chunk 1/N` (where N is the number of sentences)

## Expected Behavior

**Before (non-streaming):**
- Single log: `🔊 LocalVoiceRAG: Received audio delta, playing...`
- Long wait before audio starts

**After (streaming):**
- Multiple logs: `🔊 Audio chunk 1/5`, `🔊 Audio chunk 2/5`, etc.
- Audio starts playing within 1-2 seconds
- Progressive playback as chunks arrive

## Troubleshooting

### Still getting 405 Method Not Allowed?
- Ensure you restarted the backend completely
- Check the backend logs for the route registration
- Verify the port is correct (should be 8767 based on your logs)

### No audio playing?
- Check browser console for audio player errors
- Verify `useAudioPlayer` is being used (not `useLocalAudioPlayer`)
- Check that audio chunks are being received

### Backend not starting?
- Check for Python errors in the terminal
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify the local voice models are initialized
