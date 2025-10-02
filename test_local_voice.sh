#!/bin/bash

# Test script to simulate voice interaction with the local voice backend

echo "Testing Local Voice Backend Integration"
echo "======================================"

# Test 1: Health check
echo "1. Testing health endpoint..."
health_response=$(curl -s http://localhost:8767/api/local-voice/health)
echo "Health Response: $health_response"
echo ""

# Test 2: Audio processing with mock data
echo "2. Testing audio processing endpoint..."
# Create a larger mock audio payload to simulate real audio
mock_audio="dGVzdCBhdWRpbyBkYXRhIGZvciB0ZXN0aW5nIHB1cnBvc2VzIG9ubHkgdGhpcyBpcyBhIGxvbmdlciBzdHJpbmcgdG8gc2ltdWxhdGUgYWN0dWFsIGF1ZGlvIGRhdGE="

audio_response=$(curl -s -X POST http://localhost:8767/api/local-voice/process-audio \
    -H "Content-Type: application/json" \
    -d "{\"audio\": \"$mock_audio\", \"format\": \"webm\"}")

echo "Audio Processing Response:"
echo "$audio_response" | python3 -m json.tool 2>/dev/null || echo "$audio_response"
echo ""

# Test 3: Check backend logs
echo "3. Backend is listening on port 8767 and should show processing logs"
echo "   Check the terminal running the backend for detailed processing information"
echo ""

echo "Test completed!"