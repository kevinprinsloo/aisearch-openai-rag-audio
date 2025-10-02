#!/bin/bash

# Voice Assistant Launcher (Self-Contained)
# Run this from the chatbot/ directory

echo "🚀 Launching Self-Contained Voice Assistant..."
echo "=============================================="

# Check if we're in the chatbot directory
if [ ! -f "voice_assistant.py" ]; then
    echo "❌ voice_assistant.py not found. Make sure you're in the chatbot/ directory"
    echo "Expected to be in: chatbot/"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if virtual environment activation script exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment activation script not found"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv .venv"
    exit 1
fi

echo "🔄 Activating virtual environment..."
source .venv/bin/activate

echo "🐍 Using Python: $(which python)"
echo "📦 Python version: $(python --version)"
echo ""

echo "🎤 Starting Voice Assistant..."
python voice_assistant.py

echo ""
echo "👋 Voice Assistant shut down successfully"
