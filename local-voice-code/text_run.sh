#!/bin/bash

# Text Assistant Launcher (Self-Contained)
# Test the chatbot functionality before trying voice mode

echo "⌨️  Launching Self-Contained Text Assistant..."
echo "============================================="

# Check if we're in the chatbot directory
if [ ! -f "text_assistant.py" ]; then
    echo "❌ text_assistant.py not found. Make sure you're in the chatbot/ directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv"
    echo "Please run: ./setup.sh first"
    exit 1
fi

echo "🔄 Activating virtual environment..."
source .venv/bin/activate

echo "🐍 Using Python: $(which python)"
echo "📦 Python version: $(python --version)"
echo ""

echo "⌨️  Starting Text Assistant..."
python text_assistant.py

echo ""
echo "👋 Text Assistant shut down successfully"
echo "🎤 Ready for voice mode? Run: ./voice_run.sh"
