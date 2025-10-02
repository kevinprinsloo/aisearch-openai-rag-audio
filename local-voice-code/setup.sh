#!/bin/bash

# Setup Script for Self-Contained Voice Assistant
# Run this once to set up the environment

echo "🛠️  Setting up Self-Contained Voice Assistant..."
echo "=============================================="

# Check if we're in the chatbot directory
if [ ! -f "voice_assistant.py" ]; then
    echo "❌ voice_assistant.py not found. Make sure you're in the chatbot/ directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo "📂 Contents: $(ls -la | head -5 | tail -4)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv .venv
    echo "✅ Virtual environment created at .venv"
else
    echo "✅ Virtual environment already exists at .venv"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

echo "🐍 Using Python: $(which python)"
echo "📦 Python version: $(python --version)"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the voice assistant:"
echo "   ./voice_run.sh"
echo ""
echo "🚀 To run text mode:"
echo "   ./text_run.sh"
echo ""
echo "🚀 Or manually:"
echo "   source .venv/bin/activate && python voice_assistant.py"
