#!/bin/bash
# Quick start script for radar-cog-processor

echo "🌩️ Radar COG Processor - Quick Start"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Run this script from the radar-cog-processor directory"
    exit 1
fi

echo "📦 Installing package..."
pip install -e . || {
    echo "❌ Installation failed"
    exit 1
}

echo ""
echo "✅ Installation complete!"
echo ""
echo "📚 Quick Reference:"
echo ""
echo "Basic usage:"
echo "  python -c \"from radar_processor import process_radar_to_cog; help(process_radar_to_cog)\""
echo ""
echo "Run tests:"
echo "  pytest tests/"
echo ""
echo "Run examples:"
echo "  cd examples && python basic_usage.py"
echo ""
echo "View documentation:"
echo "  cat README.md"
echo ""
echo "Install with dev dependencies:"
echo "  pip install -e \".[dev]\""
echo ""
echo "Happy radar processing! 🎉"
