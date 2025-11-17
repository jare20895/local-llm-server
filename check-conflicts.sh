#!/bin/bash
# Quick script to check for running LLM server instances

echo "Checking for running LLM server instances..."
echo ""

# Check for Docker
DOCKER_RUNNING=$(docker ps --filter "name=homelab-llm-server" --format "{{.Names}}" 2>/dev/null)
if [ -n "$DOCKER_RUNNING" ]; then
    echo "❌ Docker container is RUNNING: $DOCKER_RUNNING"
    echo "   To stop: docker compose down"
else
    echo "✅ Docker container: Not running"
fi

# Check for local Python processes
LOCAL_RUNNING=$(ps aux | grep -E "python.*main.py" | grep -v "grep" | grep -v "root" | wc -l)
if [ "$LOCAL_RUNNING" -gt 0 ]; then
    echo "❌ Local dev processes found:"
    ps aux | grep -E "python.*main.py" | grep -v "grep" | grep -v "root"
    echo "   To kill: pkill -f 'python.*main.py'"
else
    echo "✅ Local dev: Not running"
fi

echo ""
if [ -n "$DOCKER_RUNNING" ] || [ "$LOCAL_RUNNING" -gt 0 ]; then
    echo "⚠️  WARNING: Server is already running!"
    exit 1
else
    echo "✅ All clear - safe to start server"
    exit 0
fi
