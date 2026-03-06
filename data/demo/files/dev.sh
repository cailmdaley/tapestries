#!/bin/bash
# dev.sh - Start hexarchy frontend + backend, cleaning ports first

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ports
FRONTEND_PORT=5173
BACKEND_PORT=4004

# Kill processes on ports
echo "Cleaning ports..."
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend (in subshell with explicit directory)
echo "Starting backend on :$BACKEND_PORT..."
(cd "$SCRIPT_DIR/server" && npm run dev) &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Start frontend (in subshell with explicit directory)
echo "Starting frontend on :$FRONTEND_PORT..."
(cd "$SCRIPT_DIR" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Hexarchy running:"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Backend:  ws://localhost:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop both"

# Trap Ctrl+C to kill both
cleanup() {
    echo ""
    echo "Stopping..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# Wait for either to exit
wait
