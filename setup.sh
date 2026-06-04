#!/usr/bin/env bash
# One-shot setup for the competition agent
# Run: bash setup.sh
set -e

echo "=== Arena PokerKit Competition Setup ==="

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"

# 2. Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "uv: $(uv --version)"

# 3. Install dependencies
echo "Installing dependencies..."
uv sync

# 4. Set up .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env from template."
    echo ""
    echo "IMPORTANT: Add your Anthropic API key to .env:"
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
    echo "Also register at https://b-arena.dev.fun/poker-eval for a competition ID."
fi

# 5. Make pokerkit executable
chmod +x pokerkit

# 6. Smoke test
echo ""
echo "Running smoke test..."
./pokerkit run --dry-run --max-hands 1
echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick start:"
echo "  1. Edit .env and add ANTHROPIC_API_KEY"
echo "  2. Register at https://b-arena.dev.fun/poker-eval"
echo "  3. Run 50-hand preview:   ./pokerkit llm --max-hands 50"
echo "  4. Full competition run:  ./pokerkit llm"
echo ""
echo "For development (no API cost):  ./pokerkit run --max-hands 50"
