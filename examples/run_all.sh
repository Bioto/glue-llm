#!/usr/bin/env bash
# Run all GlueLLM examples. Execute from gluellm/ directory or ensure gluellm package is installed.
# Usage: ./examples/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PROJECT_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

FAILED=()
for script in "$EXAMPLES_DIR"/*.py; do
  name="$(basename "$script")"
  echo ""
  echo "========================================"
  echo "Running: $name"
  echo "========================================"
  if python "$script"; then
    echo "✓ $name completed"
  else
    echo "✗ $name failed (exit $?)"
    FAILED+=("$name")
  fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All examples completed successfully."
  exit 0
else
  echo "Failed examples: ${FAILED[*]}"
  exit 1
fi
