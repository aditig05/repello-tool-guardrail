#!/usr/bin/env bash
# Convenience wrapper: runs the novelty check with the project venv.
cd "$(dirname "$0")"
exec .venv/bin/python scripts/redteam_novelty.py "$@"
