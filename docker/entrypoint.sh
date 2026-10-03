#!/bin/sh
# Build the search index on first start (or when forced), then serve the API.
set -e
INDEX_DIR="${INDEX_DIR:-data/index}"
if [ ! -f "$INDEX_DIR/manifest.json" ] || [ "${REINGEST:-0}" = "1" ]; then
  python -m legal_core.ingest --corpus "${CORPUS_PATH:-data/corpus}" --index-dir "$INDEX_DIR"
fi
exec uvicorn legal_core.api:app --host 0.0.0.0 --port 8000
