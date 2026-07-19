"""Shared constants for the RAG pipeline.

Centralized here so main.py, dataIngestion.py, and userQuery.py can't drift
out of sync (previously COLLECTION_NAME and the Qdrant path were duplicated
independently in multiple files).
"""

import os
from pathlib import Path

# Resolved relative to this file rather than the process's CWD, so the DB
# location doesn't change depending on where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_DB_PATH = str(PROJECT_ROOT / "qdrant_db")

COLLECTION_NAME = "my_docs"

EMBEDDING_MODEL = "BAAI/bge-m3"

# Small, official Google QAT-quantized checkpoint (~7.3 GiB VRAM). A
# different size class from the A2A orchestrator's E4B-it model, so once
# integrated the two can coexist on one GPU without competing for memory.
# Overridable per-machine since VRAM budgets differ standalone vs. alongside
# the orchestrator.
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "google/gemma-4-E2B-it-qat-w4a16-ct")
GPU_MEMORY_UTILIZATION = float(os.environ.get("RAG_GPU_MEMORY_UTILIZATION", "0.5"))
MAX_MODEL_LEN = int(os.environ.get("RAG_MAX_MODEL_LEN", "4096"))
