#!/usr/bin/env python3
"""Entry point for RAG Fusion Factory application."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.main import main
    sys.exit(main())