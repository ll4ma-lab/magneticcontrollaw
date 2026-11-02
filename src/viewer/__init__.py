"""viser viewer for the eddy-current simulator. Run it from src/ with `python -m viewer`."""
import os
import sys

# Griffin's modules use flat imports (from models import ...), so src/ must be importable,
# here and in the solver pool's workers, which copy sys.path from this process.
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
