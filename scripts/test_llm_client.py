from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.client import llm_generate_text

print(llm_generate_text("Say hello in 1 sentence."))
