from typing import List
import re

def tokenize_input_text(txt: str) -> List[str]:
    if not isinstance(txt, str):
        raise ValueError("txt must be a string")
    try:
        pattern = r"\w+(?:['’\-]\w+)*"
        return re.findall(pattern, txt, flags=re.UNICODE)
    except Exception as exc:
        raise ValueError("Tokenization failed") from exc
