from typing import List

LEVELS = {
    "TRACE",
    "DEBUG",
    "INFO",
    "WARN",
    "WARNING",
    "ERROR",
    "ERR",
    "CRITICAL",
    "FATAL",
    "NOTICE",
}


def split_log_entry(entry: str) -> List[str]:
    """
    Split a log entry string into [timestamp, level, message] using basic string operations.

    Heuristics supported (no regex):
    - "[timestamp] [LEVEL] message"
    - "[timestamp] LEVEL message"
    - "timestamp - LEVEL - message"
    - "timestamp | LEVEL | message"
    - "timestamp [LEVEL] message"
    - "timestamp LEVEL message" (LEVEL is a known level token)
    - Fallback: ["", "", original_entry_stripped]

    Returns:
        List[str]: [timestamp, level, message]
    """
    s = (entry or "").strip()
    if not s:
        return ["", "", ""]

    def strip_surrounding(s_: str, left: str, right: str) -> str:
        if len(s_) >= 2 and s_.startswith(left) and s_.endswith(right):
            return s_[1:-1].strip()
        return s_.strip()

    def split_first_token(s_: str):
        s_ = s_.lstrip()
        if not s_:
            return "", ""
        for i, ch in enumerate(s_):
            if ch.isspace():
                return s_[:i], s_[i + 1 :].lstrip()
        return s_, ""

    # 1) Bracketed prefix like "[timestamp] ..."
    if s.startswith("["):
        close = s.find("]")
        if close != -1:
            ts = s[1:close].strip()
            rest = s[close + 1 :].lstrip(" -:\t")
            if rest.startswith("["):
                # "[timestamp] [LEVEL] message"
                close2 = rest.find("]")
                if close2 != -1:
                    lvl = rest[1:close2].strip().upper()
                    msg = rest[close2 + 1 :].lstrip(" -:\t")
                    return [ts, lvl, msg]
            # "[timestamp] LEVEL message" or "[timestamp] message"
            token, remainder = split_first_token(rest)
            if token:
                token_up = token.upper().strip("[]")
                if token_up in LEVELS:
                    return [ts, token_up, remainder]
                # Not a level: treat the whole rest as message
                return [ts, "", rest]
            else:
                return [ts, "", ""]
        # If no closing bracket, fall through to other heuristics

    # 2) Pipe-delimited: "timestamp | LEVEL | message"
    if " | " in s:
        parts = s.split(" | ")
        if len(parts) >= 3:
            ts = parts[0].strip()
            lvl = parts[1].strip().upper()
            msg = " | ".join(parts[2:]).strip()
            return [ts, lvl, msg]

    # 3) Dash-delimited: "timestamp - LEVEL - message"
    if " - " in s:
        parts = s.split(" - ")
        if len(parts) >= 3:
            ts = parts[0].strip()
            lvl = parts[1].strip().upper()
            msg = " - ".join(parts[2:]).strip()
            return [ts, lvl, msg]

    # 4) "timestamp [LEVEL] message"
    lb = s.find("[")
    rb = s.find("]", lb + 1) if lb != -1 else -1
    if lb != -1 and rb != -1:
        ts = s[:lb].strip(" -:\t")
        lvl = s[lb + 1 : rb].strip().upper()
        msg = s[rb + 1 :].lstrip(" -:\t")
        return [ts, lvl, msg]

    # 5) Space-separated with known level token: "timestamp LEVEL message"
    tokens = s.split()
    # Find first token that looks like a level
    level_idx = -1
    for i, tok in enumerate(tokens):
        tok_up = tok.strip("[]").upper()
        if tok_up in LEVELS:
            level_idx = i
            break

    if level_idx != -1:
        ts = " ".join(tokens[:level_idx]).strip(" -:\t")
        lvl = tokens[level_idx].strip("[]").upper()
        msg = " ".join(tokens[level_idx + 1 :]).strip()
        return [ts, lvl, msg]

    # 6) Try to infer timestamp as first two tokens if they look like date + time
    #    e.g., "2025-09-22 10:34:56 something happened"
    if len(tokens) >= 3:
        t1, t2 = tokens[0], tokens[1]
        looks_like_date = "-" in t1 or "/" in t1 or "T" in t1
        looks_like_time = ":" in t2 or "T" in t2 or t2.endswith("Z")
        if looks_like_date and looks_like_time:
            ts = f"{t1} {t2}"
            msg = " ".join(tokens[2:]).strip()
            return [ts, "", msg]

    # Fallback: treat entire line as message
    return ["", "", s]


if __name__ == "__main__":
    # Simple manual tests when run as a script
    samples = [
        "[2025-09-22 10:11:12] [INFO] System started",
        "[2025-09-22 10:11:12] INFO System started",
        "2025-09-22 10:11:12 - INFO - System started",
        "2025-09-22T10:11:12Z | ERROR | Failure occurred",
        "2025-09-22 10:11:12 [DEBUG] Cache warmed",
        "2025-09-22 10:11:12 INFO System started",
        "2025-09-22 10:11:12 System started without level",
        "Unstructured line with no parts",
    ]
    for line in samples:
        print(line, "->", split_log_entry(line))
