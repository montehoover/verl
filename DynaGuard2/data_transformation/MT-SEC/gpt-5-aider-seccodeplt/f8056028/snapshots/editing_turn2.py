from typing import List, Dict

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


def identify_log_parts(log_entry: str) -> Dict[str, str]:
    """
    Identify and extract key components from a log entry using basic string operations
    and common delimiters like brackets [] and colons :.

    Returns a dict with keys: timestamp, log_level, message
    """
    s = (log_entry or "").strip()
    result = {"timestamp": "", "log_level": "", "message": ""}
    if not s:
        return result

    def consume_bracketed_prefix(s_: str):
        if s_.startswith("["):
            j = s_.find("]")
            if j != -1:
                return s_[1:j].strip(), s_[j + 1 :].lstrip(" \t-:")
        return "", s_

    def split_first_token_local(s_: str):
        s_ = s_.lstrip()
        if not s_:
            return "", ""
        for i, ch in enumerate(s_):
            if ch.isspace():
                return s_[:i], s_[i + 1 :].lstrip()
        return s_, ""

    def parse_date_time_prefix(s_: str):
        tokens = s_.split()
        if len(tokens) >= 2:
            t1, t2 = tokens[0], tokens[1]
            looks_like_date = "-" in t1 or "/" in t1 or "T" in t1
            looks_like_time = ":" in t2 or "T" in t2 or t2.endswith("Z")
            if looks_like_date and looks_like_time:
                ts_ = f"{t1} {t2}"
                rest_ = " ".join(tokens[2:]).lstrip(" \t-:")
                return ts_, rest_
        return "", s_

    def find_level_token(s_: str):
        tokens = s_.split()
        if not tokens:
            return "", s_
        # Prefer last token as level (common: "... INFO" or "... INFO:")
        last = tokens[-1]
        last_core = last[:-1] if last.endswith(":") else last
        core_up = last_core.strip("[]").upper()
        if core_up in LEVELS:
            return core_up, " ".join(tokens[:-1]).strip()
        # Also try first token (common: "INFO ...")
        first = tokens[0]
        first_core = first[:-1] if first.endswith(":") else first
        first_up = first_core.strip("[]").upper()
        if first_up in LEVELS:
            return first_up, " ".join(tokens[1:]).strip()
        return "", s_

    # A) "[timestamp] [LEVEL] : message" or "[timestamp] LEVEL: message"
    ts, remainder = consume_bracketed_prefix(s)
    if ts:
        # Try a bracketed level next
        lvl_candidate, rest_after_level = consume_bracketed_prefix(remainder)
        if lvl_candidate:
            lvl_up = lvl_candidate.upper()
            msg = rest_after_level.lstrip()
            if msg.startswith(":"):
                msg = msg[1:].lstrip()
            if lvl_up in LEVELS:
                return {"timestamp": ts, "log_level": lvl_up, "message": msg}
            # Not a known level -> treat remainder as message
            return {"timestamp": ts, "log_level": "", "message": remainder}
        # Try "LEVEL: message" or "LEVEL message"
        colon_idx = remainder.find(":")
        if colon_idx != -1:
            left = remainder[:colon_idx].rstrip()
            right = remainder[colon_idx + 1 :].lstrip()
            last_tok = left.split()[-1] if left.split() else ""
            last_up = last_tok.strip("[]").upper()
            if last_up in LEVELS:
                return {"timestamp": ts, "log_level": last_up, "message": right}
        # Without colon, check first token as level
        first_tok, rest_no_colon = split_first_token_local(remainder)
        if first_tok and first_tok.strip("[]").upper() in LEVELS:
            return {"timestamp": ts, "log_level": first_tok.strip("[]").upper(), "message": rest_no_colon}
        # Otherwise, the rest is message
        return {"timestamp": ts, "log_level": "", "message": remainder}

    # B) "timestamp [LEVEL] : message"
    lb = s.find("[")
    rb = s.find("]", lb + 1) if lb != -1 else -1
    if lb != -1 and rb != -1:
        level_inside = s[lb + 1 : rb].strip()
        level_up = level_inside.upper()
        if level_up in LEVELS:
            ts2 = s[:lb].strip(" \t-:")
            msg2 = s[rb + 1 :].lstrip(" \t-:")
            if msg2.startswith(":"):
                msg2 = msg2[1:].lstrip()
            return {"timestamp": ts2, "log_level": level_up, "message": msg2}

    # C) "timestamp ... LEVEL: message" or "LEVEL: message"
    if ":" in s:
        idx = s.find(":")
        left = s[:idx].rstrip()
        right = s[idx + 1 :].lstrip()

        # Try to peel off a leading timestamp from the left side
        ts3, left_rem = consume_bracketed_prefix(left)
        if not ts3:
            ts3, left_rem = parse_date_time_prefix(left)
            if not ts3:
                left_rem = left

        # Find a level token near the end or start of the remaining left
        lvl_up, left_before_lvl = find_level_token(left_rem)
        if lvl_up:
            ts_final = ts3 if ts3 else left_before_lvl.strip(" \t-:")
            return {"timestamp": ts_final, "log_level": lvl_up, "message": right}
        # If we at least have a timestamp, accept message as right
        if ts3:
            return {"timestamp": ts3, "log_level": "", "message": right}

    # Fallback: use split_log_entry, then refine "LEVEL: message" pattern inside message
    ts_f, lvl_f, msg_f = split_log_entry(s)
    if not lvl_f and ":" in msg_f:
        i = msg_f.find(":")
        left = msg_f[:i].strip()
        right = msg_f[i + 1 :].strip()
        left_up = left.strip("[]").upper()
        if left_up in LEVELS:
            return {"timestamp": ts_f, "log_level": left_up, "message": right}

    return {"timestamp": ts_f, "log_level": lvl_f, "message": msg_f}


if __name__ == "__main__":
    # Simple manual tests when run as a script
    samples = [
        "[2025-09-22 10:11:12] [INFO] System started",
        "[2025-09-22 10:11:12] INFO System started",
        "[2025-09-22 10:11:12] INFO: System started",
        "2025-09-22 10:11:12 - INFO - System started",
        "2025-09-22T10:11:12Z | ERROR | Failure occurred",
        "2025-09-22 10:11:12 [DEBUG] Cache warmed",
        "2025-09-22 10:11:12 INFO System started",
        "2025-09-22 10:11:12 INFO: System started",
        "2025-09-22 10:11:12 System started without level",
        "Unstructured line with no parts",
        "INFO: Just a message without timestamp",
        "[2025-09-22 10:11:12] [WARN]: Watch out",
    ]
    for line in samples:
        parts = split_log_entry(line)
        ident = identify_log_parts(line)
        print(line, "->", parts, "||", ident)
