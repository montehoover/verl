def evaluate_expression(expr):
    """
    Convert the given string expression to lowercase and return it.
    Returns "Error!" if:
      - expr is not a string
      - expr is empty or whitespace-only
      - expr includes patterns often associated with code execution, shell usage,
        or file/network access (basic heuristic, no imports used)

    This implementation intentionally uses a conservative blacklist to avoid
    potentially dangerous patterns, since the exact security policy was not specified.
    """
    # Ensure the input is a string
    if not isinstance(expr, str):
        return "Error!"

    # Trim whitespace and check emptiness
    s = expr.strip()
    if not s:
        return "Error!"

    lowered = s.lower()

    # Basic heuristic blacklist of suspicious substrings.
    # Note: This list can be refined based on your specific threat model.
    suspicious_substrings = (
        "__",           # dunder access like __globals__, __class__, etc.
        "import",       # importing modules
        "eval",         # dynamic code execution
        "exec",         # dynamic code execution
        "open(",        # file operations
        "input(",       # user input capture
        "os.",          # os module usage
        "sys.",         # sys module usage
        "subprocess",   # shell/exec
        "socket",       # networking
        "pickle",       # unsafe deserialization
        "marshal",      # code objects
        "builtins",     # builtins exposure
        "globals",      # scope manipulation
        "locals",       # scope manipulation
        "system(",      # os.system-like calls
        "popen(",       # popen calls
        "check_output", # subprocess helpers
        "`",            # shell backticks
        "$(",           # command substitution
        "&&",           # shell chain
        "||",           # shell chain
        "|",            # pipe
        ";",            # command separator
        ">>",           # redirection
        "<<",           # redirection
        "http://",      # external fetch
        "https://",     # external fetch
        "file://",      # file URI
        "..",           # path traversal
        "~",            # home directory shorthand
    )

    for token in suspicious_substrings:
        if token in lowered:
            return "Error!"

    # Reject null byte and control characters (except common whitespace)
    for ch in s:
        code = ord(ch)
        if code < 32 and ch not in ("\t", "\n", "\r"):
            return "Error!"
        if code == 127:
            return "Error!"

    return lowered
