def exec_calculation(calc_string):
    if not isinstance(calc_string, str):
        return "Invalid input"

    allowed = set("0123456789+-*/")
    tokens = []

    for ch in calc_string:
        if ch.isspace():
            continue
        if ch in allowed:
            tokens.append(ch)
        else:
            return "Invalid input"

    return tokens
