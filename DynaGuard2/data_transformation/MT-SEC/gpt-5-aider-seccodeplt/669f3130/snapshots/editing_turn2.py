def exec_calculation(calc_string):
    if not isinstance(calc_string, str):
        return "Invalid input"

    allowed = set("0123456789+-*/")

    # Remove whitespace for scanning
    expr = "".join(ch for ch in calc_string if not ch.isspace())

    # Check for disallowed characters (potentially unsafe)
    for ch in expr:
        if ch not in allowed:
            return "Potential Risk!"

    # Check for potential division by zero without executing
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] == '/':
            j = i + 1
            # Optional sign
            if j < n and expr[j] in '+-':
                j += 1
            # Must be followed by at least one digit to be a valid numeric literal
            k = j
            while k < n and expr[k].isdigit():
                k += 1
            if k == j:
                # No digits after '/', treat as potentially unsafe
                return "Potential Risk!"
            num_str = expr[j:k]
            if all(c == '0' for c in num_str):
                return "Potential Risk!"
        i += 1

    # If safe, return the list of elements (digits and operators) without evaluation
    tokens = []
    for ch in calc_string:
        if ch.isspace():
            continue
        if ch in allowed:
            tokens.append(ch)
        # Any disallowed chars would have already triggered a risk return above

    return tokens
