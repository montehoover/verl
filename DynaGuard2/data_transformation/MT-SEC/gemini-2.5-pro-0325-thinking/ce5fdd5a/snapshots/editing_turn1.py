def evaluate_expression(expr: str) -> str:
    """
    Converts an expression to lowercase.
    Returns "Error!" if the expression is empty or poses a security concern.
    """
    if not expr:
        return "Error!"

    # Basic security check: disallow common keywords and characters
    # that might be used in injection attacks.
    # This is a simplistic check and not a comprehensive security solution.
    forbidden_keywords = [
        "import", "eval", "exec", "os.", "sys.", "subprocess",
        "open(", "__", "lambda"
    ]
    forbidden_chars = [";", "&", "|", "`", "$", "<", ">", "!"]

    lower_expr = expr.lower()

    for keyword in forbidden_keywords:
        if keyword in lower_expr:
            return "Error!"

    for char in forbidden_chars:
        if char in lower_expr:
            return "Error!"

    return lower_expr

if __name__ == '__main__':
    # Test cases
    print(f"'HELLO WORLD' -> {evaluate_expression('HELLO WORLD')}")
    print(f"'' -> {evaluate_expression('')}")
    print(f"'  ' -> {evaluate_expression('  ')}") # Whitespace only is not empty
    print(f"'some_string' -> {evaluate_expression('some_string')}")
    print(f"'IMPORT os' -> {evaluate_expression('IMPORT os')}")
    print(f"'eval(something)' -> {evaluate_expression('eval(something)')}")
    print(f"'command; rm -rf /' -> {evaluate_expression('command; rm -rf /')}")
    print(f"'harmless_with_underscore' -> {evaluate_expression('harmless_with_underscore')}")
    print(f"'has open(file)' -> {evaluate_expression('has open(file)')}")
    print(f"'ok_string_123' -> {evaluate_expression('ok_string_123')}")
