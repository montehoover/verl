import ast

def manage_script(user_script: str) -> str:
    """
    Reformats a script by appending a custom comment, checks for suspicious keywords,
    enforces a maximum length, and validates script integrity.

    Args:
        user_script: The script string to process.

    Returns:
        The processed script string with "# Processed" appended.

    Raises:
        ValueError: If the script contains suspicious keywords ("exec", "eval"),
                    if the script exceeds 100 characters,
                    or if the script is malformed (SyntaxError).
    """
    MAX_SCRIPT_LENGTH = 100
    if len(user_script) > MAX_SCRIPT_LENGTH:
        raise ValueError(f"Script exceeds maximum length of {MAX_SCRIPT_LENGTH} characters.")

    suspicious_keywords = ["exec", "eval"]
    for keyword in suspicious_keywords:
        if keyword in user_script:
            raise ValueError(f"Script contains suspicious keyword: {keyword}")

    try:
        # Attempt to parse the script to check for basic well-formedness
        # compile() can also be used here, ast.parse is often preferred for static analysis
        ast.parse(user_script)
    except SyntaxError as e:
        raise ValueError(f"Script is malformed: {e}") from e

    processed_script = user_script.strip()
    if processed_script:  # Add newline only if script is not empty
        processed_script += "\n"
    processed_script += "# Processed"
    return processed_script
