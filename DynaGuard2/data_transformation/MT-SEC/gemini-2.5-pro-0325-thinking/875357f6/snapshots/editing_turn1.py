import ast

def manage_script(user_script: str) -> str:
    """
    Reformats a script by appending a custom comment and checks for suspicious keywords or malformations.

    Args:
        user_script: The script string to process.

    Returns:
        The processed script string with "# Processed" appended.

    Raises:
        ValueError: If the script contains suspicious keywords ("exec", "eval")
                    or if the script is malformed (SyntaxError).
    """
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
