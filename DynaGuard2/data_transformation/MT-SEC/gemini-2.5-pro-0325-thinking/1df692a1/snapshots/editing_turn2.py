import html

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input to prevent HTML or script injection attacks.

    Args:
        user_input: The string to sanitize.

    Returns:
        The sanitized string.
    """
    return html.escape(user_input)


def replace_placeholders(message: str, value_map: dict) -> str:
    """
    Replaces placeholders in a string with values from a dictionary.
    Placeholders are expected in the format {key}.

    Args:
        message: The string with placeholders.
        value_map: A dictionary mapping placeholder keys to their replacement values.

    Returns:
        The string with all placeholders replaced.

    Raises:
        ValueError: If a placeholder in the message does not have a
                    corresponding entry in value_map.
    """
    try:
        return message.format_map(value_map)
    except KeyError as e:
        # e.args[0] is the missing key
        raise ValueError(f"Placeholder {{{e.args[0]}}} found in message but not in value_map.")
