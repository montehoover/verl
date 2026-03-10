def check_keyword_in_string(text: str, keyword: str) -> bool:
    """
    Checks if a keyword is present anywhere in a given text.

    Args:
        text: The string to search within.
        keyword: The string to search for.

    Returns:
        True if the keyword is found in the text, False otherwise.
    """
    return keyword in text
