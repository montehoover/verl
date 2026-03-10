from typing import Final


def check_keyword_in_string(text: str, keyword: str) -> bool:
    """
    Return True if `keyword` is present anywhere in `text`, otherwise False.

    Notes:
    - The check is case-sensitive.
    - An empty `keyword` returns False to avoid treating the empty string as present in any text.
    """
    if not isinstance(text, str) or not isinstance(keyword, str):
        raise TypeError("Both 'text' and 'keyword' must be strings.")

    if keyword == "":
        return False

    return keyword in text
