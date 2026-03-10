from typing import Iterable


def evaluate_expression(expr: str) -> str:
    """
    Replace every numeric digit in the input string with an asterisk '*'.
    Allow only alphanumeric characters and spaces; raise ValueError otherwise.

    Examples:
        "abc 123" -> "abc ***"
        "NoDigits" -> "NoDigits"

    :param expr: Input string to process.
    :return: Transformed string with digits replaced by '*'.
    :raises ValueError: If any character is not alphanumeric or a space.
    """
    result_chars = []
    for ch in expr:
        if ch.isalnum() or ch == " ":
            result_chars.append('*' if ch.isdigit() else ch)
        else:
            raise ValueError("Input contains invalid characters; only alphanumeric characters and spaces are allowed.")
    return ''.join(result_chars)
