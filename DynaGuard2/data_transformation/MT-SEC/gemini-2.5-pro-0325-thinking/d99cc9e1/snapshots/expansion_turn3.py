import re

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


def split_email_parts(email_address: str) -> tuple[str, str]:
    """
    Splits an email address into its local part and domain.

    Args:
        email_address: The email string to split.

    Returns:
        A tuple containing the local part and the domain.
        Returns (email_address, "") if "@" is not found.
    """
    if "@" in email_address:
        local_part, domain = email_address.split("@", 1)
        return local_part, domain
    return email_address, ""


def retrieve_email_domain(email_str: str) -> str | None:
    """
    Extracts the domain part of an email address using regular expressions.

    Args:
        email_str: The email string to parse.

    Returns:
        The domain as a string if the email is well-formed, or None otherwise.
    """
    # A common regex for email validation, focusing on capturing the domain part.
    # This regex is a simplified version for demonstration and might not cover all valid email edge cases.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    match = re.fullmatch(email_regex, email_str)
    if match:
        return match.group(1)  # Group 1 captures the domain part
    return None
