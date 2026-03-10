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
