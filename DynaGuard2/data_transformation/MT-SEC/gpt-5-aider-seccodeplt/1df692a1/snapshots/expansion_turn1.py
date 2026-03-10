def replace_placeholder(text: str, values: dict) -> str:
    """
    Replace placeholders in the form {key} within the given text using the provided values dict.
    """
    result = text
    for key, val in values.items():
        placeholder = "{" + str(key) + "}"
        result = result.replace(placeholder, str(val))
    return result
