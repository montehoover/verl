def replace_placeholder(text: str, replacements: dict) -> str:
    """
    Replaces placeholders in a string with corresponding values from a dictionary.

    Args:
        text: The string containing placeholders.
        replacements: A dictionary where keys are placeholders (without delimiters)
                      and values are their replacements. Assumes placeholders in
                      the text are formatted like "{key}".

    Returns:
        The string with placeholders replaced.
    """
    for placeholder, value in replacements.items():
        text = text.replace(f"{{{placeholder}}}", str(value))
    return text

if __name__ == '__main__':
    # Example usage:
    template_string = "Hello, {name}! Welcome to {city}."
    data = {"name": "Alice", "city": "Wonderland"}
    result = replace_placeholder(template_string, data)
    print(f"Original: {template_string}")
    print(f"Processed: {result}")

    template_string_2 = "My favorite numbers are {num1} and {num2}."
    data_2 = {"num1": 7, "num2": 42}
    result_2 = replace_placeholder(template_string_2, data_2)
    print(f"Original: {template_string_2}")
    print(f"Processed: {result_2}")

    template_string_3 = "This has no placeholders."
    data_3 = {"unused": "value"}
    result_3 = replace_placeholder(template_string_3, data_3)
    print(f"Original: {template_string_3}")
    print(f"Processed: {result_3}")

    template_string_4 = "This has a {missing_placeholder} and a {present_placeholder}."
    data_4 = {"present_placeholder": "value_here"}
    result_4 = replace_placeholder(template_string_4, data_4)
    print(f"Original: {template_string_4}")
    print(f"Processed: {result_4}")
