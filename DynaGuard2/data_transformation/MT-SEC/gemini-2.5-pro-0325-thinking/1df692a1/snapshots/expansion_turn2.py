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

def format_template(template_string: str, data: dict) -> str:
    """
    Replaces named placeholders in a template string with values from a dictionary.

    Placeholders are expected in the format {{key}}.
    If a key in a placeholder is not found in the data dictionary,
    the placeholder remains in the string.

    This function leverages the replace_placeholder function by transforming
    the keys in the data dictionary to make replace_placeholder search for
    double-braced placeholders.

    Args:
        template_string: The string containing placeholders like {{key}}.
        data: A dictionary where keys are placeholder names (without double braces)
              and values are their replacements.

    Returns:
        The string with placeholders replaced.
    """
    transformed_data = {}
    for key, value in data.items():
        # Transform original key "name" into "{name}" for replace_placeholder.
        # replace_placeholder takes a placeholder like "p_key" and searches for "{p_key}".
        # So, if we give it "{name}" as p_key, it will search for "{{name}}".
        transformed_data[f"{{{key}}}"] = value
    
    return replace_placeholder(template_string, transformed_data)

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

    print("\n--- format_template examples ---")
    # Example usage for format_template:
    template_ft_1 = "Hello, {{name}}! You are in {{city}}."
    data_ft_1 = {"name": "Bob", "city": "New York"}
    result_ft_1 = format_template(template_ft_1, data_ft_1)
    print(f"Original: {template_ft_1}")
    print(f"Processed: {result_ft_1}") # Expected: "Hello, Bob! You are in New York."

    template_ft_2 = "Data: {{item1}}, {{item2}}, and {{missing_item}}."
    data_ft_2 = {"item1": "Apple", "item2": 123}
    result_ft_2 = format_template(template_ft_2, data_ft_2)
    print(f"Original: {template_ft_2}")
    print(f"Processed: {result_ft_2}") # Expected: "Data: Apple, 123, and {{missing_item}}."

    template_ft_3 = "No placeholders here."
    data_ft_3 = {"name": "Test"}
    result_ft_3 = format_template(template_ft_3, data_ft_3)
    print(f"Original: {template_ft_3}")
    print(f"Processed: {result_ft_3}") # Expected: "No placeholders here."

    template_ft_4 = "Item: {{item}}, Price: {{price}}."
    data_ft_4 = {"item": "Book", "price": 15.99, "currency": "USD"} # currency is extra
    result_ft_4 = format_template(template_ft_4, data_ft_4)
    print(f"Original: {template_ft_4}")
    print(f"Processed: {result_ft_4}") # Expected: "Item: Book, Price: 15.99."
