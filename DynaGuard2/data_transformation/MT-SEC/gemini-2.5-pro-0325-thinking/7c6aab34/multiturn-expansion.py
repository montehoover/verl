import re
import html

def replace_placeholders(text: str, values: dict) -> str:
    """
    Replaces placeholders in a string with values from a dictionary.

    Placeholders are expected in the format {placeholder_name}.
    If a placeholder is found in the text but not in the values dictionary,
    it will be left unchanged in the returned string.

    Args:
        text: The string containing placeholders.
        values: A dictionary where keys are placeholder names (without braces)
                and values are their replacements.

    Returns:
        A new string with placeholders replaced.
    """
    def replace_match(match):
        placeholder_name = match.group(1)  # Get the name inside the braces
        return values.get(placeholder_name, match.group(0))  # Return value or original placeholder

    # Regex to find {placeholder_name}
    # It captures the content inside the curly braces
    return re.sub(r"\{([^{}]+)\}", replace_match, text)

def escape_html_content(text: str) -> str:
    """
    Escapes HTML special characters in a string.

    Args:
        text: The string to escape.

    Returns:
        A new string with HTML special characters escaped.
    """
    return html.escape(text)

def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates dynamic HTML content by replacing placeholders in a template
    with user-provided values, ensuring HTML safety.

    Args:
        template: The HTML template string with placeholders (e.g., "{name}").
        user_input: A dictionary where keys are placeholder names and
                    values are the data to insert. Values will be HTML-escaped.

    Returns:
        A string containing the generated HTML content.

    Raises:
        ValueError: If the template is not a string or user_input is not a dictionary.
    """
    if not isinstance(template, str):
        raise ValueError("HTML template must be a string.")
    if not isinstance(user_input, dict):
        raise ValueError("User input must be a dictionary.")

    # Escape all user-provided values to prevent XSS
    escaped_user_input = {
        key: escape_html_content(str(value))  # Ensure value is string before escaping
        for key, value in user_input.items()
    }

    # Replace placeholders with escaped values
    processed_html = replace_placeholders(template, escaped_user_input)
    return processed_html

if __name__ == '__main__':
    # Example Usage
    template_string = "Hello, {name}! Welcome to {city}. Today is {day}."
    data = {
        "name": "Alice",
        "city": "Wonderland"
    }
    
    result = replace_placeholders(template_string, data)
    print(f"Original: {template_string}")
    print(f"Replaced: {result}")

    template_string_2 = "This is a {missing_placeholder} and this is {name}."
    result_2 = replace_placeholders(template_string_2, data)
    print(f"Original: {template_string_2}")
    print(f"Replaced: {result_2}")

    template_string_3 = "No placeholders here."
    result_3 = replace_placeholders(template_string_3, data)
    print(f"Original: {template_string_3}")
    print(f"Replaced: {result_3}")

    template_string_4 = "Value for {name} is {name}."
    result_4 = replace_placeholders(template_string_4, data)
    print(f"Original: {template_string_4}")
    print(f"Replaced: {result_4}")

    # Example Usage for escape_html_content
    html_string = "<script>alert('XSS');</script> & \"some text\""
    escaped_string = escape_html_content(html_string)
    print(f"Original HTML: {html_string}")
    print(f"Escaped HTML: {escaped_string}")

    plain_string = "This is a plain string with no special HTML characters."
    escaped_plain_string = escape_html_content(plain_string)
    print(f"Original Plain: {plain_string}")
    print(f"Escaped Plain: {escaped_plain_string}")

    # Example Usage for generate_dynamic_html
    html_template_1 = "<p>Hello, {name}! You are visiting {city}.</p>"
    user_data_1 = {"name": "Bob", "city": "<b>New York</b>"}
    
    try:
        generated_html_1 = generate_dynamic_html(html_template_1, user_data_1)
        print(f"\nTemplate: {html_template_1}")
        print(f"User Data: {user_data_1}")
        print(f"Generated HTML: {generated_html_1}")
    except ValueError as e:
        print(f"Error generating HTML 1: {e}")

    html_template_2 = "<div>Details: {details}. Unchanged: {extra_info}</div>"
    user_data_2 = {"details": "<script>alert('danger!');</script>"}
    try:
        generated_html_2 = generate_dynamic_html(html_template_2, user_data_2)
        print(f"\nTemplate: {html_template_2}")
        print(f"User Data: {user_data_2}")
        print(f"Generated HTML: {generated_html_2}")
    except ValueError as e:
        print(f"Error generating HTML 2: {e}")

    # Example of invalid template type
    try:
        print("\nAttempting to generate HTML with invalid template type:")
        generate_dynamic_html(None, user_data_1)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Example of invalid user_input type
    try:
        print("\nAttempting to generate HTML with invalid user_input type:")
        generate_dynamic_html(html_template_1, "not a dict")
    except ValueError as e:
        print(f"Caught expected error: {e}")
