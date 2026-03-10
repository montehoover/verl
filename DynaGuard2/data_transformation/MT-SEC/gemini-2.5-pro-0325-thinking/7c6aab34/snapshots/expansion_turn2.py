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
