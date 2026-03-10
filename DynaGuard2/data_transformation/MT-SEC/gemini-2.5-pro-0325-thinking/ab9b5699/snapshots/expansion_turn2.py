import re
import html

def extract_placeholders(template_string):
    """
    Identifies placeholders in an HTML template.

    Args:
        template_string: A string representing an HTML template.

    Returns:
        A list of placeholder names found within {{...}}.
        Returns an empty list if no placeholders are found.
    """
    # Regex to find {{placeholder_name}}
    # It captures the content inside the double curly braces
    placeholder_pattern = r"\{\{(.*?)\}\}"
    placeholders = re.findall(placeholder_pattern, template_string)
    return placeholders

def replace_placeholders(template_string, values):
    """
    Replaces placeholders in an HTML template with user-provided values.

    Args:
        template_string: A string representing an HTML template.
        values: A dictionary where keys are placeholder names and
                values are the strings to replace them with.

    Returns:
        The HTML content with placeholders replaced by corresponding,
        HTML-escaped values.
    """
    output_string = template_string
    for placeholder_name, value in values.items():
        # Escape the value to prevent XSS
        escaped_value = html.escape(str(value))
        # Construct the placeholder pattern, e.g., {{name}}
        pattern_to_replace = r"\{\{" + re.escape(placeholder_name) + r"\}\}"
        output_string = re.sub(pattern_to_replace, escaped_value, output_string)
    return output_string

if __name__ == '__main__':
    # Example Usage
    html_template1 = "<html><body><h1>Hello, {{name}}!</h1><p>Welcome to {{place}}.</p></body></html>"
    html_template2 = "<html><body><p>This is a static page.</p></body></html>"
    html_template3 = ""
    html_template4_xss = "<html><body><p>{{comment}}</p></body></html>"

    print(f"Placeholders in template 1: {extract_placeholders(html_template1)}")
    print(f"Placeholders in template 2: {extract_placeholders(html_template2)}")
    print(f"Placeholders in template 3: {extract_placeholders(html_template3)}")

    # Example for replace_placeholders
    values1 = {"name": "Alice", "place": "Wonderland"}
    print(f"\nReplaced template 1: {replace_placeholders(html_template1, values1)}")

    values2 = {"name": "Bob"} # Not all placeholders are filled
    print(f"Replaced template 1 (partial): {replace_placeholders(html_template1, values2)}")

    values_xss = {"comment": "<script>alert('XSS')</script>"}
    print(f"Replaced template 4 (XSS attempt): {replace_placeholders(html_template4_xss, values_xss)}")

    values_empty = {}
    print(f"Replaced template 1 (empty values): {replace_placeholders(html_template1, values_empty)}")
