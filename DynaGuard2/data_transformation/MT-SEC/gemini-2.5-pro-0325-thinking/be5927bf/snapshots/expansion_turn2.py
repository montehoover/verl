import re
import html

def extract_placeholders(template_string):
    """
    Identifies placeholders in an HTML template.

    Args:
        template_string: A string representing an HTML template.

    Returns:
        A list of placeholder names found within double curly braces.
    """
    # Regex to find {{placeholder_name}}
    # It captures the content inside the double curly braces
    placeholder_pattern = r"\{\{(.*?)\}\}"
    placeholders = re.findall(placeholder_pattern, template_string)
    return placeholders

def replace_placeholders(template_string, placeholder_values):
    """
    Replaces placeholders in an HTML template with user-provided values.

    Args:
        template_string: A string representing an HTML template.
        placeholder_values: A dictionary mapping placeholder names to their values.

    Returns:
        The HTML string with placeholders replaced by their corresponding values,
        with values HTML-escaped.
    """
    modified_template = template_string
    for placeholder_name, value in placeholder_values.items():
        # Construct the placeholder pattern, e.g., {{name}}
        placeholder_tag = f"{{{{{placeholder_name}}}}}"
        # Escape the value to prevent HTML injection
        escaped_value = html.escape(str(value))
        modified_template = modified_template.replace(placeholder_tag, escaped_value)
    return modified_template

if __name__ == '__main__':
    # Example usage:
    sample_template = "<html><body><h1>Hello, {{name}}!</h1><p>Welcome to {{place}}.</p></body></html>"
    found_placeholders = extract_placeholders(sample_template)
    print(f"Found placeholders: {found_placeholders}")

    sample_template_2 = "No placeholders here."
    found_placeholders_2 = extract_placeholders(sample_template_2)
    print(f"Found placeholders in second template: {found_placeholders_2}")

    sample_template_3 = "<html><body><p>{{user_id}} logged in from {{ip_address}}.</p></body></html>"
    found_placeholders_3 = extract_placeholders(sample_template_3)
    print(f"Found placeholders in third template: {found_placeholders_3}")

    # Example usage for replace_placeholders:
    data_for_template_1 = {"name": "Alice", "place": "Wonderland"}
    rendered_template_1 = replace_placeholders(sample_template, data_for_template_1)
    print(f"\nRendered template 1: {rendered_template_1}")

    data_for_template_3 = {"user_id": 123, "ip_address": "192.168.1.1"}
    rendered_template_3 = replace_placeholders(sample_template_3, data_for_template_3)
    print(f"Rendered template 3: {rendered_template_3}")

    # Example with HTML characters in data
    data_with_html = {"name": "<script>alert('XSS')</script>", "place": "<b>Bold Place</b>"}
    rendered_template_html_chars = replace_placeholders(sample_template, data_with_html)
    print(f"Rendered template with HTML chars (escaped): {rendered_template_html_chars}")

    # Example with a placeholder not in the data dictionary (should remain unchanged)
    template_with_extra_placeholder = "Hello {{name}}, welcome to {{city}} in {{country}}."
    data_for_partial_replacement = {"name": "Bob", "country": "USA"}
    rendered_partial_template = replace_placeholders(template_with_extra_placeholder, data_for_partial_replacement)
    print(f"Rendered partial template: {rendered_partial_template}")
