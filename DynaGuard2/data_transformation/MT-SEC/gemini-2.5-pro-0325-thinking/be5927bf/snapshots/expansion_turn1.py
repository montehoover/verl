import re

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
