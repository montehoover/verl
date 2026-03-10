import re

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

if __name__ == '__main__':
    # Example Usage
    html_template1 = "<html><body><h1>Hello, {{name}}!</h1><p>Welcome to {{place}}.</p></body></html>"
    html_template2 = "<html><body><p>This is a static page.</p></body></html>"
    html_template3 = ""

    print(f"Placeholders in template 1: {extract_placeholders(html_template1)}")
    print(f"Placeholders in template 2: {extract_placeholders(html_template2)}")
    print(f"Placeholders in template 3: {extract_placeholders(html_template3)}")
