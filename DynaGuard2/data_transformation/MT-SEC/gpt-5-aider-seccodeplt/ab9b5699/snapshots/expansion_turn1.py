import re

def extract_placeholders(template):
    """
    Extract placeholder names contained within double curly braces from a template string.

    Examples:
        "Hello, {{name}}!" -> ["name"]
        "<div>{{ user.name }}</div>{{id}}" -> ["user.name", "id"]

    Args:
        template (str): The HTML template string.

    Returns:
        list: List of placeholder names found; empty list if none.
    """
    if not template:
        return []
    pattern = re.compile(r"{{\s*([A-Za-z_][\w\.-]*)\s*}}")
    return pattern.findall(template)
