import re
import html

def parse_placeholders(html_template: str) -> list:
    """Extract all placeholders from the template."""
    placeholder_pattern = r'\{\{([^}]+)\}\}'
    return [(match.group(0), match.group(1).strip()) 
            for match in re.finditer(placeholder_pattern, html_template)]

def validate_placeholders(placeholders: list, values_dict: dict) -> None:
    """Validate that all placeholders have corresponding values."""
    for _, placeholder_name in placeholders:
        if placeholder_name not in values_dict:
            raise ValueError(f"Missing value for placeholder: {placeholder_name}")

def replace_placeholders(html_template: str, placeholders: list, values_dict: dict) -> str:
    """Replace placeholders in template with escaped values."""
    result = html_template
    for placeholder_with_braces, placeholder_name in placeholders:
        if placeholder_name in values_dict:
            escaped_value = html.escape(str(values_dict[placeholder_name]))
            result = result.replace(placeholder_with_braces, escaped_value)
    return result

def build_html_content(html_template: str, values_dict: dict) -> str:
    try:
        # Pipeline pattern: parse -> validate -> replace
        placeholders = parse_placeholders(html_template)
        validate_placeholders(placeholders, values_dict)
        result = replace_placeholders(html_template, placeholders, values_dict)
        return result
        
    except Exception as e:
        raise ValueError(f"Error processing template: {str(e)}")
