import re
import html

def extract_placeholders(template: str) -> set:
    """
    Extract all placeholder names from the template.
    
    Args:
        template: str, an HTML template with placeholders
    
    Returns:
        set of placeholder names found in the template
    """
    placeholder_pattern = r'\{\{?\s*(\w+)\s*\}?\}'
    return set(re.findall(placeholder_pattern, template))


def validate_placeholder_mapping(placeholders: set, data_mapping: dict) -> None:
    """
    Validate that all placeholders have corresponding values in the data mapping.
    
    Args:
        placeholders: set of placeholder names
        data_mapping: dict mapping placeholder names to values
    
    Raises:
        ValueError: if any placeholders are missing from the data mapping
    """
    missing_placeholders = placeholders - set(data_mapping.keys())
    if missing_placeholders:
        raise ValueError(f"Incomplete placeholder mapping. Missing values for: {', '.join(sorted(missing_placeholders))}")


def sanitize_and_replace_placeholders(template: str, data_mapping: dict) -> str:
    """
    Replace placeholders in the template with sanitized values from the data mapping.
    
    Args:
        template: str, an HTML template with placeholders
        data_mapping: dict mapping placeholder names to values
    
    Returns:
        str, the template with all placeholders replaced by sanitized values
    """
    placeholder_pattern = r'\{\{?\s*(\w+)\s*\}?\}'
    
    def replace_match(match):
        placeholder_name = match.group(1)
        if placeholder_name in data_mapping:
            return html.escape(str(data_mapping[placeholder_name]))
        return match.group(0)
    
    return re.sub(placeholder_pattern, replace_match, template)


def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Crafts dynamic HTML by mapping user-provided values to placeholders within a template.
    
    Args:
        template: str, an HTML template where specific placeholders exist.
        data_mapping: dict, a mapping of placeholder names and their corresponding values.
    
    Returns:
        str, returns HTML content after processing the template and user data.
    
    Raises:
        ValueError: Raised if there is failure related to processing the template or if the placeholder mapping is incomplete.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    
    if not isinstance(data_mapping, dict):
        raise ValueError("Data mapping must be a dictionary")
    
    # Extract placeholders from template
    placeholders = extract_placeholders(template)
    
    # Validate that all placeholders have corresponding values
    validate_placeholder_mapping(placeholders, data_mapping)
    
    # Replace placeholders with sanitized values
    return sanitize_and_replace_placeholders(template, data_mapping)
