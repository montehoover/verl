import re
import html

def _extract_placeholders(template: str) -> set[str]:
    """
    Extracts all placeholder names from a template string.
    Placeholders are expected in the format {{placeholder_name}}.

    Args:
        template: The template string.

    Returns:
        A set of placeholder names found in the template.
    """
    return set(re.findall(r"\{\{(\w+)\}\}", template))

def _replace_placeholders_in_template(template: str, data_mapping: dict) -> str:
    """
    Replaces placeholders in a template with values from data_mapping, sanitizing inputs.

    Args:
        template: The HTML template string with placeholders.
        data_mapping: A dictionary mapping placeholder names to their values.

    Returns:
        The template string with placeholders replaced by sanitized values.
    
    Raises:
        ValueError: If a regex error occurs during replacement.
    """
    processed_template = template
    try:
        for placeholder_name, value in data_mapping.items():
            escaped_value = html.escape(str(value))
            pattern = r"\{\{" + re.escape(placeholder_name) + r"\}\}"
            processed_template = re.sub(pattern, escaped_value, processed_template)
    except re.error as e:
        raise ValueError(f"Regex error during template placeholder replacement: {e}")
    return processed_template

def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Crafts dynamic HTML by mapping user-provided values to placeholders within a template.

    Args:
        template: An HTML template where specific placeholders exist.
                  Placeholders should be in the format {{placeholder_name}}.
        data_mapping: A mapping of placeholder names and their corresponding values.

    Returns:
        HTML content after processing the template and user data.

    Raises:
        ValueError: Raised if there is failure related to processing the template
                    or if the placeholder mapping is incomplete.
    """
    placeholders_in_template = _extract_placeholders(template)
    
    # Validate that all placeholders in the template have corresponding data
    missing_placeholders = placeholders_in_template - set(data_mapping.keys())
    if missing_placeholders:
        raise ValueError(f"Incomplete placeholder mapping. Missing keys: {', '.join(missing_placeholders)}")

    try:
        # Perform the replacement using the helper function
        # Only pass relevant data_mapping entries to avoid processing unused keys,
        # though _replace_placeholders_in_template would ignore them anyway.
        # For clarity, we could filter data_mapping, but it's not strictly necessary
        # if _replace_placeholders_in_template iterates its provided data_mapping.
        # The current _replace_placeholders_in_template iterates all given data_mapping keys.
        processed_template = _replace_placeholders_in_template(template, data_mapping)
    except ValueError as e: # Catch regex errors from the helper
        raise ValueError(f"Failed to process template due to placeholder replacement error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise ValueError(f"Failed to process template: {e}")

    return processed_template
