import re
import html

def _extract_placeholders(html_template: str) -> list[str]:
    """
    Extracts unique placeholder names from an HTML template.
    Placeholders are expected in the format {{placeholder_name}}.
    """
    # Find all content within {{...}}
    raw_placeholders = re.findall(r"\{\{(.*?)\}\}", html_template)
    # Clean and deduplicate placeholder names
    # Using a set for deduplication and then converting to list
    # Stripping whitespace from each placeholder name
    return list(set(name.strip() for name in raw_placeholders))

def _substitute_placeholder(current_template: str, placeholder_name: str, value_to_insert: any) -> str:
    """
    Substitutes a single placeholder in the template with its HTML-escaped value.
    """
    escaped_value = html.escape(str(value_to_insert))
    # Construct the full placeholder string, e.g., {{placeholder_name}}
    # Note: The placeholder_name itself is already stripped by _extract_placeholders
    full_placeholder_regex = r"\{\{\s*" + re.escape(placeholder_name) + r"\s*\}\}"
    return re.sub(full_placeholder_regex, escaped_value, current_template)

def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produces dynamic HTML content by replacing placeholders in a given template
    with corresponding user-provided values.

    Args:
        html_template: An HTML template string containing placeholders
                       delimited with {{...}}.
        user_values: A mapping of placeholder names to the values to be inserted.

    Returns:
        The HTML content generated after placeholder replacement.

    Raises:
        ValueError: Raised when any error occurs during template processing
                    or if a placeholder is missing.
    """
    try:
        placeholders = _extract_placeholders(html_template)
        
        processed_template = html_template

        for name in placeholders:
            # The 'name' from _extract_placeholders is already stripped.
            if name not in user_values:
                raise ValueError(f"Missing value for placeholder: {name}")
            
            value = user_values[name]
            processed_template = _substitute_placeholder(processed_template, name, value)
            
        return processed_template

    except ValueError: # Re-raise ValueError directly (e.g., from missing placeholder)
        raise
    except Exception as e:
        # Catch any other exceptions during processing and re-raise as ValueError
        raise ValueError(f"Error during template processing: {e}")
