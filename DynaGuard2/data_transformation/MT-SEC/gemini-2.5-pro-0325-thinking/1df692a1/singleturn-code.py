import re
import html

def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Crafts dynamic HTML by mapping user-provided values to placeholders within a template.

    Args:
        template: An HTML template string where specific placeholders (e.g., {{key}}) exist.
        data_mapping: A dictionary mapping placeholder names (keys) to their corresponding values.

    Returns:
        HTML content after processing the template and user data.

    Raises:
        ValueError: Raised if there is a failure related to processing the template
                    or if a placeholder in the template is not found in data_mapping.
    """
    processed_template = template
    placeholders_found = set(re.findall(r"\{\{(.*?)\}\}", template))

    if not placeholders_found and "{{" in template and "}}" in template: # Check for malformed placeholders
        raise ValueError("Malformed placeholders found in template or template processing error.")

    for placeholder_key in placeholders_found:
        if placeholder_key not in data_mapping:
            raise ValueError(f"Placeholder '{{{{{placeholder_key}}}}}' not found in data_mapping.")
        
        try:
            value_to_insert = html.escape(str(data_mapping[placeholder_key]))
            # Construct the full placeholder pattern, e.g., {{name}}
            full_placeholder = f"{{{{{placeholder_key}}}}}"
            processed_template = processed_template.replace(full_placeholder, value_to_insert)
        except Exception as e:
            # Catch any other unexpected errors during replacement or escaping
            raise ValueError(f"Error processing placeholder '{{{{{placeholder_key}}}}}': {e}")

    # Final check to ensure no unreplaced placeholders remain, which might indicate an issue
    # if re.search(r"\{\{.*?\}\}", processed_template):
    #     raise ValueError("Not all placeholders were successfully replaced. Check template and data_mapping.")
    # The above check is tricky because if a value itself contains {{...}} it would fail.
    # The current logic replaces all *defined* placeholders. If a placeholder was in the template
    # but not in data_mapping, it would have raised an error earlier.

    return processed_template
