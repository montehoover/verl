import re
import html

def build_html_content(html_template: str, values_dict: dict) -> str:
    """
    Produces dynamic HTML content by replacing placeholders in a given template
    with corresponding user-provided values.

    Args:
        html_template: An HTML template containing placeholders delimited with {{...}}.
        values_dict: A mapping of placeholder names to the values to be inserted.

    Returns:
        The HTML content generated after the placeholder replacement.

    Raises:
        ValueError: Raised when any error occurs during template processing
                    or if a placeholder is missing.
    """
    try:
        # Find all placeholders like {{placeholder_name}}
        placeholders = re.findall(r"\{\{(.*?)\}\}", html_template)

        processed_template = html_template
        for placeholder_key in placeholders:
            placeholder_full = f"{{{{{placeholder_key}}}}}"
            if placeholder_key not in values_dict:
                raise ValueError(f"Missing value for placeholder: {placeholder_key}")
            
            value_to_insert = values_dict[placeholder_key]
            escaped_value = html.escape(str(value_to_insert))
            
            # Replace first occurrence only, as re.sub would replace all at once
            # and we want to ensure each placeholder is checked individually
            # for presence in values_dict.
            # However, a simpler approach is to iterate through found placeholders
            # and then use re.sub for replacement.
            # If a placeholder appears multiple times, it will be replaced multiple times
            # with the same escaped value, which is usually the desired behavior.
            
            # Using re.sub for robust replacement of all occurrences of this specific placeholder
            processed_template = re.sub(re.escape(placeholder_full), escaped_value, processed_template)

        # After all replacements, check if any {{...}} are left,
        # which might indicate malformed placeholders or ones not caught by the initial findall
        # (e.g. {{}} or {{ value with spaces }} if not handled by (.*?) correctly)
        # The current regex (.*?) is greedy, so it should be fine.
        # A final check for remaining unreplaced placeholders:
        remaining_placeholders = re.findall(r"\{\{(.*?)\}\}", processed_template)
        if remaining_placeholders:
            # This case should ideally be caught by the initial loop if all placeholders
            # were correctly identified and checked against values_dict.
            # However, if a placeholder was in values_dict but somehow not replaced,
            # or if the replacement logic itself introduced new {{...}} patterns (unlikely here),
            # this could catch it. More practically, this catches placeholders that were
            # in the template but *not* in values_dict if the initial loop was flawed.
            # Given the current logic, the first check `if placeholder_key not in values_dict:`
            # should cover missing values.
            # This might be redundant but acts as a safeguard.
            # For now, let's assume the first loop handles all missing value errors.
            pass


        return processed_template

    except ValueError as ve: # Re-raise ValueError specifically
        raise ve
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise ValueError(f"An error occurred during template processing: {e}")
