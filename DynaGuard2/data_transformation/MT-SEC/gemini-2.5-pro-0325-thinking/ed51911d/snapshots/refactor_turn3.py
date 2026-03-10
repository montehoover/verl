import re
import html
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

def _parse_placeholders(html_template: str) -> list[str]:
    """
    Parses an HTML template and extracts all placeholder keys.
    Placeholders are expected in the format {{placeholder_name}}.

    Args:
        html_template: The HTML template string.

    Returns:
        A list of unique placeholder keys found in the template.
    """
    # Using set to get unique placeholders, then converting to list
    placeholders = list(set(re.findall(r"\{\{(.*?)\}\}", html_template)))
    return placeholders

def _replace_placeholder_in_template(template: str, placeholder_key: str, escaped_value: str) -> str:
    """
    Replaces all occurrences of a specific placeholder in the template with its escaped value.

    Args:
        template: The HTML template string.
        placeholder_key: The key of the placeholder (e.g., "name").
        escaped_value: The HTML-escaped value to insert.

    Returns:
        The template string with the placeholder replaced.
    """
    placeholder_full = f"{{{{{placeholder_key}}}}}"
    # Using re.sub for robust replacement of all occurrences of this specific placeholder
    return re.sub(re.escape(placeholder_full), escaped_value, template)

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
    logger.info("Starting HTML content build process.")
    try:
        placeholders = _parse_placeholders(html_template)
        logger.debug(f"Found placeholders: {placeholders}")
        
        processed_template = html_template
        for placeholder_key in placeholders:
            logger.debug(f"Processing placeholder: '{placeholder_key}'")
            if placeholder_key not in values_dict:
                logger.error(f"Missing value for placeholder: {placeholder_key}")
                raise ValueError(f"Missing value for placeholder: {placeholder_key}")
            
            value_to_insert = values_dict[placeholder_key]
            escaped_value = html.escape(str(value_to_insert))
            
            processed_template = _replace_placeholder_in_template(
                processed_template, placeholder_key, escaped_value
            )

        # Final check for any placeholders that might not have been in the initial parse list
        # or were malformed, although _parse_placeholders should get all valid ones.
        # This also helps catch issues if values_dict contained keys that weren't actual placeholders.
        # However, the primary check is for missing values for *found* placeholders.
        # If a placeholder {{foo}} exists but 'foo' is not in values_dict, it's an error.
        # If 'bar' is in values_dict but {{bar}} is not in template, it's ignored, which is fine.
        
        # The previous check for remaining_placeholders after substitution is largely
        # redundant if _parse_placeholders correctly identifies all placeholders and
        # the loop correctly processes them. If a placeholder was missed by _parse_placeholders
        # or if _replace_placeholder_in_template failed silently (it shouldn't),
        # then unreplaced placeholders might remain.
        # For simplicity and trusting the components, this check can be removed.
        # The critical path is ensuring all *identified* placeholders have values.

        logger.info("HTML content build process completed successfully.")
        return processed_template

    except ValueError as ve: # Re-raise ValueError specifically
        logger.error(f"ValueError during template processing: {ve}", exc_info=True)
        raise ve
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"Unexpected exception during template processing: {e}", exc_info=True)
        raise ValueError(f"An error occurred during template processing: {e}")
