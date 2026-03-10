import re
import html
import logging

# Configure logger
logger = logging.getLogger(__name__)
# Example basic configuration:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# The user can configure this further as needed.

def _extract_placeholders(template: str) -> set[str]:
    """
    Extracts all placeholder names from a template string.
    Placeholders are expected in the format {{placeholder_name}}.

    Args:
        template: The template string.

    Returns:
        A set of placeholder names found in the template.
    """
    logger.debug(f"Extracting placeholders from template: {template[:50]}...") # Log snippet of template
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", template))
    logger.info(f"Found placeholders: {placeholders}")
    return placeholders

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
    logger.debug(f"Starting placeholder replacement. Data mapping keys: {list(data_mapping.keys())}")
    try:
        for placeholder_name, value in data_mapping.items():
            escaped_value = html.escape(str(value))
            pattern = r"\{\{" + re.escape(placeholder_name) + r"\}\}"
            # Check if placeholder actually exists in the current state of processed_template
            # to avoid logging replacements that don't happen (e.g. if data_mapping has extra keys)
            if re.search(pattern, processed_template):
                processed_template = re.sub(pattern, escaped_value, processed_template)
                logger.debug(f"Replaced '{{{{{placeholder_name}}}}}' with escaped value.")
            else:
                logger.debug(f"Placeholder '{{{{{placeholder_name}}}}}' not found in current template state for replacement.")
    except re.error as e:
        logger.error(f"Regex error during placeholder replacement for '{placeholder_name}': {e}")
        raise ValueError(f"Regex error during template placeholder replacement: {e}")
    logger.info("Placeholder replacement process completed.")
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
    logger.info(f"Starting HTML creation from template. Data mapping keys: {list(data_mapping.keys())}")
    placeholders_in_template = _extract_placeholders(template)
    
    # Validate that all placeholders in the template have corresponding data
    missing_placeholders = placeholders_in_template - set(data_mapping.keys())
    if missing_placeholders:
        error_message = f"Incomplete placeholder mapping. Missing keys: {', '.join(missing_placeholders)}"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        logger.debug("All required placeholders found in data_mapping. Proceeding with replacement.")
        # Perform the replacement using the helper function
        processed_template = _replace_placeholders_in_template(template, data_mapping)
    except ValueError as e: # Catch regex errors from the helper
        logger.error(f"Error during placeholder replacement: {e}")
        raise ValueError(f"Failed to process template due to placeholder replacement error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"Unexpected error during template processing: {e}", exc_info=True)
        raise ValueError(f"Failed to process template: {e}")

    logger.info("HTML creation from template completed successfully.")
    return processed_template
