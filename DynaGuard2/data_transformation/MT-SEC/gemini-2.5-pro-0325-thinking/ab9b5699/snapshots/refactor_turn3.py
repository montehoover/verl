import re
import html
import logging

# Configure basic logging
# For a real application, you might want to configure this in a more centralized way
# (e.g., in your application's entry point) with handlers, formatters, etc.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_placeholders(html_template: str) -> list[str]:
    """
    Extracts unique, stripped placeholder names from an HTML template.

    Placeholders are expected in the format {{placeholder_name}}, where
    placeholder_name can be surrounded by optional whitespace which will be
    stripped.

    Args:
        html_template: The HTML template string to parse.

    Returns:
        A list of unique placeholder names found in the template,
        with leading/trailing whitespace removed from each name.
        Returns an empty list if no placeholders are found.
    """
    logger.debug("Attempting to extract placeholders from template.")
    # Find all content within {{...}}
    raw_placeholders = re.findall(r"\{\{(.*?)\}\}", html_template)
    # Clean (strip whitespace) and deduplicate placeholder names
    # Using a set for deduplication and then converting to list
    extracted_names = list(set(name.strip() for name in raw_placeholders))
    logger.debug(f"Extracted placeholder names: {extracted_names}")
    return extracted_names


def _substitute_placeholder(
    current_template: str, placeholder_name: str, value_to_insert: any
) -> str:
    """
    Substitutes a single placeholder in the template with its HTML-escaped value.

    The placeholder matching is case-sensitive and accounts for optional
    whitespace within the {{...}} delimiters around the placeholder name.

    Args:
        current_template: The current state of the HTML template string.
        placeholder_name: The exact name of the placeholder to replace (already stripped).
        value_to_insert: The value to insert for the placeholder. This value
                         will be converted to a string and HTML-escaped.

    Returns:
        The template string with the specified placeholder replaced by its
        HTML-escaped value.
    """
    logger.debug(
        f"Substituting placeholder '{{{{{placeholder_name}}}}}' "
        f"with value: '{value_to_insert}'"
    )
    escaped_value = html.escape(str(value_to_insert))
    # Construct the regex to match {{ placeholder_name }} allowing for spaces
    # The placeholder_name itself is assumed to be already stripped.
    # re.escape is used to ensure placeholder_name is treated literally if it contains regex special chars.
    full_placeholder_regex = r"\{\{\s*" + re.escape(placeholder_name) + r"\s*\}\}"
    
    substituted_template = re.sub(full_placeholder_regex, escaped_value, current_template)
    if substituted_template == current_template and f"{{{{{placeholder_name}}}}}" not in current_template:
        # This check is a bit tricky because the regex allows for spaces.
        # A more robust check might involve re-checking with the regex if a substitution was expected.
        # For now, we log if the template didn't change, which might indicate an issue if a change was expected.
        logger.warning(
            f"Placeholder '{{{{{placeholder_name}}}}}' (with flexible spacing) "
            f"was not found or did not lead to a change in the template. "
            f"Ensure the placeholder name and format are correct."
        )
    return substituted_template


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produces dynamic HTML content by replacing placeholders in a given template
    with corresponding user-provided values.

    Placeholders in the template should be in the format {{placeholder_name}}.
    Whitespace around the placeholder_name within the curly braces is tolerated
    (e.g., {{  my_value  }} is treated as {{my_value}}).

    Args:
        html_template: An HTML template string containing placeholders.
        user_values: A dictionary mapping placeholder names (keys) to the
                     values (values) to be inserted. Placeholder names in this
                     dictionary should match the stripped names from the template.

    Returns:
        The HTML content string generated after all placeholder replacements.

    Raises:
        ValueError: If a placeholder found in the template is not present in
                    `user_values`, or if any other error occurs during
                    template processing.
    """
    logger.info("Starting HTML template processing.")
    logger.debug(f"User values provided: {user_values}")

    try:
        placeholders = _extract_placeholders(html_template)
        logger.info(f"Found placeholders: {placeholders}")
        
        processed_template = html_template

        for name in placeholders:
            # The 'name' from _extract_placeholders is already stripped.
            if name not in user_values:
                logger.error(f"Missing value for placeholder: '{name}'")
                raise ValueError(f"Missing value for placeholder: {name}")
            
            value = user_values[name]
            logger.debug(f"Processing placeholder: '{name}' with value: '{value}'")
            processed_template = _substitute_placeholder(
                processed_template, name, value
            )
            
        logger.info("HTML template processing completed successfully.")
        return processed_template

    except ValueError as ve: # Re-raise ValueError directly (e.g., from missing placeholder)
        logger.error(f"ValueError during template processing: {ve}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during template processing: {e}", exc_info=True)
        # Catch any other exceptions during processing and re-raise as ValueError
        raise ValueError(f"Error during template processing: {e}")
