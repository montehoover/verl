import re
import html
import logging


# Initialize logging to track template processing steps
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
logger.setLevel(logging.INFO)

# Compiled regex pattern to capture placeholders like {{ placeholder_name }}
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([^{}]+?)\s*}}")


def parse_placeholders(html_template: str) -> list:
    """
    Parse and return a list of normalized placeholder names found in the template.
    """
    return [p.strip() for p in PLACEHOLDER_PATTERN.findall(html_template)]


def ensure_all_placeholders_present(placeholders: set, values_dict: dict) -> None:
    """
    Validate that all placeholders have corresponding values in values_dict.
    Raises ValueError on missing placeholders.
    """
    missing = [name for name in placeholders if name not in values_dict]
    if missing:
        logger.error("Missing placeholders during validation: %s", ", ".join(sorted(missing)))
        raise ValueError(f"Missing placeholders: {', '.join(sorted(missing))}")


def replace_placeholders(html_template: str, values_dict: dict) -> str:
    """
    Replace placeholders in the html_template with escaped values from values_dict.
    """
    def replacer(match: re.Match) -> str:
        key = match.group(1).strip()
        if key not in values_dict:
            logger.error("Placeholder '%s' not found during replacement", key)
            raise ValueError(f"Missing placeholder: {key}")
        value = values_dict[key]
        return html.escape("" if value is None else str(value), quote=True)

    return PLACEHOLDER_PATTERN.sub(replacer, html_template)


def build_html_content(html_template: str, values_dict: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in an HTML template
    with corresponding user-provided values.

    Args:
        html_template (str): An HTML template containing placeholders delimited with {{...}}.
        values_dict (dict): A mapping of placeholder names to the values to be inserted.

    Returns:
        str: The HTML content generated after the placeholder replacement.

    Raises:
        ValueError: Raised when any error occurs during template processing or if
                    a placeholder is missing.
    """
    try:
        logger.info("Starting HTML template processing")

        if not isinstance(html_template, str):
            raise ValueError("html_template must be a string")
        if not isinstance(values_dict, dict):
            raise ValueError("values_dict must be a dictionary")

        # Pipeline: parse -> validate -> replace
        placeholders_list = parse_placeholders(html_template)
        placeholders = set(placeholders_list)
        logger.info("Found %d unique placeholder(s): %s", len(placeholders), sorted(placeholders))

        ensure_all_placeholders_present(placeholders, values_dict)
        logger.info("All placeholders are present. Proceeding to replacement.")

        result = replace_placeholders(html_template, values_dict)
        logger.info("Template processing complete.")
        return result

    except Exception as e:
        if isinstance(e, ValueError):
            logger.error("Template processing failed: %s", e)
            raise
        logger.exception("Unexpected error during template processing")
        raise ValueError(f"Error during template processing: {e}")
