import re
import html
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def parse_placeholders(html_template: str) -> list:
    """Extract all placeholders from the template."""
    placeholder_pattern = r'\{\{([^}]+)\}\}'
    return [(match.group(0), match.group(1).strip()) 
            for match in re.finditer(placeholder_pattern, html_template)]

def validate_placeholders(placeholders: list, values_dict: dict) -> None:
    """Validate that all placeholders have corresponding values."""
    for _, placeholder_name in placeholders:
        if placeholder_name not in values_dict:
            logger.warning(f"Placeholder '{placeholder_name}' not found in values dictionary")
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
        logger.info("Starting template processing")
        logger.debug(f"Template length: {len(html_template)} characters")
        logger.debug(f"Number of values provided: {len(values_dict)}")
        
        # Pipeline pattern: parse -> validate -> replace
        placeholders = parse_placeholders(html_template)
        logger.debug(f"Found {len(placeholders)} placeholders in template")
        
        validate_placeholders(placeholders, values_dict)
        logger.debug("All placeholders validated successfully")
        
        result = replace_placeholders(html_template, placeholders, values_dict)
        logger.info("Template processing completed successfully")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error during template processing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during template processing: {str(e)}")
        raise ValueError(f"Error processing template: {str(e)}")
