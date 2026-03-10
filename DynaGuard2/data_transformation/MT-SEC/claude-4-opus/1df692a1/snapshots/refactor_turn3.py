import re
import html
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)

def extract_placeholders(template: str) -> set:
    """
    Extract all placeholder names from the template.
    
    Args:
        template: str, an HTML template with placeholders
    
    Returns:
        set of placeholder names found in the template
    """
    placeholder_pattern = r'\{\{?\s*(\w+)\s*\}?\}'
    placeholders = set(re.findall(placeholder_pattern, template))
    logger.debug(f"Extracted {len(placeholders)} placeholders: {sorted(placeholders)}")
    return placeholders


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
        error_msg = f"Incomplete placeholder mapping. Missing values for: {', '.join(sorted(missing_placeholders))}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"All {len(placeholders)} placeholders have corresponding values in data mapping")


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
    replacement_count = 0
    
    def replace_match(match):
        nonlocal replacement_count
        placeholder_name = match.group(1)
        if placeholder_name in data_mapping:
            original_value = data_mapping[placeholder_name]
            sanitized_value = html.escape(str(original_value))
            logger.debug(f"Replacing placeholder '{placeholder_name}' with sanitized value: '{original_value}' -> '{sanitized_value}'")
            replacement_count += 1
            return sanitized_value
        return match.group(0)
    
    result = re.sub(placeholder_pattern, replace_match, template)
    logger.info(f"Successfully replaced {replacement_count} placeholders in template")
    return result


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
    logger.info("Starting HTML template creation")
    logger.debug(f"Template length: {len(template)} characters")
    logger.debug(f"Data mapping contains {len(data_mapping)} entries")
    
    try:
        if not isinstance(template, str):
            error_msg = "Template must be a string"
            logger.error(f"Invalid template type: {type(template).__name__}")
            raise ValueError(error_msg)
        
        if not isinstance(data_mapping, dict):
            error_msg = "Data mapping must be a dictionary"
            logger.error(f"Invalid data mapping type: {type(data_mapping).__name__}")
            raise ValueError(error_msg)
        
        # Extract placeholders from template
        placeholders = extract_placeholders(template)
        
        # Validate that all placeholders have corresponding values
        validate_placeholder_mapping(placeholders, data_mapping)
        
        # Replace placeholders with sanitized values
        result = sanitize_and_replace_placeholders(template, data_mapping)
        
        logger.info("HTML template creation completed successfully")
        logger.debug(f"Result length: {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Failed to create HTML from template: {str(e)}")
        raise
