import re
import html
import logging

__all__ = ["create_html_from_template"]

logger = logging.getLogger(__name__)

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][\w\-]*)\s*\}\}")


def _validate_inputs(template: str, data_mapping: dict) -> None:
    """
    Validate input types and mapping key types.
    Raises ValueError if validation fails.
    """
    if not isinstance(template, str):
        raise ValueError("template must be a string")
    if not isinstance(data_mapping, dict):
        raise ValueError("data_mapping must be a dict")
    for k in data_mapping.keys():
        if not isinstance(k, str):
            raise ValueError("All keys in data_mapping must be strings")


def _extract_placeholders(template: str) -> set[str]:
    """
    Extract placeholder names from the template of the form {{name}}.

    Args:
        template: The HTML template string.

    Returns:
        A set of placeholder names found in the template.
    """
    placeholders = {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}
    logger.debug("Extracted placeholders: %s", sorted(placeholders))
    return placeholders


def _render_template(template: str, data_mapping: dict) -> str:
    """
    Render the template by replacing placeholders with escaped values from data_mapping.

    Args:
        template: The HTML template string.
        data_mapping: Mapping of placeholder names to values.

    Returns:
        The rendered HTML string.
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in data_mapping:
            # Defensive: should not happen if caller validated completeness,
            # but keep explicit error to maintain contract.
            logger.error("Missing value for placeholder '%s' during processing", key)
            raise ValueError(f"Missing value for placeholder '{key}' during processing")
        value = data_mapping[key]
        value_str = str(value)
        escaped = html.escape(value_str, quote=True)
        logger.debug(
            "Replacing placeholder '%s' with escaped value (orig_len=%d, escaped_len=%d)",
            key, len(value_str), len(escaped)
        )
        return escaped

    rendered = _PLACEHOLDER_PATTERN.sub(_replace, template)
    return rendered


def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Create dynamic HTML by replacing placeholders in the template with values
    from data_mapping. Placeholders use the syntax {{placeholder_name}}.

    Args:
        template: str - The HTML template containing placeholders.
        data_mapping: dict - Mapping of placeholder names to their values.

    Returns:
        str: The processed HTML with placeholders replaced by escaped values.

    Raises:
        ValueError: If processing fails or the placeholder mapping is incomplete.
    """
    logger.info("Starting HTML template rendering")
    try:
        _validate_inputs(template, data_mapping)

        placeholders = _extract_placeholders(template)
        if placeholders:
            provided_keys = set(data_mapping.keys())
            missing = placeholders - provided_keys
            if missing:
                logger.warning(
                    "Incomplete placeholder mapping. Missing keys: %s",
                    ", ".join(sorted(missing))
                )
                raise ValueError(f"Incomplete placeholder mapping. Missing keys: {', '.join(sorted(missing))}")
            extra = provided_keys - placeholders
            if extra:
                logger.debug("Unused mapping keys: %s", ", ".join(sorted(extra)))
        else:
            logger.debug("No placeholders found in template")

        result = _render_template(template, data_mapping)
        logger.info("Template rendering complete (placeholders_processed=%d)", len(placeholders))
        return result

    except ValueError as ve:
        logger.error("Template processing error: %s", ve)
        raise
    except Exception as exc:
        logger.exception("Unexpected error during template processing")
        raise ValueError(f"Failed to process template: {exc}") from exc
