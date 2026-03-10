import re
import html

__all__ = ["create_html_from_template"]

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
    return {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}


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
            raise ValueError(f"Missing value for placeholder '{key}' during processing")
        value = data_mapping[key]
        return html.escape(str(value), quote=True)

    return _PLACEHOLDER_PATTERN.sub(_replace, template)


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
    try:
        _validate_inputs(template, data_mapping)

        placeholders = _extract_placeholders(template)
        if placeholders:
            provided_keys = set(data_mapping.keys())
            missing = placeholders - provided_keys
            if missing:
                raise ValueError(f"Incomplete placeholder mapping. Missing keys: {', '.join(sorted(missing))}")

        result = _render_template(template, data_mapping)
        return result

    except ValueError:
        # Re-raise ValueError as-is for clarity
        raise
    except Exception as exc:
        # Normalize unexpected issues under ValueError as per contract
        raise ValueError(f"Failed to process template: {exc}") from exc
