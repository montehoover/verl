import re
import html

__all__ = ["create_html_from_template"]

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][\w\-]*)\s*\}\}")

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
    # Basic type validations
    if not isinstance(template, str):
        raise ValueError("template must be a string")
    if not isinstance(data_mapping, dict):
        raise ValueError("data_mapping must be a dict")
    for k in data_mapping.keys():
        if not isinstance(k, str):
            raise ValueError("All keys in data_mapping must be strings")

    try:
        # Extract placeholders from the template
        placeholders = {m.group(1) for m in _PLACEHOLDER_PATTERN.finditer(template)}
        if placeholders:
            provided_keys = set(data_mapping.keys())
            missing = placeholders - provided_keys
            if missing:
                raise ValueError(f"Incomplete placeholder mapping. Missing keys: {', '.join(sorted(missing))}")

        # Replace using re.sub with escaping of user-provided values
        def _replace(match: re.Match) -> str:
            key = match.group(1)
            # At this point, key must exist; if not, raise to signal processing issue
            if key not in data_mapping:
                raise ValueError(f"Missing value for placeholder '{key}' during processing")
            value = data_mapping[key]
            # Convert to string and escape for safe HTML output
            return html.escape(str(value), quote=True)

        result = _PLACEHOLDER_PATTERN.sub(_replace, template)
        return result
    except ValueError:
        # Re-raise ValueError as-is for clarity
        raise
    except Exception as exc:
        # Normalize unexpected issues under ValueError as per contract
        raise ValueError(f"Failed to process template: {exc}") from exc
