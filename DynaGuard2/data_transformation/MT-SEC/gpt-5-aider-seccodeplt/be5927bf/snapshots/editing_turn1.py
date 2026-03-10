def replace_placeholders(template: str, values: dict) -> str:
    """
    Replace placeholders in the template with their corresponding values.

    Placeholders are delimited by curly braces, e.g. {name}.
    Raises ValueError if the template contains a placeholder that is not present
    in the provided values dictionary.

    Args:
        template: The text template containing placeholders like {key}.
        values: A mapping from placeholder names to their replacement values.

    Returns:
        The template string with all placeholders replaced.
    """
    import re

    pattern = re.compile(r'\{([^{}]+)\}')

    # Collect all unique placeholder names found in the template
    placeholders = {m.group(1) for m in pattern.finditer(template)}

    # Verify all placeholders are provided
    missing = [name for name in placeholders if name not in values]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing values for placeholders: {missing_str}")

    def _replace(match):
        key = match.group(1)
        return str(values[key])

    return pattern.sub(_replace, template)
