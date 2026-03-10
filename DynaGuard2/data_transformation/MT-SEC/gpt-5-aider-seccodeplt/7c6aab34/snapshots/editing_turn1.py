def format_text(template, values):
    """
    Replace placeholders in a text template with values from a dictionary.

    Args:
        template (str): A template string containing placeholders in braces, e.g., "{name}".
        values (dict): A mapping of placeholder names to values.

    Returns:
        str: The formatted string.

    Raises:
        ValueError: If the template is malformed (e.g., unmatched braces) or if any placeholder
                    does not have a corresponding value.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    if not isinstance(values, dict):
        raise ValueError("Values must be a dictionary")

    try:
        return template.format_map(values)
    except KeyError as e:
        missing = e.args[0] if e.args else "<unknown>"
        raise ValueError(f"Missing value for placeholder: {missing}") from None
    except (ValueError, IndexError) as e:
        # ValueError covers malformed templates (e.g., unmatched braces).
        # IndexError can occur with invalid field references.
        raise ValueError(f"Malformed template: {e}") from None
