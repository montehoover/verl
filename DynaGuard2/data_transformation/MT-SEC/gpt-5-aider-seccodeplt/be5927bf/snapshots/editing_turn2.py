def replace_placeholders(template, values):
    """
    Replace placeholders in the template(s) with their corresponding values.

    Supports:
      - Single template string and a single dict of values -> returns a string.
      - List/tuple of template strings and a single dict -> returns a list of strings.
      - List/tuple of template strings and a list/tuple of dicts (same length) -> returns a list of strings.

    Placeholders are delimited by curly braces, e.g. {name}.
    Raises ValueError if any template contains a placeholder that is not present
    in the provided values mapping for that template.

    Args:
        template: A template string or a list/tuple of template strings.
        values: A dict mapping placeholder names to values, or a list/tuple of such dicts.

    Returns:
        A formatted string if a single template is provided, otherwise a list of formatted strings.
    """
    import re

    pattern = re.compile(r'\{([^{}]+)\}')

    def _format_one(tmpl: str, vals: dict) -> str:
        placeholders = {m.group(1) for m in pattern.finditer(tmpl)}
        missing = [name for name in placeholders if name not in vals]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing values for placeholders: {missing_str}")
        return pattern.sub(lambda m: str(vals[m.group(1)]), tmpl)

    # Handle list/tuple of templates
    if isinstance(template, (list, tuple)):
        if isinstance(values, (list, tuple)):
            if len(template) != len(values):
                raise ValueError("When providing multiple templates and multiple values mappings, their lengths must match.")
            return [_format_one(t, v) for t, v in zip(template, values)]
        if isinstance(values, dict):
            return [_format_one(t, values) for t in template]
        raise TypeError("For multiple templates, 'values' must be a dict or a list/tuple of dicts.")
    # Handle single template
    if isinstance(values, dict):
        return _format_one(template, values)
    raise TypeError("For a single template, 'values' must be a dict.")
