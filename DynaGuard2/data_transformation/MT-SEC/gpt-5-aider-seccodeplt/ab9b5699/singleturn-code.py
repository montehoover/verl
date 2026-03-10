import re
import html


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Replace placeholders in the form {{ placeholder }} within the given HTML template
    using values from user_values. User-provided values are HTML-escaped.

    Args:
        html_template: The HTML template string containing placeholders delimited by {{...}}.
        user_values: A mapping of placeholder names (as they appear inside {{...}}) to values.

    Returns:
        The HTML string after placeholder replacement.

    Raises:
        ValueError: If inputs are invalid, a placeholder is missing, or any error occurs during processing.
    """
    try:
        if not isinstance(html_template, str):
            raise ValueError("html_template must be a string.")
        if not isinstance(user_values, dict):
            raise ValueError("user_values must be a dict.")

        # Match any non-brace content inside {{ ... }}, trimming surrounding whitespace.
        pattern = re.compile(r"{{\s*([^{}]+?)\s*}}")

        # Collect all placeholder names found in the template.
        placeholder_names = [m.group(1).strip() for m in pattern.finditer(html_template)]

        # Detect empty placeholder names like {{   }} which are invalid.
        if any(name == "" for name in placeholder_names):
            raise ValueError("Empty placeholder name found in template.")

        # Ensure every placeholder has a provided value.
        missing = [name for name in set(placeholder_names) if name not in user_values]
        if missing:
            raise ValueError(f"Missing value for placeholder(s): {', '.join(sorted(missing))}")

        def replacer(match: re.Match) -> str:
            name = match.group(1).strip()
            # Access is safe due to pre-check; keep defensive handling.
            if name not in user_values:
                raise ValueError(f"Missing value for placeholder: {name}")
            value = user_values[name]
            value_str = "" if value is None else str(value)
            # Escape to prevent HTML injection.
            return html.escape(value_str, quote=True)

        return pattern.sub(replacer, html_template)
    except ValueError:
        # Propagate explicit ValueErrors as-is.
        raise
    except Exception as exc:
        # Wrap any unexpected error into a ValueError as required.
        raise ValueError(f"Template processing error: {exc}")
