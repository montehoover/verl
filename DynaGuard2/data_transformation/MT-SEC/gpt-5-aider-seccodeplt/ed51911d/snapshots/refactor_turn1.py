import re
import html


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
        if not isinstance(html_template, str):
            raise ValueError("html_template must be a string")
        if not isinstance(values_dict, dict):
            raise ValueError("values_dict must be a dictionary")

        # Regex to capture placeholders like {{ placeholder_name }}
        pattern = re.compile(r"{{\s*([^{}]+?)\s*}}")

        # Find all placeholders and ensure all are provided
        found_placeholders = pattern.findall(html_template)
        normalized_placeholders = {p.strip() for p in found_placeholders}

        missing = [name for name in normalized_placeholders if name not in values_dict]
        if missing:
            raise ValueError(f"Missing placeholders: {', '.join(sorted(missing))}")

        def replacer(match: re.Match) -> str:
            key = match.group(1).strip()
            if key not in values_dict:
                # Defensive: In case placeholders change during processing
                raise ValueError(f"Missing placeholder: {key}")
            value = values_dict[key]
            # Safely escape user-provided values to prevent HTML injection
            return html.escape("" if value is None else str(value), quote=True)

        result = pattern.sub(replacer, html_template)
        return result

    except Exception as e:
        # Convert any unexpected error into a ValueError as per spec
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error during template processing: {e}")
