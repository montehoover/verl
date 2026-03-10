import re
import html

def build_html_content(html_template: str, values_dict: dict) -> str:
    """
    Generates dynamic HTML content by replacing placeholders in an HTML template.

    Placeholders are delimited by {{...}}.
    User-provided values are HTML-escaped before insertion.
    If a placeholder in the template doesn't have a match in the values_dict,
    it raises a ValueError.

    Args:
        html_template: The HTML template string containing placeholders.
        values_dict: A dictionary mapping placeholder names to values.

    Returns:
        The resulting HTML string after all replacements.

    Raises:
        ValueError: If a placeholder in the template is not found in the
                    values_dict.
    """
    def replace_match(match):
        placeholder_name = match.group(1).strip()
        if placeholder_name not in values_dict:
            raise ValueError(f"Placeholder '{{{{{placeholder_name}}}}}' not found in values_dict.")
        # Escape user input to prevent XSS
        return html.escape(str(values_dict[placeholder_name]))

    # Regex to find {{ placeholder_name }}
    # It captures the content inside the curly braces.
    # Using re.sub with a function allows for custom logic for each replacement.
    processed_text = re.sub(r"\{\{\s*(.*?)\s*\}\}", replace_match, html_template)
    
    return processed_text
