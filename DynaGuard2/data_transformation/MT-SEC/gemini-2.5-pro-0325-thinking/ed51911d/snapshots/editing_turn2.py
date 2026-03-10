import re

def build_html_content(template_text: str, replacements: dict) -> str:
    """
    Replaces placeholders in a JSON-like string template with values from a dictionary.

    Placeholders are delimited by {{...}}.
    If a placeholder in the template doesn't have a match in the dictionary,
    it raises a ValueError.

    Args:
        template_text: The JSON-like string template containing placeholders.
        replacements: A dictionary mapping placeholder names to replacement values.

    Returns:
        The JSON-like string with all placeholders replaced.

    Raises:
        ValueError: If a placeholder in the template is not found in the
                    replacements dictionary.
    """
    def replace_match(match):
        placeholder_name = match.group(1).strip()
        if placeholder_name not in replacements:
            raise ValueError(f"Placeholder '{{{{{placeholder_name}}}}}' not found in replacements dictionary.")
        return str(replacements[placeholder_name])

    # Regex to find {{ placeholder_name }}
    # It captures the content inside the curly braces.
    # Using re.sub with a function allows for custom logic for each replacement.
    processed_text = re.sub(r"\{\{\s*(.*?)\s*\}\}", replace_match, template_text)
    
    return processed_text
