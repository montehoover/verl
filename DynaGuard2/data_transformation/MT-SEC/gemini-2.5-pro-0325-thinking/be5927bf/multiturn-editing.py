import re
import html

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by replacing placeholders in a template.

    Placeholders are expected to be in the format {key}. For example,
    "<h1>Hello {name}</h1>" where "name" is the key. Values from
    input_map will be HTML-escaped before substitution.

    Args:
        html_template: The HTML template string with placeholders.
        input_map: A dictionary mapping placeholder keys (str) to their
                   replacement values. Values will be converted to strings
                   and HTML-escaped.

    Returns:
        The final HTML string with placeholders safely replaced.

    Raises:
        ValueError: If a placeholder key found in the template (e.g., "key"
                    from "{key}") is not present in the input_map, or if
                    any other template processing error occurs.
    """
    # Regex to find placeholders like {key} and capture the key
    pattern = re.compile(r"\{([^}]+)\}")

    # Find all unique keys referenced by placeholders in the template.
    # This is done first to check for missing keys before attempting substitution.
    placeholder_keys_in_template = set(pattern.findall(html_template))
    
    for key in placeholder_keys_in_template:
        if key not in input_map:
            raise ValueError(
                f"Placeholder '{{{key}}}' found in HTML template but its key "
                f"'{key}' is missing from the input_map dictionary."
            )

    # Define the replacement function for re.sub
    # This function will be called for each match found by pattern.sub
    def replacer(match):
        # match.group(0) is the full placeholder, e.g., "{key}"
        # match.group(1) is the captured key, e.g., "key"
        key = match.group(1)
        
        # We've already checked that key is in input_map.
        # Get the value, convert to string, and then HTML escape it.
        value = input_map[key]
        escaped_value = html.escape(str(value))
        return escaped_value

    try:
        # Substitute all placeholders in the template using the replacer function
        output_html = pattern.sub(replacer, html_template)
    except Exception as e:
        # Catch any unexpected errors during re.sub or replacer execution
        raise ValueError(f"Error during HTML template processing: {e}")
    
    return output_html
