import re

def replace_placeholders(template: str, values: dict) -> str:
    """
    Replaces placeholders in a text template with values from a dictionary.

    Placeholders are expected to be in the format {key}. For example,
    "Hello {name}" where "name" is the key.

    Args:
        template: The text template string with placeholders.
        values: A dictionary mapping placeholder keys (str) to their
                replacement values. Values will be converted to strings.

    Returns:
        The text with all placeholders replaced.

    Raises:
        ValueError: If a placeholder key found in the template (e.g., "key"
                    from "{key}") is not present in the values dictionary.
    """
    # Regex to find placeholders like {key} and capture the key
    pattern = re.compile(r"\{([^}]+)\}")

    # First, check for missing placeholders to satisfy the requirement
    # of raising ValueError if a placeholder in the template is not in values.
    # We find all unique keys referenced by placeholders in the template.
    placeholder_keys_in_template = set(pattern.findall(template))
    
    for key in placeholder_keys_in_template:
        if key not in values:
            raise ValueError(
                f"Placeholder '{{{key}}}' found in template but its key "
                f"'{key}' is missing from the values dictionary."
            )

    # Define the replacement function for re.sub
    # This function will be called for each match found by pattern.sub
    def replacer(match):
        # match.group(0) is the full placeholder, e.g., "{key}"
        # match.group(1) is the captured key, e.g., "key"
        key = match.group(1)
        # We've already checked that key is in values.
        return str(values[key])

    # Substitute all placeholders in the template using the replacer function
    output_text = pattern.sub(replacer, template)
    
    return output_text
