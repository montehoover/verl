import re

def replace_placeholders(templates: list[str], values: dict) -> list[str]:
    """
    Replaces placeholders in a list of text templates with values from a dictionary.

    Placeholders are expected to be in the format {key}. For example,
    "Hello {name}" where "name" is the key.

    Args:
        templates: A list of text template strings with placeholders.
        values: A dictionary mapping placeholder keys (str) to their
                replacement values. Values will be converted to strings.
                This dictionary is applied to all templates.

    Returns:
        A list of texts with all placeholders replaced, corresponding to the
        input templates.

    Raises:
        ValueError: If a placeholder key found in any template (e.g., "key"
                    from "{key}") is not present in the values dictionary.
    """
    # Regex to find placeholders like {key} and capture the key
    pattern = re.compile(r"\{([^}]+)\}")

    # Define the replacement function for re.sub
    # This function will be called for each match found by pattern.sub
    # It captures 'values' from the outer scope.
    def replacer(match):
        # match.group(0) is the full placeholder, e.g., "{key}"
        # match.group(1) is the captured key, e.g., "key"
        key = match.group(1)
        # The check for key in values will be done per template before substitution.
        return str(values[key])

    processed_texts = []
    for single_template in templates:
        # First, check for missing placeholders for the current template
        # We find all unique keys referenced by placeholders in this template.
        placeholder_keys_in_template = set(pattern.findall(single_template))
        
        for key in placeholder_keys_in_template:
            if key not in values:
                # Add a snippet of the problematic template for better error reporting
                template_snippet = single_template[:75] + "..." if len(single_template) > 75 else single_template
                raise ValueError(
                    f"Placeholder '{{{key}}}' found in template but its key "
                    f"'{key}' is missing from the values dictionary. "
                    f"Problem in template (first 75 chars): \"{template_snippet}\""
                )

        # Substitute all placeholders in the current template using the replacer function
        output_text = pattern.sub(replacer, single_template)
        processed_texts.append(output_text)
    
    return processed_texts
