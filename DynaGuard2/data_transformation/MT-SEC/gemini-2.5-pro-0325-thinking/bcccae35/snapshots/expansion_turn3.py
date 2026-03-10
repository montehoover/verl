import re

def contains_html(input_string: str) -> bool:
    """
    Checks if a string contains any HTML tags.

    Args:
        input_string: The string to check.

    Returns:
        True if HTML tags are present, False otherwise.
    """
    # A simple regex to find anything that looks like an HTML tag
    # This will match <tag>, <tag arg="value">, </tag>, <!-- comment -->, etc.
    html_pattern = re.compile(r'<[^>]+>')
    return bool(html_pattern.search(input_string))

def extract_html_attributes(tag_string: str) -> dict:
    """
    Extracts attributes and their values from an HTML tag string.

    Args:
        tag_string: The HTML tag string (e.g., "<a href='#' class='link'>").

    Returns:
        A dictionary of attributes and their values.
        Example: {'href': '#', 'class': 'link'}
    """
    # Regex to find attribute="value" or attribute='value' or attribute=value
    # It captures the attribute name in group 1 and the value (without quotes) in group 2
    # It handles single quotes, double quotes, or no quotes around the value.
    attribute_pattern = re.compile(r"""
        \s+                                # leading whitespace
        ([\w-]+)                           # attribute name (group 1)
        \s*=\s*                            # equals sign with optional surrounding whitespace
        (?:                                # non-capturing group for value
            (?:"([^"]*)")|                 # value in double quotes (group 2)
            (?:'([^']*)')|                 # value in single quotes (group 3)
            (?:([^\s>'"=]+))               # value without quotes (group 4)
        )
    """, re.VERBOSE)

    attributes = {}
    # Find all attribute-value pairs in the tag string
    # The tag name itself is not part of the attributes, so we skip the part before the first space.
    # Example: <a href='#'> -> we search in " href='#'"
    # Find the first space to isolate attributes part
    first_space_index = tag_string.find(' ')
    if first_space_index == -1: # No attributes
        return {}

    attributes_part = tag_string[first_space_index:]

    for match in attribute_pattern.finditer(attributes_part):
        attr_name = match.group(1)
        # The value can be in group 2 (double quotes), group 3 (single quotes), or group 4 (no quotes)
        attr_value = match.group(2) or match.group(3) or match.group(4)
        if attr_value is not None:
            attributes[attr_name] = attr_value
    return attributes

def parse_html_tags(html: str) -> list:
    """
    Extracts all HTML tags from a given string using a regular expression.

    Args:
        html: The string containing HTML content.

    Returns:
        A list of all HTML tags found in the input string.
        Returns an empty list if no tags are found or in case of an error.
    """
    try:
        # Regex to find any HTML tag: <...>
        # This includes opening tags, closing tags, self-closing tags, and comments.
        tag_pattern = re.compile(r'<[^>]+>')
        tags = tag_pattern.findall(html)
        return tags
    except Exception:
        # As per requirement, do not raise exceptions.
        # In case of any regex error or other issue, return an empty list.
        return []
