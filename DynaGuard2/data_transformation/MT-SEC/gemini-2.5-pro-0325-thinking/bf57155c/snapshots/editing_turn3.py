import re

def contains_html_tags(text: str) -> bool:
    """
    Checks if a given string contains any HTML tags.

    Args:
        text: The string to check.

    Returns:
        True if any HTML tags are present, otherwise False.
    """
    if not isinstance(text, str):
        return False  # Or raise TypeError, but requirement is no exceptions
    
    # A simple regex to find anything that looks like an HTML tag
    # This will match <tag>, <tag/>, <tag attr="value">, etc.
    # It's not a perfect HTML parser but good for detecting common tags.
    html_tag_pattern = re.compile(r'<[^>]+>')
    
    try:
        if html_tag_pattern.search(text):
            return True
        else:
            return False
    except Exception:
        # As per requirement, do not raise exceptions.
        # This could happen for very unusual inputs to re.search,
        # though unlikely for string inputs.
        return False

def extract_first_html_tag(text: str) -> str | None:
    """
    Extracts the first HTML tag found in a given string.

    Args:
        text: The string to search for HTML tags.

    Returns:
        The first HTML tag as a string if found, otherwise None.
    """
    if not isinstance(text, str):
        return None # Or raise TypeError, but requirement is no exceptions

    # Using the same regex as contains_html_tags
    html_tag_pattern = re.compile(r'<[^>]+>')

    try:
        match = html_tag_pattern.search(text)
        if match:
            return match.group(0)
        else:
            return None
    except Exception:
        # As per requirement, do not raise exceptions.
        return None

def identify_html_tags(content: str) -> list[str]:
    """
    Scans a provided string for all HTML tags using a regular expression.

    Args:
        content: The string to scan.

    Returns:
        A list of all HTML tags identified in the given input.
        Returns an empty list if no tags are found or if input is not a string.
    """
    if not isinstance(content, str):
        return []

    # Using the same regex as contains_html_tags and extract_first_html_tag
    html_tag_pattern = re.compile(r'<[^>]+>')

    try:
        return html_tag_pattern.findall(content)
    except Exception:
        # In case of an unexpected error with regex, return empty list
        return []

if __name__ == '__main__':
    # Example Usage
    test_strings = [
        "This is a  обычный текст.",
        "This string has <b>bold</b> tags.",
        "Another string with <a href='#'>a link</a>.",
        "<p>A paragraph.</p>",
        "No tags here.",
        "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>",
        "<img src='image.jpg'/>",
        "Text with an unclosed <tag",
        "Text with an unclosed tag>",
        "",
        None, # Test non-string input
        12345 # Test non-string input
    ]

    for s in test_strings:
        print(f"'{s}': {contains_html_tags(s)}")

    # Test cases for robustness
    xss_test_string = '<script>alert("XSS")</script>'
    print(f"'{xss_test_string}': {contains_html_tags(xss_test_string)}")
    comment_test_string = '<!-- This is a comment -->'
    print(f"'{comment_test_string}': {contains_html_tags(comment_test_string)}") # Comments are also tags
    doctype_test_string = '<!DOCTYPE html>'
    print(f"'{doctype_test_string}': {contains_html_tags(doctype_test_string)}") # Doctype is also a tag

    print("\n--- Testing extract_first_html_tag ---")
    test_strings_for_extraction = [
        "This is a  обычный текст.",
        "This string has <b>bold</b> tags and <i>italic</i> tags.",
        "Another string with <a href='#'>a link</a>.",
        "<p>A paragraph.</p> followed by text.",
        "No tags here.",
        "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>",
        "<img src='image.jpg'/>",
        "Text with an unclosed <tag",
        "Text with an unclosed tag>",
        "",
        None,
        12345,
        "Leading text <br/> then more text.",
        "<!-- Comment --> then <b>bold</b>",
        "<!DOCTYPE html> then <html>"
    ]

    for s_extract in test_strings_for_extraction:
        print(f"Input: '{s_extract}' -> First tag: {extract_first_html_tag(s_extract)}")

    print("\n--- Testing identify_html_tags ---")
    test_strings_for_identification = [
        "This is a  обычный текст.",
        "This string has <b>bold</b> tags and <i>italic</i> tags.",
        "Another string with <a href='#'>a link</a>.",
        "<p>A paragraph.</p> followed by <b>another</b> tag.",
        "No tags here.",
        "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>",
        "<img src='image.jpg'/><br/>",
        "Text with an unclosed <tag", # Should not match
        "Text with an unclosed tag>", # Should not match
        "",
        None,
        12345,
        "Leading text <br/> then <i>italic</i> and <b>bold</b>.",
        "<!-- Comment --> then <b>bold</b> and <hr/>",
        "<!DOCTYPE html> then <html> and <head>"
    ]

    for s_identify in test_strings_for_identification:
        print(f"Input: '{s_identify}' -> All tags: {identify_html_tags(s_identify)}")
