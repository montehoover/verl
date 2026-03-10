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
