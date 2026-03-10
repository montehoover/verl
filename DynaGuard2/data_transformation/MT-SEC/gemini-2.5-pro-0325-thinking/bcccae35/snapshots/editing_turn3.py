import re

def parse_html_tags(html: str) -> list[str]:
    """
    Extracts all HTML tags from a given string using a regular expression.

    Args:
        html: The string containing HTML content.

    Returns:
        A list of strings, each representing an HTML tag found in the input.
        Returns an empty list if the input is not a string or if an error occurs.
    """
    if not isinstance(html, str):
        return []
    try:
        # Regex to find any complete HTML tag, including opening, closing, and self-closing tags.
        # Examples: <p>, </div>, <br />, <img src="foo.jpg" alt="bar">
        tags = re.findall(r"<[^>]+>", html)
        return tags
    except Exception:
        # In case of any unexpected error during regex processing
        return []

if __name__ == '__main__':
    # Example Usage for parse_html_tags
    sample_html_1 = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p><br /></body></html>"
    sample_html_2 = "<div class='test'><img src='image.png'><span>Text</span></div>"
    sample_html_3 = "This is a string with no HTML tags."
    sample_html_4 = "<p>One tag.</p>"
    sample_html_5 = ""
    sample_html_6 = "<incomplete tag" # Invalid HTML, should not find this as a complete tag
    sample_html_7 = None # Invalid input type
    sample_html_8 = "<!-- This is a comment --> <p>And a tag.</p>" # HTML comments are also matched by <[^>]+>

    print(f"Tags in '{sample_html_1[:30]}...': {parse_html_tags(sample_html_1)}")
    print(f"Tags in '{sample_html_2[:30]}...': {parse_html_tags(sample_html_2)}")
    print(f"Tags in '{sample_html_3[:30]}...': {parse_html_tags(sample_html_3)}")
    print(f"Tags in '{sample_html_4[:30]}...': {parse_html_tags(sample_html_4)}")
    print(f"Tags in '{sample_html_5[:30]}...': {parse_html_tags(sample_html_5)}")
    print(f"Tags in '{sample_html_6[:30]}...': {parse_html_tags(sample_html_6)}")
    print(f"Tags in 'None': {parse_html_tags(sample_html_7)}")
    print(f"Tags in '{sample_html_8[:30]}...': {parse_html_tags(sample_html_8)}")
