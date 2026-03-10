import re


def detect_html_tags(html_str: str) -> list:
    """
    Identifies all HTML tags in a given HTML string.

    This function uses a regular expression to find all occurrences of
    HTML-like tags. The regex is designed to be broad and capture
    anything enclosed in angle brackets (`< >`).

    Args:
        html_str: A string representing the input HTML content to be
                  analyzed.

    Returns:
        A list of strings, where each string is an HTML tag found in
        the input `html_str`. If no tags are found, an empty list is
        returned.
    """
    # The regex r"<[^>]+>" aims to match:
    #   <     : a literal opening angle bracket
    #   [^>]+ : one or more characters that are NOT a closing angle bracket
    #   >     : a literal closing angle bracket
    # This pattern will match:
    #   - Opening tags: <p>, <div class="example">
    #   - Closing tags: </p>, </div>
    #   - Self-closing tags: <img src="..."/>, <br />
    #   - HTML comments: <!-- This is a comment -->
    #   - DOCTYPE declarations: <!DOCTYPE html>
    #
    # For more specific tag matching, the regex pattern would need to be refined.
    # For example:
    #   - To match only opening/closing tag names (e.g., "p", "div"): r"</?([a-zA-Z0-9]+)"
    #   - To match only standard opening tags: r"<([a-zA-Z0-9]+)\s*[^>]*>" (excluding self-closing)
    html_tag_pattern_str = r"<[^>]+>"
    
    return _parse_html_tags_with_regex(html_str, html_tag_pattern_str)


def _parse_html_tags_with_regex(html_content: str, regex_pattern: str) -> list:
    """
    Parses HTML content using a regex pattern to find all matching tags.

    This is a helper function that encapsulates the regular expression
    compilation and matching logic.

    Args:
        html_content: The HTML string to parse.
        regex_pattern: The regular expression pattern string to use for
                       finding tags.

    Returns:
        A list of strings, where each string is a match found by the
        regex pattern in the `html_content`. Returns an empty list if
        no matches are found.
    """
    compiled_pattern = re.compile(regex_pattern)
    tags = compiled_pattern.findall(html_content)
    return tags
