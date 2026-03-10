import re


def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.

    The regex pattern used is r"<[/]?([a-zA-Z0-9]+)(?:[^>]*)?>":
    - `<`: Matches the opening angle bracket of a tag.
    - `[/]?`: Optionally matches a forward slash (for closing tags like </p>).
    - `([a-zA-Z0-9]+)`: This is the main capturing group. It captures the tag
      name itself (e.g., 'div', 'p', 'h1'). Tag names are assumed to consist
      of one or more alphanumeric characters.
    - `(?:[^>]*)?`: This is an optional non-capturing group.
        - `(?: ... )`: Defines a non-capturing group.
        - `[^>]*`: Matches zero or more characters that are NOT a closing
          angle bracket '>'. This part accounts for attributes (e.g.,
          class="example", href="url"), spaces, and self-closing slashes
          (e.g., <br />).
        - `?` after the group makes this entire part optional, though `*`
          already allows for zero characters.
    - `>`: Matches the closing angle bracket of a tag.

    Args:
        html_code (str): An HTML string that serves as the input for parsing.

    Returns:
        list: A list of all HTML tag names (as strings) identified in the
              given input (e.g., ['html', 'head', 'title', 'title', 'head', 'body']).
              Returns an empty list if no tags are found or in case of an error
              during regex processing.
    """
    try:
        # Use regex to find all HTML tag names.
        # The pattern r"<[/]?([a-zA-Z0-9]+)(?:[^>]*)?>" captures the main tag name,
        # ignoring attributes and whether it's an opening, closing, or self-closing tag.
        # See the docstring for a detailed breakdown of this regular expression.
        tags = re.findall(r"<[/]?([a-zA-Z0-9]+)(?:[^>]*)?>", html_code)
        return tags
    except Exception:
        # As per requirement, do not raise exceptions.
        # In case of any regex processing error or other unexpected issue,
        # return an empty list to ensure robust operation.
        return []
