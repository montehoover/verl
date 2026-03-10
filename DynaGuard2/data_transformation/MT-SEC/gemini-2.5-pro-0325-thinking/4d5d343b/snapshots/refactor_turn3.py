import re
import logging

# Configure basic logging
# To see debug logs, you might need to configure the root logger level, e.g.:
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _get_html_tag_regex_pattern() -> str:
    """
    Returns the regex pattern for extracting HTML tag names.

    The regex pattern is r"<[/]?([a-zA-Z0-9]+)(?:[^>]*)?>":
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

    Returns:
        str: The regex pattern string.
    """
    return r"<[/]?([a-zA-Z0-9]+)(?:[^>]*)?>"


def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.

    Args:
        html_code (str): An HTML string that serves as the input for parsing.

    Returns:
        list: A list of all HTML tag names (as strings) identified in the
              given input (e.g., ['html', 'head', 'title', 'title', 'head', 'body']).
              Returns an empty list if no tags are found or in case of an error
              during regex processing.
    """
    logger.debug(f"Attempting to grab HTML tags from input: {html_code[:100]}...") # Log snippet
    try:
        regex_pattern = _get_html_tag_regex_pattern()
        # Use regex to find all HTML tag names.
        # The pattern captures the main tag name, ignoring attributes and
        # whether it's an opening, closing, or self-closing tag.
        # See _get_html_tag_regex_pattern() docstring for regex details.
        tags = re.findall(regex_pattern, html_code)
        logger.debug(f"Extracted tags: {tags}")
        return tags
    except Exception as e:
        # As per requirement, do not raise exceptions.
        # Log the error and return an empty list to ensure robust operation.
        logger.error(f"Error processing HTML string: {e}", exc_info=True)
        return []
