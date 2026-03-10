import re
import logging

# Configure basic logging
# In a real application, you might want to configure this more extensively
# (e.g., in your application's entry point or a dedicated logging config file)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_html_tag_regex() -> str:
    """
    Returns the regular expression pattern for identifying HTML tags.

    The regex is designed to capture the name of an HTML tag from
    both opening (e.g., <p>, <h1>) and closing (e.g., </p>, </div>) tags.

    Breakdown of the regex: r"</?([a-zA-Z][a-zA-Z0-9]*)"
    - `</?`: Matches the literal character '<' optionally followed by '/'.
             This handles both opening tags like `<tag>` and closing
             tags like `</tag>`.
    - `([a-zA-Z][a-zA-Z0-9]*)`: This is the capturing group for the tag name.
      - `[a-zA-Z]`: Ensures the tag name starts with an alphabet character
                    (case-insensitive).
      - `[a-zA-Z0-9]*`: Allows the rest of the tag name to consist of
                       alphanumeric characters.

    Returns:
        str: The regex pattern string.
    """
    return r"</?([a-zA-Z][a-zA-Z0-9]*)"

def identify_html_tags(content: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.

    Args:
        content: str, an HTML string that serves as the input for parsing.

    Returns:
        A list of all HTML tags identified in the given input.
        This function should operate without raising exceptions.
    """
    logger.debug(f"Processing content: '{content[:100]}{'...' if len(content) > 100 else ''}'") # Log a snippet

    tag_regex = _get_html_tag_regex()
    tags = re.findall(tag_regex, content)

    logger.debug(f"Identified tags: {tags}")
    return tags
