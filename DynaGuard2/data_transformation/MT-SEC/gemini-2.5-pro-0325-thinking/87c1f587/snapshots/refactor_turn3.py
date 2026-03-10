import re
import logging

# Configure basic logging. This logs to the console.
# For more advanced scenarios, consider external configuration or more detailed setup.
logging.basicConfig(
    level=logging.INFO,  # Set default logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _get_html_tag_pattern() -> str:
    """
    Returns the regular expression pattern for finding HTML tags.

    The pattern is designed to match standard HTML tags, including opening,
    closing, and self-closing tags. For example:
    - <p>
    - </div>
    - <img src="image.jpg" />

    Returns:
        str: The regex pattern string.
    """
    # This pattern matches:
    # <      : a literal '<'
    # [^>]+  : one or more characters that are NOT '>' (i.e., the tag content)
    # >      : a literal '>'
    return r"<[^>]+>"


def find_html_tags(html_content: str) -> list[str]:
    """
    Identifies all HTML tags in a given string using regular expressions.

    Args:
        html_content: A string representing the input HTML content to be analyzed.

    Returns:
        A list containing the strings that match as HTML tags within the input.
        Returns an empty list if no tags are found or if input is not a string.
    """
    logger.info(f"Initiating HTML tag search. Input type: {type(html_content).__name__}.")
    if isinstance(html_content, str):
        # Log a snippet for string inputs at DEBUG level to avoid cluttering INFO logs
        # if the content is very large.
        snippet = html_content[:100] + ('...' if len(html_content) > 100 else '')
        logger.debug(f"Input content (snippet): \"{snippet}\"")
    else:
        # For non-string inputs, log the value itself at DEBUG level.
        logger.debug(f"Input content (non-string): {html_content}")

    if not isinstance(html_content, str):
        logger.warning(
            f"Invalid input: Expected a string for HTML content, but got {type(html_content).__name__}. "
            "Returning an empty list."
        )
        return []

    # Fetch the regex pattern from the helper function.
    html_tag_pattern = _get_html_tag_pattern()
    logger.debug(f"Using regex pattern for HTML tags: '{html_tag_pattern}'")

    try:
        tags = re.findall(html_tag_pattern, html_content)
        logger.info(f"Successfully found {len(tags)} HTML tag(s).")
        # Log discovered tags at DEBUG level as the list can be long
        logger.debug(f"Discovered tags: {tags}")
    except Exception as e:
        # This catch block is a safeguard. re.findall with a static valid pattern
        # and string input is unlikely to raise an error here.
        # However, logging provides insight if an unexpected issue occurs.
        logger.error(f"An unexpected error occurred during regex operation: {e}", exc_info=True)
        return []  # Ensure no exceptions are raised, return empty list on error

    return tags
