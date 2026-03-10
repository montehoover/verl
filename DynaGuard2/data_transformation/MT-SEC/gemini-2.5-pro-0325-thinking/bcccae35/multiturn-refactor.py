import re
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _get_html_tag_regex_pattern() -> str:
    """
    Returns the regex pattern used for extracting HTML tags.

    The pattern is designed as follows:
    - `<`: Matches the literal opening angle bracket.
    - `[^>]+`: Matches one or more characters that are not a closing angle bracket ('>').
               This part captures the tag name and any attributes.
    - `>`: Matches the literal closing angle bracket.

    Returns:
        str: The regex pattern string.
    """
    return r"<[^>]+>"


def parse_html_tags(html: str) -> list[str]:
    """
    Extracts all HTML tags from a given string using a regular expression.

    This function employs a regex pattern, obtained from _get_html_tag_regex_pattern(),
    to identify HTML tags.
    The pattern is designed as follows:
    - `<`: Matches the literal opening angle bracket.
    - `[^>]+`: Matches one or more characters that are not a closing angle bracket ('>').
               This part captures the tag name and any attributes.
    - `>`: Matches the literal closing angle bracket.

    While effective for common HTML structures (e.g., <p>, <div>, <a href="...">, </p>),
    it's a simplified approach. It may not correctly parse all HTML complexities,
    such as tags within HTML comments, CDATA sections, or malformed/nested tags
    where '>' might appear in an attribute value (though rare in valid HTML).

    Args:
        html: str, the input HTML string to be parsed.

    Returns:
        A list of strings, each representing an HTML tag found in the input.
        Returns an empty list if no tags are found or if an error occurs
        during parsing.
    """
    logger.debug(f"Input HTML string for tag parsing: \"{html[:100]}{'...' if len(html) > 100 else ''}\"")
    try:
        regex_pattern = _get_html_tag_regex_pattern()
        # Use re.findall to find all non-overlapping matches of the regex pattern.
        tags = re.findall(regex_pattern, html)
        logger.info(f"Successfully extracted tags: {tags}")
        return tags
    except Exception as e:
        # As per requirement, do not raise exceptions. Return an empty list on error.
        logger.error(f"Error parsing HTML tags: {e}", exc_info=True)
        return []
