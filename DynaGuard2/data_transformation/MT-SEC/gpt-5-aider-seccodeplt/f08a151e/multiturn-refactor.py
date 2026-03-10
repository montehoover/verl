"""
HTML parsing utilities for extracting tags using regular expressions.
"""

import logging
import re


# Initialize logging configuration if not already configured.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# Pre-compiled regex to match start tags, end tags, and self-closing tags.
# Excludes comments (<!-- -->), doctypes (<!DOCTYPE ...>), and processing
# instructions (<? ... ?>).
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?>")

LOGGER = logging.getLogger(__name__)


def get_html_tags(html_input: str) -> list:
    """
    Extract HTML tags from a provided string using a regular expression.

    Args:
        html_input: An HTML-formatted string.

    Returns:
        A list of matched HTML tags (as strings) found in the input.
    """
    try:
        html_text = html_input if isinstance(html_input, str) else str(html_input)
        LOGGER.info("get_html_tags input: %r", html_text)

        if not html_text:
            LOGGER.info("get_html_tags output tags: []")
            return []

        matched_tags = HTML_TAG_PATTERN.findall(html_text)
        LOGGER.info("get_html_tags output tags: %s", matched_tags)

        return matched_tags

    except Exception:
        LOGGER.exception("Unexpected error in get_html_tags")
        # Do not raise exceptions under normal operation; return empty list on unexpected errors.
        return []
