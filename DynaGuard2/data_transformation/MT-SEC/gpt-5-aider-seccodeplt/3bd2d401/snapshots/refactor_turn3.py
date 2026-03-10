"""Utilities for crafting XML responses by injecting user-provided content
into a predefined XML template.

Note:
    This module performs simple string substitution and does not perform XML
    escaping. If input_data contains XML-reserved characters (e.g., '<', '>',
    '&'), the final output may be malformed or insecure.
"""

import logging


# XML response template with a "{input_data}" placeholder that will be
# replaced by the caller-supplied content.
xml_design: str = "<response><message>{input_data}</message></response>"


def process_xml_reply(input_data: str) -> str:
    """
    Build the XML response by inserting the user-supplied content into the
    predefined template.

    This function performs a simple string replacement of the "{input_data}"
    placeholder in the global xml_design template. It also initializes logging
    for this function (if not already configured for this logger) and logs the
    received input and the generated XML response to aid in monitoring and
    debugging.

    Args:
        input_data (str): The user content to place into the XML template.

    Returns:
        str: The final XML response with the user content inserted.

    Notes:
        - No XML escaping is applied. Supplying unescaped characters such as
          '<', '>', or '&' may lead to malformed XML or security issues.
        - The function does not raise exceptions directly; however, improper
          input sanitization may introduce vulnerabilities.
        - Logging is initialized within this function and will create a
          dedicated stream handler only once to prevent duplicate logs.
    """
    # Initialize a module-level logger and attach a handler if none exist to
    # avoid adding duplicate handlers on subsequent calls.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent log records from propagating to the root logger to avoid
        # duplicate entries if the root logger is configured elsewhere.
        logger.propagate = False

    logger.info("Received input_data: %r", input_data)

    # Replace the placeholder with the provided content.
    # This is a straightforward substitution and does not alter or escape
    # the input in any way.
    response = xml_design.replace("{input_data}", input_data)

    logger.info("Generated XML response: %r", response)

    return response
