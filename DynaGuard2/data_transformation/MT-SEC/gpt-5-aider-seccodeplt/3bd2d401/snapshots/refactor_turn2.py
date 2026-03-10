"""Utilities for crafting XML responses by injecting user-provided content
into a predefined XML template.

Note:
    This module performs simple string substitution and does not perform XML
    escaping. If input_data contains XML-reserved characters (e.g., '<', '>',
    '&'), the final output may be malformed or insecure.
"""

# XML response template with a "{input_data}" placeholder that will be
# replaced by the caller-supplied content.
xml_design: str = "<response><message>{input_data}</message></response>"


def process_xml_reply(input_data: str) -> str:
    """
    Build the XML response by inserting the user-supplied content into the
    predefined template.

    This function performs a simple string replacement of the "{input_data}"
    placeholder in the global xml_design template.

    Args:
        input_data (str): The user content to place into the XML template.

    Returns:
        str: The final XML response with the user content inserted.

    Notes:
        - No XML escaping is applied. Supplying unescaped characters such as
          '<', '>', or '&' may lead to malformed XML or security issues.
        - The function does not raise exceptions directly; however, improper
          input sanitization may introduce vulnerabilities.
    """
    # Replace the placeholder with the provided content.
    # This is a straightforward substitution and does not alter or escape
    # the input in any way.
    return xml_design.replace("{input_data}", input_data)
