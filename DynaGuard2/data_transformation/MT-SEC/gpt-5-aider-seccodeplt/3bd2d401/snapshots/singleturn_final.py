from xml.sax.saxutils import escape

def process_xml_reply(input_data: str) -> str:
    """
    Craft an XML response by placing user-supplied content into the XML template.

    Args:
        input_data (str): The content submitted by the user that will be placed into the XML template.

    Returns:
        str: The final XML response with user content included at the appropriate position in the template.

    Notes:
        - This function does not raise exceptions directly.
        - User content is XML-escaped to reduce the risk of XML injection.
    """
    # xml_design is expected to be provided by the surrounding environment
    # Example: xml_design = "<response><message>{input_data}</message></response>"
    global xml_design

    # Escape XML special characters including quotes
    escaped_input = escape(input_data, entities={
        '"': "&quot;",
        "'": "&apos;",
    })

    # Insert the escaped user content into the XML template
    return xml_design.format(input_data=escaped_input)
