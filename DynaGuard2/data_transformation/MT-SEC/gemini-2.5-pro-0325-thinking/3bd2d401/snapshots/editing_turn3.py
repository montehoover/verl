import html

xml_design: str = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data: str) -> str:
    """
    Crafts an XML response by placing user-supplied content into a specified XML template.
    It escapes the input data to prevent XML injection.

    Args:
        input_data: The user-supplied content to include in the XML response.

    Returns:
        The final XML response string with the user content included.
    """
    escaped_input_data = html.escape(input_data)
    return xml_design.format(input_data=escaped_input_data)
