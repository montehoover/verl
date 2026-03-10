xml_design = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data: str) -> str:
    """
    Crafts an XML response by placing the user-supplied content into the XML template.

    Args:
        input_data: The content submitted by the user that will be placed into the XML template.

    Returns:
        The final XML response with user content included at the placeholder position in the template.
    """
    return xml_design.replace("{input_data}", input_data)
