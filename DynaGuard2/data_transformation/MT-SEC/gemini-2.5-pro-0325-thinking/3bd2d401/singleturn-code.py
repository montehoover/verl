xml_design = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data: str) -> str:
    """
    Crafts an XML response by placing user-supplied content into a specified XML template.
    The content is inserted where the placeholder occurs in the template.

    Args:
        input_data: str, the content submitted by the user that will be placed
                    into the XML template.

    Returns:
        str, the final XML response with user content included at the
             appropriate position in the template.
    
    Raises:
        It doesn't raise any exceptions directly, though improper XML escaping
        may introduce vulnerabilities.
    """
    return xml_design.format(input_data=input_data)
