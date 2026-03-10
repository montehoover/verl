import html

xml_design = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data: str) -> str:
    """
    Crafts an XML response by placing user-supplied content into a specified XML template.
    
    Args:
        input_data: str - The content submitted by the user that will be placed into the XML template
        
    Returns:
        str - The final XML response with user content included at the appropriate position in the template
    """
    # Escape special XML characters to prevent XML injection
    escaped_input = html.escape(input_data)
    
    # Insert the escaped input into the XML template
    return xml_design.format(input_data=escaped_input)
