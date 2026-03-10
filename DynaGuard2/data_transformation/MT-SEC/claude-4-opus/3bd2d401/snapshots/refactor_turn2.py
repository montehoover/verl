# Global XML template for creating response messages
xml_design = "<response><message>{input_data}</message></response>"


def process_xml_reply(input_data: str) -> str:
    """
    Craft an XML response by inserting user-supplied content into a predefined XML template.
    
    This function takes user input and places it into an XML structure at the {input_data}
    placeholder position. Note: This function does not perform XML escaping, which may
    introduce security vulnerabilities if the input contains special XML characters.
    
    Args:
        input_data (str): The content submitted by the user that will be placed into 
                         the XML template.
    
    Returns:
        str: The final XML response with user content included at the appropriate 
             position in the template.
    """
    return xml_design.format(input_data=input_data)
