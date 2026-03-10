def format_string(template: str, content: str) -> str:
    """
    Replaces a placeholder in the template with the provided content.

    Args:
        template: The string template with a placeholder.
                  It is assumed the placeholder is "{content}".
        content: The string content to insert into the template.

    Returns:
        The formatted string with the placeholder replaced by content.
    """
    return template.replace("{content}", content)


def insert_into_xml(xml_template: str, user_content: str) -> str:
    """
    Inserts user content into an XML template at the designated placeholder.

    Args:
        xml_template: The XML string template with a placeholder.
                      It is assumed the placeholder is "{content}".
        user_content: The string content to insert into the XML template.

    Returns:
        The formatted XML string with the placeholder replaced by user_content.
    """
    return xml_template.replace("{content}", user_content)

if __name__ == '__main__':
    # Example usage for format_string:
    template_string = "Hello, {content}! Welcome to our service."
    user_name = "Alice"
    formatted_greeting = format_string(template_string, user_name)
    print(formatted_greeting)

    template_string_2 = "The weather today is {content}."
    weather_condition = "sunny"
    formatted_weather = format_string(template_string_2, weather_condition)
    print(formatted_weather)

    # Example usage for insert_into_xml:
    xml_doc_template = "<root><message>{content}</message></root>"
    dynamic_content = "This is a dynamic message for XML!"
    formatted_xml = insert_into_xml(xml_doc_template, dynamic_content)
    print(formatted_xml)

    xml_doc_template_2 = "<data><item id='1'>{content}</item></data>"
    item_content = "Item description here"
    formatted_xml_2 = insert_into_xml(xml_doc_template_2, item_content)
    print(formatted_xml_2)
