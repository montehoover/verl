import html

def create_greeting(name):
    return f'Hello, {name}!'

def create_xml_message(content):
    return f'<message>{content}</message>'

xml_design = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data):
    escaped_data = html.escape(input_data)
    return xml_design.format(input_data=escaped_data)
