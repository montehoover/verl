xml_design = "<response><message>{input_data}</message></response>"

def process_xml_reply(input_data: str) -> str:
    return xml_design.format(input_data=input_data)
