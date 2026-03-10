from html.parser import HTMLParser

class HTMLElementExtractor(HTMLParser):
    def __init__(self, target_tag):
        super().__init__()
        self.target_tag = target_tag.lower()
        self.elements = []
        self.current_element = None
        self.recording = False
        
    def handle_starttag(self, tag, attrs):
        if tag.lower() == self.target_tag:
            self.recording = True
            self.current_element = f"<{tag}"
            for attr in attrs:
                self.current_element += f' {attr[0]}="{attr[1]}"'
            self.current_element += ">"
            
    def handle_endtag(self, tag):
        if tag.lower() == self.target_tag and self.recording:
            self.current_element += f"</{tag}>"
            self.elements.append(self.current_element)
            self.recording = False
            self.current_element = None
            
    def handle_data(self, data):
        if self.recording and self.current_element is not None:
            self.current_element += data

def extract_html_elements(html_content, tag_name):
    parser = HTMLElementExtractor(tag_name)
    parser.feed(html_content)
    return parser.elements
