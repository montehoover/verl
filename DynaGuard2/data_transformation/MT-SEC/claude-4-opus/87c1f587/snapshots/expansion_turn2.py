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

class NestedHTMLExtractor(HTMLParser):
    def __init__(self, parent_tag):
        super().__init__()
        self.parent_tag = parent_tag.lower()
        self.nested_elements = []
        self.inside_parent = False
        self.parent_depth = 0
        self.current_nested = None
        self.nested_depth = 0
        
    def handle_starttag(self, tag, attrs):
        if tag.lower() == self.parent_tag:
            self.inside_parent = True
            self.parent_depth += 1
        elif self.inside_parent and self.parent_depth > 0:
            if self.nested_depth == 0:
                self.current_nested = f"<{tag}"
                for attr in attrs:
                    self.current_nested += f' {attr[0]}="{attr[1]}"'
                self.current_nested += ">"
            else:
                if self.current_nested:
                    self.current_nested += f"<{tag}"
                    for attr in attrs:
                        self.current_nested += f' {attr[0]}="{attr[1]}"'
                    self.current_nested += ">"
            self.nested_depth += 1
            
    def handle_endtag(self, tag):
        if tag.lower() == self.parent_tag and self.inside_parent:
            self.parent_depth -= 1
            if self.parent_depth == 0:
                self.inside_parent = False
        elif self.inside_parent and self.nested_depth > 0:
            if self.current_nested:
                self.current_nested += f"</{tag}>"
            self.nested_depth -= 1
            if self.nested_depth == 0 and self.current_nested:
                self.nested_elements.append(self.current_nested)
                self.current_nested = None
                
    def handle_data(self, data):
        if self.inside_parent and self.nested_depth > 0 and self.current_nested:
            self.current_nested += data

def extract_nested_html(html_content, parent_tag):
    parser = NestedHTMLExtractor(parent_tag)
    parser.feed(html_content)
    return parser.nested_elements
