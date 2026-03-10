from html.parser import HTMLParser
import re


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._ignore_depth = 0  # depth inside script/style
        self._block_tags = {
            'address', 'article', 'aside', 'blockquote', 'br', 'canvas', 'dd', 'div', 'dl', 'dt',
            'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'header', 'hr', 'li', 'main', 'nav', 'noscript', 'ol', 'output', 'p', 'pre', 'section',
            'table', 'tfoot', 'thead', 'tbody', 'tr', 'td', 'th', 'ul'
        }

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in ('script', 'style'):
            self._ignore_depth += 1
            return
        if t in self._block_tags:
            self._parts.append(' ')

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ('script', 'style'):
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return
        if t in self._block_tags:
            self._parts.append(' ')

    def handle_startendtag(self, tag, attrs):
        t = tag.lower()
        if t in self._block_tags:
            self._parts.append(' ')

    def handle_data(self, data):
        if self._ignore_depth == 0 and data:
            self._parts.append(data)

    def handle_comment(self, data):
        # ignore comments
        pass

    def get_text(self):
        text = ''.join(self._parts)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def extract_text_content(html):
    """
    Extract text content from an HTML string and return it without tags.
    Text is returned in the order it appears in the HTML.
    """
    if not html:
        return ''
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


class _ElementFinder(HTMLParser):
    def __init__(self, target_tag):
        super().__init__(convert_charrefs=True)
        self._target = target_tag.lower()
        self.matches = []

    def _attrs_to_dict(self, attrs):
        d = {}
        for k, v in attrs:
            d[k] = v
        return d

    def handle_starttag(self, tag, attrs):
        if tag.lower() == self._target:
            self.matches.append({
                'attrs': self._attrs_to_dict(attrs),
                'self_closing': False
            })

    def handle_startendtag(self, tag, attrs):
        if tag.lower() == self._target:
            self.matches.append({
                'attrs': self._attrs_to_dict(attrs),
                'self_closing': True
            })


def find_html_elements(html, tag_name):
    """
    Find all occurrences of a given HTML tag in an HTML string.

    Args:
        html (str): HTML content to search.
        tag_name (str): Tag name to find (case-insensitive).

    Returns:
        list[dict]: A list of occurrences in document order. Each item is a dict:
            {
                'attrs': {attr_name: attr_value_or_None, ...},
                'self_closing': bool
            }
    """
    if not html or not tag_name:
        return []
    parser = _ElementFinder(tag_name)
    parser.feed(html)
    parser.close()
    return parser.matches
