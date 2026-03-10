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
