from html.parser import HTMLParser
from typing import List
import html as _html


class _ElementExtractor(HTMLParser):
    def __init__(self, target_tag: str):
        # Keep character references as-is so we can reproduce them in output
        super().__init__(convert_charrefs=False)
        self.target_tag = target_tag.lower()
        self._active_buffers: List[List[str]] = []
        self.results: List[str] = []

    @staticmethod
    def _format_attrs(attrs) -> str:
        if not attrs:
            return ""
        parts = []
        for k, v in attrs:
            if v is None:
                parts.append(k)
            else:
                parts.append(f'{k}="{_html.escape(v, quote=True)}"')
        return " " + " ".join(parts)

    def handle_starttag(self, tag, attrs):
        tag_text = f"<{tag}{self._format_attrs(attrs)}>"
        # Append to all currently active captures
        for buf in self._active_buffers:
            buf.append(tag_text)
        # Start a new capture if this tag matches
        if tag.lower() == self.target_tag:
            self._active_buffers.append([tag_text])

    def handle_endtag(self, tag):
        tag_text = f"</{tag}>"
        for buf in self._active_buffers:
            buf.append(tag_text)
        if tag.lower() == self.target_tag:
            if self._active_buffers:
                # Close the most-recently opened matching element
                buf = self._active_buffers.pop()
                self.results.append("".join(buf))

    def handle_startendtag(self, tag, attrs):
        tag_text = f"<{tag}{self._format_attrs(attrs)}/>"
        for buf in self._active_buffers:
            buf.append(tag_text)
        if tag.lower() == self.target_tag:
            # Self-contained occurrence
            self.results.append(tag_text)

    def handle_data(self, data):
        for buf in self._active_buffers:
            buf.append(data)

    def handle_comment(self, data):
        txt = f"<!--{data}-->"
        for buf in self._active_buffers:
            buf.append(txt)

    def handle_entityref(self, name):
        txt = f"&{name};"
        for buf in self._active_buffers:
            buf.append(txt)

    def handle_charref(self, name):
        txt = f"&#{name};"
        for buf in self._active_buffers:
            buf.append(txt)

    def handle_decl(self, decl):
        txt = f"<!{decl}>"
        for buf in self._active_buffers:
            buf.append(txt)

    def handle_pi(self, data):
        txt = f"<?{data}>"
        for buf in self._active_buffers:
            buf.append(txt)


def extract_html_elements(html_content: str, tag_name: str) -> List[str]:
    """
    Extract all occurrences of a specific HTML tag (including nested ones) from the given HTML content.

    Args:
        html_content: The HTML source as a string.
        tag_name: The tag name to extract (case-insensitive), e.g., 'div', 'a'.

    Returns:
        A list of strings, each containing the full HTML for an occurrence of the specified tag.
    """
    parser = _ElementExtractor(tag_name)
    parser.feed(html_content)
    parser.close()
    return parser.results
