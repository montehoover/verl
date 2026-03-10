from html.parser import HTMLParser
from typing import Iterable, Set
import re


class _StopParsing(Exception):
    pass


class _ElementFinder(HTMLParser):
    def __init__(self, targets: Set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.targets = targets
        self.found = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in self.targets:
            self.found = True
            raise _StopParsing()

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag.lower() in self.targets:
            self.found = True
            raise _StopParsing()


def _normalize_elements(elements: Iterable[str]) -> Set[str]:
    normalized = set()
    for el in elements:
        if not isinstance(el, str):
            continue
        # Extract a tag name if the element string looks like "<div>" or "DIV" etc.
        m = re.search(r'([a-zA-Z][a-zA-Z0-9:-]*)', el)
        if m:
            normalized.add(m.group(1).lower())
    return normalized


def find_html_elements(html_content: str, elements: Iterable[str]) -> bool:
    """
    Return True if the HTML content contains any of the specified element tags.

    Args:
        html_content: A string containing HTML.
        elements: An iterable of tag names (e.g., 'div', 'span'). Angle brackets are optional.

    Returns:
        bool: True if any specified element is present; False otherwise.
    """
    if not html_content or not elements:
        return False

    targets = _normalize_elements(elements)
    if not targets:
        return False

    parser = _ElementFinder(targets)
    try:
        parser.feed(html_content)
        parser.close()
    except _StopParsing:
        # Early exit when a target element is found
        pass
    return parser.found
