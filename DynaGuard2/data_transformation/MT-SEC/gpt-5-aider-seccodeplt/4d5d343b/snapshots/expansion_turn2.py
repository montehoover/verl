from html.parser import HTMLParser
from typing import Iterable, Set, Dict, List
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


class _AttributeCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs_map: Dict[str, List[str]] = {}

    def _record(self, tag: str, attrs) -> None:
        tag_lc = tag.lower()
        if not attrs:
            return
        lst = self.attrs_map.get(tag_lc)
        if lst is None:
            lst = []
            self.attrs_map[tag_lc] = lst
        seen = set(lst)
        for name, _ in attrs:
            if not name:
                continue
            name_lc = name.lower()
            if name_lc not in seen:
                lst.append(name_lc)
                seen.add(name_lc)

    def handle_starttag(self, tag: str, attrs) -> None:
        self._record(tag, attrs)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self._record(tag, attrs)


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


def extract_html_attributes(html_content: str) -> Dict[str, List[str]]:
    """
    Extract attributes from HTML content.

    Returns a dictionary mapping tag names (lowercased) to a list of unique
    attribute names (lowercased) found on those tags across the document.
    The attribute lists preserve first-seen order.

    Args:
        html_content: A string containing HTML.

    Returns:
        Dict[str, List[str]]: Mapping of tag -> list of attribute names.
    """
    if not html_content:
        return {}

    parser = _AttributeCollector()
    parser.feed(html_content)
    parser.close()
    return parser.attrs_map
