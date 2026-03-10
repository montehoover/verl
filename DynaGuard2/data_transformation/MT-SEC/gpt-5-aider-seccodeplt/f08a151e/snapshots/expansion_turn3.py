import re
from typing import Any, Dict, List, Optional

def tag_exists(html_string: str, tag: str) -> bool:
    """
    Check if the specified HTML tag exists in the given HTML string.

    Args:
        html_string: The HTML content as a string.
        tag: The tag name to search for (e.g., 'div', 'p').

    Returns:
        True if the tag is found, otherwise False.
    """
    if not isinstance(html_string, str) or not isinstance(tag, str) or not tag:
        return False

    # Matches opening, closing, or self-closing tags, ensuring the tag name is exact.
    pattern = re.compile(
        r'<\s*/?\s*' + re.escape(tag) + r'(?![A-Za-z0-9:-])[^>]*>',
        re.IGNORECASE
    )
    return pattern.search(html_string) is not None


def extract_tag_contents(html_string: str) -> Dict[str, Any]:
    """
    Parse an HTML-formatted string and return a nested dictionary structure
    representing tags and their contents. This parser differentiates between
    opening and closing tags and nests content correctly.

    Structure:
        {
            "type": "root",
            "children": [
                {
                    "type": "element",
                    "tag": "div",
                    "attrs": {"class": "container"},
                    "children": [
                        {"type": "text", "content": "Hello"},
                        {
                            "type": "element",
                            "tag": "span",
                            "attrs": {},
                            "children": [{"type": "text", "content": "world"}]
                        }
                    ]
                }
            ]
        }

    Notes:
    - Self-closing and void elements (e.g., br, img) are supported.
    - Attributes are parsed into a dictionary; boolean attributes are set to None.
    - This is a lightweight parser intended for well-formed HTML-like input.
      Complex cases (script/style contents, malformed markup, comments) may require
      a more robust HTML parser.

    Args:
        html_string: The HTML content as a string.

    Returns:
        A dictionary representing the parsed, nested structure of the HTML.
    """
    if not isinstance(html_string, str):
        return {"type": "root", "children": []}

    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"
    }

    tag_re = re.compile(
        r'(?s)<\s*(/)?\s*([A-Za-z][A-Za-z0-9:-]*)\b([^>]*)>',
        re.IGNORECASE
    )

    def _parse_attributes(attr_text: str) -> Dict[str, Optional[str]]:
        attrs: Dict[str, Optional[str]] = {}
        # Attribute parsing regex: name[=value], where value can be quoted or bare.
        attr_re = re.compile(r'''
            (?P<name>[A-Za-z_:][A-Za-z0-9_.:-]*)
            (?:\s*=\s*
                (?:
                    "(?P<dq>[^"]*)"
                  | '(?P<sq>[^\']*)'
                  | (?P<bare>[^\s"\'=<>`]+)
                )
            )?
        ''', re.VERBOSE | re.DOTALL)
        for m in attr_re.finditer(attr_text):
            name = m.group('name')
            if not name:
                continue
            if m.group('dq') is not None:
                attrs[name] = m.group('dq')
            elif m.group('sq') is not None:
                attrs[name] = m.group('sq')
            elif m.group('bare') is not None:
                attrs[name] = m.group('bare')
            else:
                # Boolean attribute (no value)
                attrs[name] = None
        return attrs

    root: Dict[str, Any] = {"type": "root", "children": []}
    stack: List[Dict[str, Any]] = [root]

    last_end = 0
    for m in tag_re.finditer(html_string):
        # Add text node for content before the tag
        if m.start() > last_end:
            text_segment = html_string[last_end:m.start()]
            if text_segment:
                # Preserve text nodes; skip if only whitespace? Keep as content for now.
                stack[-1]["children"].append({"type": "text", "content": text_segment})

        is_closing = m.group(1) is not None
        tag_name = m.group(2).lower()
        raw_attr_text = m.group(3) or ""

        # Determine if self-closing, either with a trailing '/' or known void tag
        stripped = raw_attr_text.rstrip()
        has_trailing_slash = stripped.endswith('/')
        attr_text_for_parsing = stripped[:-1] if has_trailing_slash else raw_attr_text

        if is_closing:
            # Pop stack until matching tag is found
            i = len(stack) - 1
            while i > 0 and not (stack[i].get("type") == "element" and stack[i].get("tag") == tag_name):
                i -= 1
            if i > 0:
                # Close all unclosed intermediate tags up to the matching one
                del stack[i:]
        else:
            attrs = _parse_attributes(attr_text_for_parsing)
            node: Dict[str, Any] = {
                "type": "element",
                "tag": tag_name,
                "attrs": attrs,
                "children": []
            }
            stack[-1]["children"].append(node)
            # Only push non-void and non-self-closing elements
            if not has_trailing_slash and tag_name not in VOID_TAGS:
                stack.append(node)

        last_end = m.end()

    # Any trailing text after the last tag
    if last_end < len(html_string):
        tail_text = html_string[last_end:]
        if tail_text:
            stack[-1]["children"].append({"type": "text", "content": tail_text})

    # If there are unclosed tags, we consider them implicitly closed at the end.
    # The remaining stack (beyond root) indicates unclosed tags; we can ignore as the tree is already built.

    return root


def get_html_tags(html_input: str) -> List[str]:
    """
    Extract all HTML tag names present in the given string using regular expressions.

    Args:
        html_input: The HTML-formatted input string.

    Returns:
        A list of tag names (lowercased) in the order they appear.
        Both opening and closing tags contribute the same tag name.
    """
    if not isinstance(html_input, str):
        return []

    # Match opening, closing, or self-closing tags; ignore comments/doctype/PIs.
    tag_pattern = re.compile(
        r'<\s*/?\s*([A-Za-z][A-Za-z0-9:-]*)\b[^>]*>',
        re.IGNORECASE | re.DOTALL
    )
    return [m.group(1).lower() for m in tag_pattern.finditer(html_input)]
