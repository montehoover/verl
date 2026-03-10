import re
import html
from typing import List, Mapping, Any


def parse_placeholders(html_str: str) -> List[str]:
    """
    Parse an HTML string for placeholders formatted as {{...}} and return a list
    of all placeholder names found. Supports nested placeholders.

    Example:
        "{{outer {{inner}} }}" -> ["inner", "outer {{inner}}"]

    Args:
        html_str: The HTML string to parse.

    Returns:
        A list of placeholder contents (names), in the order they are closed
        (inner placeholders appear before their containing outer placeholders).
    """
    results: List[str] = []
    stack: List[int] = []
    i = 0
    n = len(html_str)

    while i < n - 1:
        pair = html_str[i:i + 2]
        if pair == "{{":
            stack.append(i)
            i += 2
            continue
        if pair == "}}" and stack:
            start = stack.pop()
            content = html_str[start + 2:i]
            results.append(content.strip())
            i += 2
            continue
        i += 1

    return results


def replace_placeholders(template: str, values: Mapping[str, Any], default: Any = "") -> str:
    """
    Replace placeholders in an HTML template with values from a dictionary.

    Placeholders are formatted as {{...}} and can be nested. Replacement occurs
    from innermost to outermost. Inner replacements used to form outer placeholder
    names are not HTML-escaped; final inserted values into the output HTML are
    HTML-escaped to safely handle user inputs.

    Args:
        template: The HTML template string containing placeholders.
        values: A mapping of placeholder names to replacement values.
        default: The value to use when a placeholder name is not found in `values`.

    Returns:
        The HTML string with all placeholders replaced.
    """
    # Buffers stack; each element is a list of string segments collected before an opening {{
    stack: List[List[str]] = []
    # Current buffer collecting either top-level HTML or current placeholder content
    buf: List[str] = []

    i = 0
    n = len(template)

    while i < n:
        # Detect opening braces
        if template[i:i + 2] == "{{":
            stack.append(buf)
            buf = []
            i += 2
            continue

        # Detect closing braces
        if template[i:i + 2] == "}}":
            # Unmatched closing braces: treat literally
            if not stack:
                buf.append("}}")
                i += 2
                continue

            # Resolve the current placeholder content
            name = "".join(buf).strip()
            # Pop the previous buffer (context we return to)
            prev_buf = stack.pop()

            # Lookup replacement
            val = values.get(name, default)
            rep = "" if val is None else str(val)

            # Escape only when inserting into top-level HTML (not into another placeholder name)
            if len(stack) == 0:
                rep = html.escape(rep, quote=True)

            # Append replacement to the previous buffer and continue
            prev_buf.append(rep)
            buf = prev_buf
            i += 2
            continue

        # Regular character
        buf.append(template[i])
        i += 1

    # If there are unmatched opening braces, reinsert them literally
    while stack:
        prev_buf = stack.pop()
        prev_buf.append("{{")
        prev_buf.extend(buf)
        buf = prev_buf

    return "".join(buf)


def build_html_content(html_template: str, values_dict: Mapping[str, Any]) -> str:
    """
    Build final HTML content by replacing placeholders in the template with values.

    Placeholders are marked as {{...}} and may be nested. Replacement occurs
    from innermost to outermost. Values inserted into the output HTML are
    HTML-escaped to safely handle user inputs. Raises ValueError on malformed
    templates (unmatched delimiters) or when a placeholder value is missing.

    Args:
        html_template: The HTML template string containing placeholders.
        values_dict: A mapping from placeholder names to replacement values.

    Returns:
        The final HTML string with all placeholders replaced.

    Raises:
        ValueError: If there are unmatched delimiters or a missing placeholder value.
    """
    stack: List[List[str]] = []
    buf: List[str] = []

    i = 0
    n = len(html_template)

    while i < n:
        # Detect opening delimiter
        if html_template[i:i + 2] == "{{":
            stack.append(buf)
            buf = []
            i += 2
            continue

        # Detect closing delimiter
        if html_template[i:i + 2] == "}}":
            if not stack:
                raise ValueError(f"Unmatched closing placeholder delimiter at position {i}")
            name = "".join(buf).strip()
            prev_buf = stack.pop()

            if name not in values_dict:
                raise ValueError(f"Missing value for placeholder: {name}")

            val = values_dict[name]
            rep = "" if val is None else str(val)

            # Escape only when inserting into top-level HTML (not into another placeholder name)
            if len(stack) == 0:
                rep = html.escape(rep, quote=True)

            prev_buf.append(rep)
            buf = prev_buf
            i += 2
            continue

        # Regular character
        buf.append(html_template[i])
        i += 1

    if stack:
        # Some opening delimiters were not closed
        raise ValueError("Unmatched opening placeholder delimiter(s) in template")

    return "".join(buf)
