import re
import html

def format_text(template, values):
    """
    Replace placeholders in a text template with SQL-safe values from a dictionary.

    Notes:
        - Placeholders should NOT be manually quoted in the template. This function will
          quote string-like values as needed to produce SQL-safe literals.
        - Strings are single-quoted with internal quotes doubled.
        - None -> NULL, bool -> TRUE/FALSE, numbers are unquoted, datetime/date/time are quoted ISO strings.
        - Byte sequences are rendered as standard SQL hex literals: X'ABCD'.
        - Sequences (list/tuple/set) are rendered as parenthesized, comma-separated SQL-safe values.

    Args:
        template (str): A template string containing placeholders in braces, e.g., "WHERE name = {name}".
        values (dict): A mapping of placeholder names to values.

    Returns:
        str: The formatted SQL-safe string.

    Raises:
        ValueError: If the template is malformed (e.g., unmatched braces) or if any placeholder
                    does not have a corresponding value. Also raised for values that cannot be
                    safely represented (e.g., empty sequences, non-finite floats).
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    if not isinstance(values, dict):
        raise ValueError("Values must be a dictionary")

    import datetime as _dt
    from decimal import Decimal as _Decimal
    import math as _math
    from collections.abc import Sequence as _Sequence

    def _escape_str(s: str) -> str:
        # Escape single quotes by doubling them.
        return "'" + s.replace("'", "''") + "'"

    def _escape_sql_value(v):
        if v is None:
            return "NULL"
        # bool must be checked before int because bool is a subclass of int.
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if not _math.isfinite(v):
                raise ValueError("Non-finite float is not representable in SQL")
            return format(v, ".15g")
        if isinstance(v, _Decimal):
            return str(v)
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            return "X'" + b.hex().upper() + "'"
        if isinstance(v, _dt.datetime):
            return _escape_str(v.isoformat(sep=" ", timespec="microseconds"))
        if isinstance(v, _dt.date):
            return _escape_str(v.isoformat())
        if isinstance(v, _dt.time):
            try:
                return _escape_str(v.isoformat(timespec="microseconds"))
            except TypeError:
                return _escape_str(v.isoformat())
        if isinstance(v, str):
            return _escape_str(v)
        if isinstance(v, _Sequence) and not isinstance(v, (str, bytes, bytearray, memoryview)):
            seq = list(v)
            if len(seq) == 0:
                raise ValueError("Empty sequence cannot be represented safely in SQL")
            return "(" + ", ".join(_escape_sql_value(x) for x in seq) + ")"
        # Fallback: convert to string and escape as a SQL string literal.
        return _escape_str(str(v))

    # Pre-escape provided values into SQL-safe literals.
    safe_values = {k: _escape_sql_value(v) for k, v in values.items()}

    try:
        return template.format_map(safe_values)
    except KeyError as e:
        missing = e.args[0] if e.args else "<unknown>"
        raise ValueError(f"Missing value for placeholder: {missing}") from None
    except (ValueError, IndexError) as e:
        # Malformed templates (e.g., unmatched braces) or invalid field references.
        raise ValueError(f"Malformed template: {e}") from None


def generate_dynamic_html(template, user_input):
    """
    Generate HTML by replacing placeholders in the template with HTML-escaped values.

    - Placeholders must be identifiers inside single braces, e.g., {username}.
    - Literal braces can be produced by doubling: {{ and }}.
    - All substituted values are HTML-escaped (including quotes) to avoid injection.

    Args:
        template (str): HTML template containing placeholders like {name}.
        user_input (dict): Mapping of placeholder names to values.

    Returns:
        str: The generated, safe HTML string.

    Raises:
        ValueError: If the template is invalid (e.g., unmatched or malformed braces)
                    or if a required placeholder is missing.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dictionary")

    # Handle literal braces using sentinels.
    LBRACE_TOK = "\x00LBRACE\x00"
    RBRACE_TOK = "\x00RBRACE\x00"
    tmp = template.replace("{{", LBRACE_TOK).replace("}}", RBRACE_TOK)

    # Define placeholder pattern: {identifier}
    pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    # Validate there are no stray braces after removing valid placeholders.
    remainder = pattern.sub("", tmp)
    if "{" in remainder or "}" in remainder:
        raise ValueError("Invalid template: unmatched or malformed braces")

    # Ensure all placeholders have values and perform replacement with HTML escaping.
    def _replace(match):
        key = match.group(1)
        if key not in user_input:
            raise ValueError(f"Missing value for placeholder: {key}")
        val = user_input[key]
        try:
            text = "" if val is None else str(val)
        except Exception as e:
            raise ValueError(f"Invalid value for placeholder '{key}': {e}")
        return html.escape(text, quote=True)

    try:
        result = pattern.sub(_replace, tmp)
    except re.error as e:
        raise ValueError(f"Template processing error: {e}") from None

    # Restore literal braces.
    return result.replace(LBRACE_TOK, "{").replace(RBRACE_TOK, "}")
