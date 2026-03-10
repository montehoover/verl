import re

def is_javascript_scheme(s: str) -> bool:
    """
    Determine if the string contains a possibly malicious URL scheme.

    Flags as malicious if:
    - It contains javascript:, jscript:, vbscript:, or livescript: (case-insensitive).
    - It contains a data: URL that is not an image (i.e., data: not followed by image/).

    Returns:
        bool: True if a possibly malicious scheme is found (excluding image data URLs), else False.
    """
    script_pattern = r'(?i)\b(?:javascript|jscript|vbscript|livescript)\s*:'
    data_non_image_pattern = r'(?i)\bdata\s*:\s*(?!image/)'

    return bool(
        re.search(script_pattern, s) or
        re.search(data_non_image_pattern, s)
    )
