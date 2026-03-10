import re

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Checks if the user_input string contains dangerous URL schemes like 'javascript' or 'vbscript'.
    It ignores legitimate image data URLs (i.e., data:image/... is not considered dangerous by itself,
    unless it internally uses a dangerous scheme).

    Args:
        user_input: The string to scan.

    Returns:
        True if a dangerous scheme is found, False otherwise.
    """
    try:
        # Pattern to find "javascript:" or "vbscript:" schemes.
        # (?<![a-zA-Z0-9+.-]) is a negative lookbehind asserting that the scheme is not preceded by
        # characters that could form a longer valid scheme name (e.g. "my-javascript:").
        # (javascript|vbscript) matches either "javascript" or "vbscript".
        # \s* allows for optional whitespace between the scheme name and the colon.
        # : matches the colon character.
        # re.IGNORECASE makes the match case-insensitive.
        dangerous_scheme_pattern = re.compile(
            r"(?<![a-zA-Z0-9+.-])(javascript|vbscript)\s*:", 
            re.IGNORECASE
        )
    except re.error:
        # In case of an error compiling the regex, treat it as if no dangerous scheme was found.
        return False

    if dangerous_scheme_pattern.search(user_input):
        return True
    
    return False

if __name__ == '__main__':
    test_cases = {
        "javascript:alert(1)": True,
        "vbscript:msgbox('hi')": True,
        "JAVASCRIPT:alert(1)": True,
        "  javascript:alert(1)": True, # Leading space, matches
        "javascript : alert(0)": True, # Space before colon, matches
        "http://example.com": False,
        "data:image/png;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7": False,
        "data:text/plain,Hello": False,
        "Some text with javascript:alert('evil') inside": True,
        "data:text/html,<a href='javascript:alert(1)'>Click me</a>": True,
        "data:image/svg+xml,<svg><desc>This is javascript: a comment</desc></svg>": True,
        "data:image/svg+xml,<svg><script>var url = 'javascript:void(0)';</script></svg>": True,
        "No schemes here.": False,
        "myjavascript:alert(0)": False, # Not preceded by boundary for "javascript"
        "custom-javascript:alert(0)": False, # "javascript:" is part of "custom-javascript:"
        "custom+javascript:alert(0)": False,
        "custom.javascript:alert(0)": False,
        "ajavascript:alert(0)": False,
        "_javascript:alert(0)": True, # Underscore is not a scheme char, so "javascript:" is standalone
        "(javascript:alert(0))": True, # Parenthesis is not a scheme char
    }

    for text, expected in test_cases.items():
        result = contains_dangerous_scheme(text)
        print(f"Input: \"{text}\"\nExpected: {expected}, Got: {result} -> {'Pass' if result == expected else 'Fail'}")
        assert result == expected, f"Test failed for input: {text}"
    
    print("\nAll tests passed.")
