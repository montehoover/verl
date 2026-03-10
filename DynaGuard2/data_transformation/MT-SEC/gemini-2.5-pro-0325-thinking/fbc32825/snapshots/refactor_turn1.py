import re

def match_strings(text: str):
    """
    Matches a string and captures the text outside and inside parentheses.

    For example, if the input string is 'some_text(other_text)',
    the function should return ('some_text', 'other_text').

    Args:
        text: The input string to be matched.

    Returns:
        A tuple containing the text outside the parentheses and the
        text inside the parentheses if the pattern is matched.
        Otherwise, returns None.
    """
    pattern = r'([^(]*)\(([^)]*)\)'
    match = re.match(pattern, text)
    if match:
        # Ensure the entire string is matched if it ends with ')'
        # and there are no other characters after the closing parenthesis.
        # For example, 'func(arg))' or 'func(arg)extra' should not match fully.
        # The regex r'([^(]*)\(([^)]*)\)$' would be more strict
        # but the current one r'([^(]*)\(([^)]*)\)' allows for text after ')'
        # which might be desired or not depending on exact requirements.
        # Given 'some_text(other_text)', it should return ('some_text', 'other_text').
        # Let's refine the pattern to ensure it matches the example strictly.
        # A better pattern might be r'^([^(]+)\(([^)]+)\)$' to ensure
        # it matches the whole string and there's content before and inside parentheses.
        # However, the prompt implies matching "some_text(other_text)" and capturing.
        # The initial pattern `r'([^(]*)\(([^)]*)\)'` captures:
        # 'func()' -> ('func', '')
        # '(arg)' -> ('', 'arg')
        # 'func(arg)' -> ('func', 'arg')
        # 'func(arg)extra' -> ('func', 'arg') - re.match matches from the beginning.
        # 'prefix_func(arg)' -> ('prefix_func', 'arg')

        # Let's use a pattern that strictly matches the 'text(text)' format
        # and ensures the entire string conforms to this.
        strict_pattern = r'^([^(]+)\(([^)]+)\)$'
        strict_match = re.fullmatch(strict_pattern, text)
        if strict_match:
            return strict_match.groups()
        else:
            # Fallback to the less strict pattern if the example implies flexibility
            # or if the "sqlparse framework" context suggests partial matches are okay.
            # For now, sticking to the example 'some_text(other_text)' -> ('some_text', 'other_text')
            # The initial pattern `r'([^(]*)\(([^)]*)\)'` with `re.match` is fine if the string
            # must start with this pattern. If it can be anywhere, `re.search` is better.
            # The prompt says "match a string", implying the whole string.
            # So, `re.fullmatch` with a pattern like `([^(]*)\(([^)]*)\)` is appropriate.
            # Let's use `([^(]+)\(([^)]+)\)` to ensure there's text before and inside.
            # If empty parts are allowed, `*` should be used instead of `+`.
            # The example 'some_text(other_text)' has non-empty parts.
            
            # Re-evaluating the pattern based on "capture the text outside the parentheses and the text inside"
            # "outside" means "before the first opening parenthesis".
            # "inside" means "between the first opening parenthesis and its corresponding closing parenthesis".
            # The initial pattern `r'([^(]*)\(([^)]*)\)'` seems most direct for this.
            # `re.match` will anchor it at the beginning of the string.

            # Let's use the pattern that captures text before the first '(' and text within the first '()'.
            # Example: 'foo(bar)baz' -> ('foo', 'bar')
            # Example: 'foo(bar(baz))' -> ('foo', 'bar(baz') - if non-greedy `[^)]*`
            # Example: 'foo(bar(baz))' -> ('foo', 'bar(baz))') - if greedy `.*` inside `()`
            # The prompt example 'some_text(other_text)' is simple.
            # `([^(]*)\(([^)]*)\)` with `re.match`
            # 'some_text(other_text)' -> match.groups() -> ('some_text', 'other_text') - Correct.
            # 'some_text()' -> match.groups() -> ('some_text', '') - Potentially valid.
            # '(other_text)' -> match.groups() -> ('', 'other_text') - Potentially valid.
            # '()' -> match.groups() -> ('', '') - Potentially valid.
            # 'no_parentheses' -> None - Correct.
            # 'text(inside)trailing' -> match.groups() -> ('text', 'inside') - `re.match` matches if the start of the string fits.
            
            # The problem asks to "capture the text outside the parentheses and the text inside the parentheses".
            # This implies a structure like `OUTSIDE_TEXT ( INSIDE_TEXT )`.
            # The pattern `r'([^(]*)\(([^)]*)\)'` seems to fit this best for `re.match`.
            # If the string must *only* be `OUTSIDE_TEXT ( INSIDE_TEXT )`, then `re.fullmatch` with `r'([^(]*)\(([^)]*)\)$'`
            # or `r'([^(]*)\(([^)]*)\)'` would be better.
            # Given the example 'some_text(other_text)', it's likely the whole string is expected to match this pattern.
            
            final_pattern = r'^([^(]*)\(([^)]*)\)$' # Ensures the string is fully matched
            full_match_obj = re.fullmatch(final_pattern, text)
            if full_match_obj:
                return full_match_obj.groups()
    return None
