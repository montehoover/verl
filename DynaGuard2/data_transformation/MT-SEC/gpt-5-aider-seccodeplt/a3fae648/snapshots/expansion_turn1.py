import re

def substitute_variables(expression: str, variables: dict) -> str:
    """
    Replace variable names in the given mathematical expression string with
    their corresponding values from the provided dictionary.

    - expression: the input expression as a string.
    - variables: mapping of variable name (as str) to a value (any), which will
      be converted to string during substitution.

    Returns the expression string with substitutions applied.
    """
    # Normalize keys to strings for robust matching
    variables_str = {str(k): v for k, v in variables.items()}

    # Match identifiers that don't start with a digit (supports Unicode letters and underscore)
    pattern = re.compile(r'\b(?!\d)\w+\b', flags=re.UNICODE)

    def replace_token(match):
        token = match.group(0)
        if token in variables_str:
            return str(variables_str[token])
        return token

    return pattern.sub(replace_token, expression)
