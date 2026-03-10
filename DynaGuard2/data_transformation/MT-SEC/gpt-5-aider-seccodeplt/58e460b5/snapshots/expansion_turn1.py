import re

def substitute_variables(expression: str, variables: dict) -> str:
    """
    Substitute variable names in the given expression with their values from the variables dict.
    Only identifiers of the form [A-Za-z_]\\w* are considered variables.
    Identifiers not present in the dict are left unchanged.
    """
    if not expression:
        return expression
    if not variables:
        return expression

    pattern = re.compile(r"[A-Za-z_]\w*")

    def replace(match: re.Match) -> str:
        name = match.group(0)
        if name in variables:
            return str(variables[name])
        return name

    return pattern.sub(replace, expression)
