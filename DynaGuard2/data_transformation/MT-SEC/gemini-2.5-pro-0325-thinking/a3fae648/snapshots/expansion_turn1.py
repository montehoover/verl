import re

def substitute_variables(expression: str, variables: dict) -> str:
    """
    Substitutes variables in a mathematical expression string with their values.

    Args:
        expression: The mathematical expression string.
        variables: A dictionary mapping variable names (str) to their values (str or numeric).

    Returns:
        A new string with variables replaced by their values.
    """
    substituted_expression = expression
    # Sort variables by length in descending order to handle cases like 'x' and 'xy' correctly.
    # If 'x' is replaced before 'xy', 'xy' might become '{value_of_x}y'.
    sorted_vars = sorted(variables.keys(), key=len, reverse=True)

    for var_name in sorted_vars:
        # Use \b for word boundaries to match whole words only
        # Escape variable name in case it contains regex special characters
        pattern = r'\b' + re.escape(var_name) + r'\b'
        substituted_expression = re.sub(pattern, str(variables[var_name]), substituted_expression)
    return substituted_expression

if __name__ == '__main__':
    # Example Usage
    expr = "x + y * 2 - z / x"
    vals = {"x": 10, "y": "5", "z": 2}
    result = substitute_variables(expr, vals)
    print(f"Original expression: {expr}")
    print(f"Variables: {vals}")
    print(f"Substituted expression: {result}")

    expr2 = "alpha + beta * gamma - alpha1"
    vals2 = {"alpha": "3", "beta": 4, "gamma": "PI", "alpha1": 1}
    result2 = substitute_variables(expr2, vals2)
    print(f"Original expression: {expr2}")
    print(f"Variables: {vals2}")
    print(f"Substituted expression: {result2}")

    expr3 = "var + var_long + another_var"
    vals3 = {"var": "1", "var_long": "100"}
    result3 = substitute_variables(expr3, vals3) # 'another_var' should remain unchanged
    print(f"Original expression: {expr3}")
    print(f"Variables: {vals3}")
    print(f"Substituted expression: {result3}")

    expr4 = "x_coord + y_coord"
    vals4 = {"x": "1", "y": "2"} # Should not substitute x in x_coord
    result4 = substitute_variables(expr4, vals4)
    print(f"Original expression: {expr4}")
    print(f"Variables: {vals4}")
    print(f"Substituted expression: {result4}")
