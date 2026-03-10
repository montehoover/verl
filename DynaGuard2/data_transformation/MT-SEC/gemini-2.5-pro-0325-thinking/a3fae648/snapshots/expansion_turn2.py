import re
import operator

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

# The if __name__ == '__main__': block was already modified in the previous SEARCH/REPLACE.
# This block is to remove the old main content.

def evaluate_expression(expression_string: str) -> float:
    """
    Evaluates a mathematical expression string (with no variables) and returns the result.

    Args:
        expression_string: The mathematical expression string.

    Returns:
        The computed result as a float.
    """
    try:
        # Using eval() for simplicity. Be cautious with eval() if the input string is not trusted.
        # For basic arithmetic operations on substituted expressions, it's generally fine.
        result = eval(expression_string)
        return float(result)
    except Exception as e:
        print(f"Error evaluating expression '{expression_string}': {e}")
        # Or raise a custom exception
        raise ValueError(f"Invalid expression for evaluation: {expression_string}") from e

if __name__ == '__main__':
    # Example Usage for substitute_variables (existing)
    expr = "x + y * 2 - z / x"
    vals = {"x": 10, "y": "5", "z": 2}
    substituted_expr = substitute_variables(expr, vals)
    print(f"\nOriginal expression: {expr}")
    print(f"Variables: {vals}")
    print(f"Substituted expression: {substituted_expr}")
    # Example for evaluate_expression
    evaluated_result = evaluate_expression(substituted_expr)
    print(f"Evaluated result: {evaluated_result}")

    expr2 = "alpha + beta * gamma - alpha1"
    vals2 = {"alpha": "3", "beta": 4, "gamma": "PI", "alpha1": 1}
    # Note: "PI" is not a number, substitute_variables will make "3 + 4 * PI - 1"
    # This will cause an error in eval unless PI is defined or replaced with a numeric value.
    # For this example, let's assume PI should be a numeric value in the substitution.
    vals2_numeric_pi = {"alpha": "3", "beta": 4, "gamma": "3.14159", "alpha1": 1}
    substituted_expr2 = substitute_variables(expr2, vals2_numeric_pi)
    print(f"\nOriginal expression: {expr2}")
    print(f"Variables (with numeric PI): {vals2_numeric_pi}")
    print(f"Substituted expression: {substituted_expr2}")
    evaluated_result2 = evaluate_expression(substituted_expr2)
    print(f"Evaluated result: {evaluated_result2}")

    # Direct evaluation example
    direct_expr = "(10 + 5) * 2 - 3 / 1.5"
    print(f"\nDirect evaluation of: {direct_expr}")
    direct_result = evaluate_expression(direct_expr)
    print(f"Evaluated result: {direct_result}")

    # Example with an expression that might result from prior substitutions
    expr_from_sub = "100 / 5 + 2 * 3"
    print(f"\nEvaluating: {expr_from_sub}")
    result_from_sub = evaluate_expression(expr_from_sub)
    print(f"Evaluated result: {result_from_sub}")
