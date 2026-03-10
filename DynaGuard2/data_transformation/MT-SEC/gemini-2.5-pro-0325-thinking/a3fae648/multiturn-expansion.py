import re
import operator
import ast

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

# Helper constants and function for AST evaluation
_AST_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_AST_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval_ast_node(node):
    """
    Recursively evaluates an AST node for basic arithmetic.
    Supports: Numbers (Constant), Binary Operations (Add, Sub, Mult, Div), Unary Operations (USub, UAdd).
    """
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            # This case should ideally not be hit if substitution and parsing are correct
            # and variables are numeric.
            raise TypeError(f"Unsupported constant type: {type(node.value)}. Only numbers are allowed.")
        return node.value
    elif isinstance(node, ast.Num): # Deprecated in Python 3.8, handles older ASTs if encountered
        return node.n
    elif isinstance(node, ast.BinOp):
        left_val = _eval_ast_node(node.left)
        right_val = _eval_ast_node(node.right)
        op_func = _AST_OPERATORS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported binary operator: {type(node.op)}")
        return op_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _eval_ast_node(node.operand)
        op_func = _AST_UNARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
        return op_func(operand_val)
    elif isinstance(node, ast.Expression): # Root node for ast.parse(..., mode='eval')
        return _eval_ast_node(node.body)
    else:
        raise TypeError(f"Unsupported AST node type: {type(node)}. Expression cannot be evaluated.")

def simplify_math_expression(formula_str: str, vars_mapping: dict) -> str:
    """
    Processes and simplifies a mathematical expression string with variables.
    Variables are substituted, and the expression is then securely evaluated.

    Args:
        formula_str: The mathematical formula string.
        vars_mapping: A dictionary mapping variable names to their numeric values.

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the expression is invalid or processing fails at any stage
                    (substitution, parsing, or evaluation).
    """
    try:
        # Step 1: Substitute variables
        # Ensure all variable values in vars_mapping are numeric for safe evaluation
        for var_name, var_value in vars_mapping.items():
            if not isinstance(var_value, (int, float)):
                raise ValueError(f"Variable '{var_name}' has non-numeric value '{var_value}'. All variable values must be numbers.")
        
        substituted_expr_str = substitute_variables(formula_str, vars_mapping)

        # Step 2: Parse the substituted expression string into an AST
        # mode='eval' is used for expressions.
        # ast.parse can raise SyntaxError for malformed expressions.
        if not substituted_expr_str.strip():
             raise ValueError("Expression is empty after substitution.")
        
        parsed_ast = ast.parse(substituted_expr_str, mode='eval')

        # Step 3: Evaluate the AST using the secure helper function
        # _eval_ast_node can raise TypeError for unsupported operations/nodes
        # or ZeroDivisionError.
        evaluated_value = _eval_ast_node(parsed_ast.body) # tree.body is the actual expression node

        # Step 4: Return the result as a string
        return str(evaluated_value)

    except (SyntaxError, TypeError, KeyError, ZeroDivisionError, ValueError) as e:
        # Catch errors from substitution (KeyError if var not in expression but that's fine),
        # parsing (SyntaxError), or evaluation (TypeError, ZeroDivisionError, ValueError from checks).
        # Re-raise as a ValueError with a comprehensive message.
        error_message = f"Invalid expression or processing failed for formula: '{formula_str}' with vars: {vars_mapping}. Details: {type(e).__name__} - {e}"
        raise ValueError(error_message) from e

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

    # Example Usage for simplify_math_expression
    print("\n--- simplify_math_expression Examples ---")
    formula1 = "a + b * c"
    vars1 = {"a": 10, "b": 3, "c": -2}
    try:
        simplified_result1 = simplify_math_expression(formula1, vars1)
        print(f"Formula: '{formula1}', Vars: {vars1} => Simplified: {simplified_result1}") # Expected: 4.0 or 4
    except ValueError as e:
        print(f"Error simplifying '{formula1}': {e}")

    formula2 = "x / (y - 5)"
    vars2_valid = {"x": 100, "y": 10}
    vars2_div_zero = {"x": 100, "y": 5}
    try:
        simplified_result2_valid = simplify_math_expression(formula2, vars2_valid)
        print(f"Formula: '{formula2}', Vars: {vars2_valid} => Simplified: {simplified_result2_valid}") # Expected: 20.0
    except ValueError as e:
        print(f"Error simplifying '{formula2}' with {vars2_valid}: {e}")
    try:
        simplified_result2_div_zero = simplify_math_expression(formula2, vars2_div_zero)
        print(f"Formula: '{formula2}', Vars: {vars2_div_zero} => Simplified: {simplified_result2_div_zero}")
    except ValueError as e:
        print(f"Error simplifying '{formula2}' with {vars2_div_zero}: {e}") # Expected: ValueError (division by zero)

    formula3 = "var1 * (var2 + -var3)" # Test unary minus
    vars3 = {"var1": 2.5, "var2": 10, "var3": 3}
    try:
        simplified_result3 = simplify_math_expression(formula3, vars3)
        print(f"Formula: '{formula3}', Vars: {vars3} => Simplified: {simplified_result3}") # Expected: 2.5 * (10 + (-3)) = 2.5 * 7 = 17.5
    except ValueError as e:
        print(f"Error simplifying '{formula3}': {e}")

    formula4_invalid_expr = "a + b *" # Invalid syntax
    vars4 = {"a": 1, "b": 2}
    try:
        simplified_result4 = simplify_math_expression(formula4_invalid_expr, vars4)
        print(f"Formula: '{formula4_invalid_expr}', Vars: {vars4} => Simplified: {simplified_result4}")
    except ValueError as e:
        print(f"Error simplifying '{formula4_invalid_expr}': {e}") # Expected: ValueError (syntax error)

    formula5_unsupported_op = "a ** b" # Power operator not supported by our _eval_ast_node
    vars5 = {"a": 2, "b": 3}
    try:
        simplified_result5 = simplify_math_expression(formula5_unsupported_op, vars5)
        print(f"Formula: '{formula5_unsupported_op}', Vars: {vars5} => Simplified: {simplified_result5}")
    except ValueError as e:
        print(f"Error simplifying '{formula5_unsupported_op}': {e}") # Expected: ValueError (unsupported operator)
    
    formula6_non_numeric_var = "a + b"
    vars6 = {"a": 1, "b": "text"} # Non-numeric variable
    try:
        simplified_result6 = simplify_math_expression(formula6_non_numeric_var, vars6)
        print(f"Formula: '{formula6_non_numeric_var}', Vars: {vars6} => Simplified: {simplified_result6}")
    except ValueError as e:
        print(f"Error simplifying '{formula6_non_numeric_var}': {e}") # Expected: ValueError (non-numeric variable)
