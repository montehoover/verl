import ast
import operator as op

# Supported operators map AST operator types to corresponding functions
_SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    # ast.Pow: op.pow, # Example: could be added if exponentiation is needed
}

# Supported unary operators
_SUPPORTED_UNARY_OPERATORS = {
    ast.UAdd: lambda x: +x,  # Unary plus
    ast.USub: op.neg,       # Unary minus
}

def _is_ast_safe_to_evaluate(node: ast.AST) -> bool:
    """
    Recursively validates if an AST node and its children are safe for evaluation.
    Only allows whitelisted node types and operations.

    Args:
        node: The AST node to validate.

    Returns:
        True if the node and its subtree are safe, False otherwise.
    """
    if isinstance(node, ast.Constant):
        # Allow only numeric constants (integers or floats)
        return isinstance(node.value, (int, float))
    elif isinstance(node, ast.BinOp):
        # Check if the binary operator is supported
        if type(node.op) not in _SUPPORTED_OPERATORS:
            return False
        # Recursively validate left and right operands
        return _is_ast_safe_to_evaluate(node.left) and \
               _is_ast_safe_to_evaluate(node.right)
    elif isinstance(node, ast.UnaryOp):
        # Check if the unary operator is supported
        if type(node.op) not in _SUPPORTED_UNARY_OPERATORS:
            return False
        # Recursively validate the operand
        return _is_ast_safe_to_evaluate(node.operand)
    else:
        # Any other AST node type is considered unsafe
        return False

def _perform_evaluation_on_safe_ast(node: ast.AST):
    """
    Recursively evaluates an AST node that has already been validated for safety.

    Args:
        node: The AST node to evaluate. Assumed to be safe.

    Returns:
        The numerical result of the evaluation.

    Raises:
        ZeroDivisionError: If division by zero occurs during evaluation.
    """
    if isinstance(node, ast.Constant):
        return node.value  # Value is already validated to be int or float
    elif isinstance(node, ast.BinOp):
        left_val = _perform_evaluation_on_safe_ast(node.left)
        right_val = _perform_evaluation_on_safe_ast(node.right)
        
        operator_func = _SUPPORTED_OPERATORS[type(node.op)] # Operator type is validated
        
        if isinstance(node.op, ast.Div) and right_val == 0:
            raise ZeroDivisionError("Division by zero")
            
        return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operand_val = _perform_evaluation_on_safe_ast(node.operand)
        operator_func = _SUPPORTED_UNARY_OPERATORS[type(node.op)] # Operator type is validated
        return operator_func(operand_val)
    # No 'else' case needed here, as the node type is assumed to be validated.
    # An unexpected node type would indicate a flaw in the validation logic
    # or how this function is called.

def evaluate_arith_expression(input_expr: str) -> str:
    """
    Interprets and computes the result of basic arithmetic expressions.

    Args:
        input_expr: A string representation of a mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the expression is invalid, malicious, or causes an error.
    """
    try:
        # Step 1: Parse the input string into an AST.
        # mode='eval' ensures it's a single expression.
        # ast.parse returns an ast.Expression wrapper node.
        parsed_expression_wrapper = ast.parse(input_expr, mode='eval')
        
        # The actual root of the expression's AST is in the 'body' attribute.
        expr_ast_node = parsed_expression_wrapper.body
        
        # Step 2: Validate the AST for safety.
        # This checks for allowed node types and operations.
        if not _is_ast_safe_to_evaluate(expr_ast_node):
            return 'Computation Error!'
            
        # Step 3: Perform the evaluation on the validated AST.
        result = _perform_evaluation_on_safe_ast(expr_ast_node)
        
        # Step 4: Ensure the final result is a number (int or float).
        # This is a safeguard; _perform_evaluation_on_safe_ast should always return a number
        # if the AST is valid and no runtime errors like ZeroDivisionError occur.
        if not isinstance(result, (int, float)):
            # This path indicates an unexpected issue, possibly a flaw in
            # the validation or evaluation logic if it's ever reached.
            return 'Computation Error!'
            
        return str(result)
    except SyntaxError:
        # Error during ast.parse (e.g., malformed expression)
        return 'Computation Error!'
    except ZeroDivisionError:
        # Specific runtime error during evaluation
        return 'Computation Error!'
    except RecursionError:
        # Expression too complex, leading to excessive recursion in parsing,
        # validation, or evaluation.
        return 'Computation Error!'
    except Exception:
        # Catch-all for any other unexpected exceptions during the process.
        return 'Computation Error!'
