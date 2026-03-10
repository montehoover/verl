import ast
import operator as op

# Supported binary operators
_SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

# Supported unary operators
_SUPPORTED_UNARY_OPERATORS = {
    ast.USub: op.neg,  # Unary minus
    ast.UAdd: op.pos,  # Unary plus
}

def _eval_node(node):
    """
    Recursively evaluates an AST node for an arithmetic expression.
    Supports: numbers (int, float), +, -, *, /, unary -, unary +.
    Parentheses are handled by the AST structure.
    """
    if isinstance(node, ast.Constant):  # Python 3.8+
        if not isinstance(node.value, (int, float)):
            raise TypeError(f"Unsupported constant type: {type(node.value)}. Only int/float allowed.")
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        # In older Python versions, node.n holds the numeric value.
        if not isinstance(node.n, (int, float)):
            raise TypeError(f"Unsupported number type: {type(node.n)}. Only int/float allowed.")
        return node.n
    elif isinstance(node, ast.BinOp):
        operator_func = _SUPPORTED_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise TypeError(f"Unsupported binary operator: {type(node.op)}")
        
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        
        if isinstance(node.op, ast.Div) and right_val == 0:
            raise ZeroDivisionError("division by zero")
            
        return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        operator_func = _SUPPORTED_UNARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
        
        operand_val = _eval_node(node.operand)
        return operator_func(operand_val)
    else:
        # This will catch ast.Name (variables), ast.Call (function calls),
        # ast.Compare, ast.BoolOp, etc., effectively disallowing them.
        raise TypeError(
            f"Unsupported AST node type: {type(node).__name__}. "
            "Only numbers, basic arithmetic operators (+, -, *, /), "
            "unary minus/plus, and parentheses are allowed."
        )

def basic_calculator(expression: str) -> float:
    """
    Parses and evaluates a simple arithmetic expression string,
    supporting addition, subtraction, multiplication, division, parentheses,
    and unary plus/minus.

    Args:
        expression: The arithmetic expression string (e.g., "4 + 3", "-(5 - 2) * 3 / 2").

    Returns:
        The result of the calculation as a float.

    Raises:
        SyntaxError: If the expression string is malformed or contains invalid syntax.
        TypeError: If the expression contains unsupported operations, data types, or constructs
                   (e.g., variables, function calls, comparisons).
        ZeroDivisionError: If division by zero is attempted.
        ValueError: If the expression is empty or results in a non-numeric value
                    (though type errors should catch most non-numeric issues).
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string.")
    
    stripped_expression = expression.strip()
    if not stripped_expression:
        raise ValueError("Expression cannot be empty.")

    try:
        # ast.parse with mode='eval' expects a single expression.
        # The result is an ast.Expression node, whose 'body' attribute
        # contains the actual root node of the expression (e.g., ast.BinOp, ast.Constant).
        ast_tree = ast.parse(stripped_expression, mode='eval')
        result = _eval_node(ast_tree.body)
        
        # Ensure the final result is a number, as _eval_node should always return one.
        # This is a safeguard; _eval_node's type checks should prevent non-numeric results.
        if not isinstance(result, (int, float)):
            raise ValueError(f"Expression evaluated to a non-numeric type: {type(result).__name__}")
            
        return float(result)
    except SyntaxError:
        # Re-raise SyntaxError to indicate issues with the expression's structure.
        raise SyntaxError("Invalid syntax in expression.")
    except TypeError as e:
        # Re-raise TypeErrors from _eval_node, which indicate unsupported operations/types.
        raise TypeError(str(e))
    except ZeroDivisionError:
        # Re-raise ZeroDivisionError.
        raise ZeroDivisionError("Cannot divide by zero.")
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation.
        # This could include RecursionError for deeply nested expressions.
        raise ValueError(f"Error evaluating expression: {e}")
