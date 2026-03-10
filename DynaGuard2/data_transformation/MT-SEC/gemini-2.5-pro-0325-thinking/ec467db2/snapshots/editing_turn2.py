import ast
import operator as op

# Supported operators and their corresponding ast node types
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

def _eval_ast_node(node):
    """
    Recursively evaluates an AST node.
    Raises ValueError for unsupported node types or operations.
    """
    if isinstance(node, ast.Constant):
        # In Python 3.8+, ast.Constant is used for numbers, strings, None, True, False.
        # We allow numeric constants (int, float) and string constants (str).
        if not isinstance(node.value, (int, float, str)): # Allow strings
            raise TypeError("Unsupported constant type; only numbers (int, float) or strings (str) are allowed.")
        return node.value
    elif isinstance(node, ast.Num): # For Python < 3.8 compatibility (ast.Num holds numeric literals)
        # This node type is for numeric literals only.
        if not isinstance(node.n, (int, float)):
            raise TypeError("Unsupported number type.")
        return node.n
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        
        left_val = _eval_ast_node(node.left)
        right_val = _eval_ast_node(node.right)

        if op_type == ast.Add:
            # For '+', allow num+num or str+str
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return op.add(left_val, right_val)
            elif isinstance(left_val, str) and isinstance(right_val, str):
                return left_val + right_val # String concatenation
            else:
                raise TypeError(f"Unsupported operand types for +: {type(left_val).__name__} and {type(right_val).__name__}")
        else:
            # For other operators (Sub, Mult, Div), they must be numeric
            operator_func = _ALLOWED_OPERATORS.get(op_type)
            if operator_func is None: # Should only catch ops not in _ALLOWED_OPERATORS (e.g. Bitwise ops)
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            if not (isinstance(left_val, (int, float)) and isinstance(right_val, (int, float))):
                op_symbol = ""
                if op_type == ast.Sub: op_symbol = "-"
                elif op_type == ast.Mult: op_symbol = "*"
                elif op_type == ast.Div: op_symbol = "/"
                else: op_symbol = op_type.__name__
                raise TypeError(f"Numeric operands required for '{op_symbol}' operator. Got {type(left_val).__name__} and {type(right_val).__name__}.")

            if op_type == ast.Div and right_val == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            
            return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub): # Unary minus (e.g., -5)
            operand_val = _eval_ast_node(node.operand)
            return -operand_val
        # ast.UAdd (unary plus, e.g., +5) could be added if needed, though often redundant.
        # elif isinstance(node.op, ast.UAdd):
        #     return _eval_ast_node(node.operand)
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.Expression): # Root node when ast.parse is called with mode='eval'
        return _eval_ast_node(node.body)
    else:
        # This will disallow ast.Name, ast.Call, ast.Attribute, ast.Lambda,
        # ast.ListComp, ast.DictComp, ast.SetComp, ast.Subscript, etc.
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

def calculate_expression(expression: str):
    """
    Calculates a simple arithmetic or string expression string and returns the result.

    The expression can include:
    - Numbers (integers and floats) with operators: + (addition), - (subtraction), 
      * (multiplication), / (division).
    - Strings with operator: + (concatenation).
    Parentheses for grouping are supported. Unary minus (e.g., -5) is supported for numbers.

    Args:
        expression: A string containing the arithmetic or string expression.

    Returns:
        The numerical or string result of the expression if it's valid and safe.
        Returns the string 'Execution Blocked!' if the expression is invalid,
        unsafe, or results in an error (e.g., division by zero, type mismatch, syntax error).
    """
    try:
        if not isinstance(expression, str):
            return 'Execution Blocked!'

        # Basic pre-check for common malicious patterns or overly long expressions.
        # The primary safety mechanism is the AST traversal and selective evaluation.
        if "__" in expression: # Disallow dunder attributes/methods
            return 'Execution Blocked!'
        
        # Limit expression length to prevent potential DoS via extremely long inputs.
        # The exact limit can be adjusted based on requirements.
        if len(expression) > 500: # Max expression length
            return 'Execution Blocked!'

        # Parse the expression string into an Abstract Syntax Tree (AST).
        # mode='eval' is used because we are evaluating a single expression.
        parsed_ast = ast.parse(expression, mode='eval')
        
        # Evaluate the AST using our restricted evaluator.
        # _eval_ast_node will raise an exception for any disallowed operations or node types.
        result = _eval_ast_node(parsed_ast)

        # Ensure the final result is a number or a string. This should be guaranteed 
        # by _eval_ast_node if it correctly handles types.
        if not isinstance(result, (int, float, str)):
            # This case should ideally not be reached if _eval_ast_node is correct.
            return 'Execution Blocked!'
            
        return result

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError) as e:
        # Catch errors related to parsing, unsupported operations, type issues,
        # division by zero, or too much recursion (for overly complex expressions).
        # Optionally, log the error `e` for debugging purposes if needed.
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected exceptions to be safe.
        # Optionally, log the error `e` for debugging purposes if needed.
        return 'Execution Blocked!'

if __name__ == '__main__':
    # Example Usage and Basic Tests
    test_expressions = {
        "1 + 1": 2,
        "2 * 3": 6,
        "10 - 4": 6,
        "10 / 2": 5.0,
        "2 * (3 + 4)": 14,
        "(10 - 2) * 3": 24,
        "10 / 4": 2.5,
        "-5": -5,
        "-5 + 10": 5,
        "5 + -2": 3,
        "-(5 + 2)": -7,
        "1.5 + 2.5": 4.0,
        "10 / 0": "Execution Blocked!",
        "1 + foo": "Execution Blocked!",
        "__import__('os').system('clear')": "Execution Blocked!",
        "1 + (lambda: 1)()": "Execution Blocked!",
        "1; 2": "Execution Blocked!", # SyntaxError
        "1 +": "Execution Blocked!", # SyntaxError
        "eval('1+1')": "Execution Blocked!", # ast.Call not allowed
        "a = 1": "Execution Blocked!", # ast.Assign not allowed in eval mode (SyntaxError)
        "[1, 2, 3]": "Execution Blocked!", # ast.List not allowed
        "{'a': 1}": "Execution Blocked!", # ast.Dict not allowed
        "'hello' + ' ' + 'world'": "hello world",
        "'foo' + 'bar'": "foobar",
        "1 + 'bar'": "Execution Blocked!", 
        "'foo' + 1": "Execution Blocked!",
        "'foo' * 2": "Execution Blocked!", # String multiplication not supported with '*'
        "'foo' - 'o'": "Execution Blocked!", # Subtraction not supported for strings
        "True": "Execution Blocked!", # Boolean constants not allowed
    }

    for expr, expected in test_expressions.items():
        result = calculate_expression(expr)
        print(f"Expression: \"{expr}\", Expected: {expected}, Got: {result}, Match: {result == expected}")

    # Test for overly long expression
    long_expr = "1" + "+1" * 1000
    print(f"Expression: \"1+...+1 (long)\", Expected: Execution Blocked!, Got: {calculate_expression(long_expr)}")
    
    short_valid_expr = "1" + "+1" * 100 # Should be less than 500 chars
    # This will be 101. Length is 1 + 2*100 = 201.
    # Expected result is 101.
    print(f"Expression: \"1+...+1 (short)\", Expected: 101, Got: {calculate_expression(short_valid_expr)}")
