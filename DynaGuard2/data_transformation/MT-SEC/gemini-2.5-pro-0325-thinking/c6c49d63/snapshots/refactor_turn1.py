import re
import ast

def compute_expression(input_expr: str):
    """
    Parses a user-provided mathematical expression string and computes the result.

    Args:
        input_expr: A string representing the arithmetic expression.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: If unsupported characters, unsafe commands,
                    or invalid operations are detected in the input.
    """
    if not isinstance(input_expr, str):
        raise TypeError("Input expression must be a string.")

    stripped_expr = input_expr.strip()
    if not stripped_expr:
        raise ValueError("Input expression cannot be empty.")

    # 1. Initial character validation (pre-filter)
    # Disallow letters and underscores to prevent variable names or function calls.
    if re.search(r"[a-zA-Z_]", stripped_expr):
        raise ValueError("Unsupported characters: letters or underscores are not allowed.")
    
    # Check for any characters not in a basic whitelist.
    # This handles '**' for power and '%' for modulo correctly.
    temp_expr_for_char_check = stripped_expr.replace('**', '') # Temporarily remove '**' to check other chars
    
    allowed_single_chars = set("0123456789.+-*/%() ")
    for char in temp_expr_for_char_check:
        if char not in allowed_single_chars:
            raise ValueError(f"Unsupported character '{char}' found in expression.")

    try:
        # 2. Parse the expression into an AST
        # mode='eval' is used as we expect a single expression.
        parsed_ast = ast.parse(stripped_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except Exception as e: # Catch other potential parsing errors
        raise ValueError(f"Error parsing expression: {e}")

    # 3. Validate the AST nodes
    # Whitelist of allowed AST node types for simple arithmetic expressions.
    # ast.Num is for Python < 3.8, ast.Constant for Python 3.8+. Both are included for compatibility.
    # ast.Load is a context node, not an operation itself.
    allowed_node_types_structural = (
        ast.Expression,  # Root node for an expression.
        ast.Constant,    # For numbers and constants (Python 3.8+).
        ast.Num,         # For numbers (Python < 3.8).
        ast.BinOp,       # For binary operations (e.g., a + b).
        ast.UnaryOp      # For unary operations (e.g., -a).
    )
    # Whitelist of allowed operator types for BinOp and UnaryOp.
    allowed_bin_op_operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    allowed_unary_op_operators = (ast.USub, ast.UAdd) # UAdd for explicit positive like `+5`.

    for node in ast.walk(parsed_ast):
        node_type = type(node)

        if node_type in allowed_node_types_structural:
            # These are the expected structural nodes.
            # Further checks for operators within BinOp/UnaryOp:
            if node_type is ast.BinOp:
                if not isinstance(node.op, allowed_bin_op_operators):
                    raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            elif node_type is ast.UnaryOp:
                if not isinstance(node.op, allowed_unary_op_operators):
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        elif node_type in allowed_bin_op_operators or \
             node_type in allowed_unary_op_operators or \
             node_type is ast.Load:
            # These are operator types (e.g., ast.Add) or context types (ast.Load).
            # They are valid components of the allowed structural nodes.
            # ast.Load is a context, e.g., for Name(id='x', ctx=Load()).
            # Since ast.Name itself is disallowed (it's not in allowed_node_types_structural),
            # this primarily acknowledges that these types can appear during ast.walk.
            pass
        else:
            # Any other node type is disallowed (e.g., ast.Name, ast.Call, ast.Attribute, etc.)
            raise ValueError(
                f"Unsupported element in expression: {node_type.__name__}. "
                "Only basic arithmetic operations are allowed."
            )

    # 4. Compile and evaluate the AST
    # If AST validation passes, it's considered safe to compile and evaluate.
    # Provide empty globals and restricted builtins for safety.
    try:
        code_object = compile(parsed_ast, filename='<string>', mode='eval')
        # Using {"__builtins__": {}} disallows all built-in functions.
        result = eval(code_object, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except OverflowError:
        raise ValueError("Calculation resulted in an overflow (number too large).")
    except TypeError as e: 
        # Handles cases like "unsupported operand type(s) for ** or pow(): 'float' and 'float'" if it leads to complex numbers
        # or other type mismatches not caught by AST (though less likely with current restrictions).
        raise ValueError(f"Type error during evaluation: {e}")
    except Exception as e:
        # Catch any other runtime errors during evaluation
        raise ValueError(f"Error during expression evaluation: {e}")

    return result
