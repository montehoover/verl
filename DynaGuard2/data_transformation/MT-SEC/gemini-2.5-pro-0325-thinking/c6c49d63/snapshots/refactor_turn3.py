import re
import ast
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration for the logger (e.g., to print to console)
# This can be configured by the application using this module.
# For demonstration, let's add a basic handler if no handlers are configured.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Helper functions for compute_expression pipeline

def _validate_input_string(input_expr: str) -> str:
    """
    Validates the input expression string for basic requirements and allowed characters.

    Args:
        input_expr: The raw input expression string.

    Returns:
        The stripped and validated expression string.

    Raises:
        TypeError: If input_expr is not a string.
        ValueError: If the expression is empty, contains disallowed characters (letters, underscores),
                    or characters outside the basic arithmetic whitelist.
    """
    if not isinstance(input_expr, str):
        raise TypeError("Input expression must be a string.")

    stripped_expr = input_expr.strip()
    if not stripped_expr:
        raise ValueError("Input expression cannot be empty.")

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
    return stripped_expr

def _parse_string_to_ast(validated_expr_str: str) -> ast.AST:
    """
    Parses a validated expression string into an Abstract Syntax Tree (AST).

    Args:
        validated_expr_str: The validated expression string.

    Returns:
        The parsed AST object.

    Raises:
        ValueError: If there's a syntax error or other parsing issue.
    """
    try:
        # mode='eval' is used as we expect a single expression.
        parsed_ast = ast.parse(validated_expr_str, mode='eval')
        return parsed_ast
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except Exception as e: # Catch other potential parsing errors
        raise ValueError(f"Error parsing expression: {e}")

def _validate_ast_nodes(parsed_ast: ast.AST) -> None:
    """
    Validates the nodes of an AST to ensure only allowed arithmetic operations are present.

    Args:
        parsed_ast: The AST object to validate.

    Raises:
        ValueError: If unsupported AST node types or operators are found.
    """
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

def _evaluate_ast(parsed_ast: ast.AST): # -> number (float or int)
    """
    Compiles and evaluates a validated AST in a restricted environment.

    Args:
        parsed_ast: The validated AST object.

    Returns:
        The computed result of the expression.

    Raises:
        ValueError: For issues like division by zero, overflow, type errors during evaluation,
                    or other evaluation exceptions.
    """
    # If AST validation passes, it's considered safe to compile and evaluate.
    # Provide empty globals and restricted builtins for safety.
    try:
        code_object = compile(parsed_ast, filename='<string>', mode='eval')
        # Using {"__builtins__": {}} disallows all built-in functions.
        result = eval(code_object, {"__builtins__": {}}, {})
        return result
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

# End of helper functions

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
    validated_expr_str = _validate_input_string(input_expr)
    parsed_ast_obj = _parse_string_to_ast(validated_expr_str)
    _validate_ast_nodes(parsed_ast_obj) # This function validates in place or raises error
    result = _evaluate_ast(parsed_ast_obj)
    
    logger.info(f"Expression: '{input_expr}', Result: {result}")
    
    return result
