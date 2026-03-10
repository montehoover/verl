import re
import operator
import ast

def evaluate_expression_safely(math_expression: str, variable_mapping: dict) -> str:
    """
    Evaluates a mathematical expression string with variable substitution using safe AST evaluation.

    Args:
        math_expression: A string representing a mathematical expression.
        variable_mapping: A dictionary where keys are variable names (str)
                          and values are their corresponding numbers (int or float).

    Returns:
        The result of the evaluated expression as a string.

    Raises:
        ValueError: If the input expression is invalid, contains disallowed elements,
                    or if the computation fails (e.g., undefined variable, division by zero).
    """

    _SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        # ast.Pow is excluded to stick to "basic" arithmetic operations
        ast.USub: operator.neg,
    }

    # Allowed AST node types for a pre-flight safety check.
    # This list includes structural nodes, value nodes, and operator type nodes.
    _ALLOWED_NODE_TYPES = (
        ast.Expression,  # Root node for mode='eval'
        ast.Constant,    # For numbers (and other literals, but we filter for numbers)
        ast.Name,        # For variables
        ast.Load,        # Context for ast.Name (e.g., when a variable is read)
        ast.BinOp,       # For binary operations (e.g., x + y)
        ast.UnaryOp,     # For unary operations (e.g., -x)
        # Specific operator types (these are types of node.op for BinOp/UnaryOp)
        ast.Add, ast.Sub, ast.Mult, ast.Div,  # Binary operators
        ast.USub, ast.UAdd,                   # Unary operators
    )

    def _eval_node_recursive(node: ast.AST) -> float | int:
        """
        Recursively evaluates an AST node.
        This function assumes node types have been validated by the pre-flight check.
        """
        if isinstance(node, ast.Constant):
            # Ensure the constant is a number, as ast.Constant can hold strings, None, etc.
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"Unsupported constant value: {node.value!r}. Only numbers (int, float) are allowed."
                )
            return node.value
        elif isinstance(node, ast.Name):
            # Handles variables
            var_name = node.id
            if var_name in variable_mapping:
                val = variable_mapping[var_name]
                # Ensure the variable's value from the mapping is a number
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Variable '{var_name}' must resolve to a number (int or float). "
                        f"Got type {type(val).__name__} with value {val!r}."
                    )
                return val
            else:
                raise ValueError(f"Unknown variable: '{var_name}'")
        elif isinstance(node, ast.BinOp):
            # Handles binary operations like +, -, *, /
            left_val = _eval_node_recursive(node.left)
            right_val = _eval_node_recursive(node.right)
            op_type = type(node.op)
            if op_type in _SAFE_OPERATORS:
                try:
                    return _SAFE_OPERATORS[op_type](left_val, right_val)
                except ZeroDivisionError:
                    raise ValueError("Division by zero.")
                except Exception as e: # Catch other potential math errors from the operator
                    raise ValueError(f"Error during binary operation '{type(node.op).__name__}': {e}")
            else:
                # This should be caught by the pre-flight check if op_type is not in _ALLOWED_NODE_TYPES
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            # Handles unary operations like - (negation) or + (unary plus)
            operand_val = _eval_node_recursive(node.operand)
            op_type = type(node.op)
            if op_type == ast.USub: # Negation
                return _SAFE_OPERATORS[ast.USub](operand_val)
            elif op_type == ast.UAdd: # Unary plus (e.g., +5)
                return operand_val # operator.pos(operand_val) would also work
            else:
                # This should be caught by the pre-flight check
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        else:
            # This case should ideally not be reached if the pre-flight check (_ALLOWED_NODE_TYPES)
            # is comprehensive and ast.walk(parsed_ast) covers all nodes.
            # It acts as a fallback for unexpected node structures.
            raise ValueError(
                f"Unsupported expression component or structure: {type(node).__name__}"
            )

    # Validate input types
    if not isinstance(math_expression, str):
        raise ValueError("Math expression must be a string.")
    if not isinstance(variable_mapping, dict):
        raise ValueError("Variable mapping must be a dictionary.")

    # Validate variable_mapping keys and values
    for var_key, var_value in variable_mapping.items():
        if not isinstance(var_key, str):
            raise ValueError(f"Variable names in mapping must be strings. Found: {var_key!r}")
        if not isinstance(var_value, (int, float)):
            raise ValueError(
                f"Variable '{var_key}' must have a numeric value (int or float). "
                f"Got type {type(var_value).__name__} with value {var_value!r}."
            )

    try:
        # Parse the expression string into an AST.
        # mode='eval' is crucial: it parses a single expression, not statements,
        # which is inherently safer. The root of this AST is an ast.Expression node.
        parsed_ast = ast.parse(math_expression, mode='eval')

        # Pre-flight safety check: iterate through all nodes in the AST
        # and ensure each node is of an allowed type. This prevents evaluation
        # of complex or potentially unsafe Python constructs that are not simple
        # arithmetic operations.
        for node in ast.walk(parsed_ast):
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                raise ValueError(
                    f"Disallowed element or structure in expression: {type(node).__name__}"
                )
        
        # The body of an 'eval' mode AST (which is ast.Expression) contains the actual expression node.
        result = _eval_node_recursive(parsed_ast.body)
        
        # Ensure the final result is a number before converting to string.
        # This should be guaranteed by _eval_node_recursive if it works correctly.
        if not isinstance(result, (int, float)):
            # This would indicate an internal logic error if reached.
            raise ValueError(f"Evaluation unexpectedly did not result in a number. Got: {type(result)}")
            
        return str(result)

    except SyntaxError as e:
        # ast.parse can raise SyntaxError for malformed expressions.
        raise ValueError(f"Invalid syntax in expression: {e}")
    except ValueError:
        # Re-raise ValueErrors from _eval_node_recursive, pre-flight check, or input validation.
        raise
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation.
        # This provides a generic fallback error message for robustness.
        raise ValueError(f"Failed to evaluate expression due to an unexpected error: {e}")
