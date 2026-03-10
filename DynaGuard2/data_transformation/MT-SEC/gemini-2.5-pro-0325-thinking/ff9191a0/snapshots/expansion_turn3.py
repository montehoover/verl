import re
import operator
import ast
from typing import List, Dict, Union, Any

# --- Existing functions parse_expression and substitute_variables ---

def parse_expression(expression: str) -> list[str]:
    """
    Parses a mathematical expression string into a list of tokens.

    Tokens can be numbers (integers or decimals), operators (+, -, *, /, ^, (, )),
    or variable names (alphanumeric, starting with a letter).

    Args:
        expression: The mathematical expression string.

    Returns:
        A list of tokens.
    """
    # Pattern to match numbers (integers and decimals),
    # operators (+, -, *, /, ^, parentheses),
    # and variable names (alphanumeric, starting with a letter, allowing underscores)
    token_pattern = re.compile(r"""
        \s*                               # Optional whitespace
        (
            \d+\.\d* | \.\d+ | \d+ |    # Numbers (float or integer)
            [a-zA-Z_][a-zA-Z0-9_]* |    # Variable names
            [+\-*/^()]                  # Operators and parentheses
        )
        \s*                               # Optional whitespace
    """, re.VERBOSE)
    
    tokens = token_pattern.findall(expression)
    return tokens

def substitute_variables(tokens: List[str], variables: Dict[str, Union[int, float]]) -> List[str]:
    """
    Substitutes variables in a list of tokens with their corresponding values.

    Args:
        tokens: A list of tokens (strings).
        variables: A dictionary mapping variable names to their numeric values.

    Returns:
        A new list of tokens with variables replaced by their stringified values.
    """
    substituted_tokens = []
    for token in tokens:
        if token in variables:
            substituted_tokens.append(str(variables[token]))
        else:
            substituted_tokens.append(token)
    return substituted_tokens

# --- New functions for safe evaluation ---

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.Constant,  # For numbers and other literals
    ast.Name,      # For variables
    ast.BinOp,     # For binary operations like +, -, *, /
    ast.UnaryOp,   # For unary operations like - (negation)
)

def _eval_node(node: ast.AST, variables: Dict[str, Union[int, float]]) -> Union[int, float]:
    """Recursively evaluate an AST node."""
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed.")
        return node.value
    elif isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Undefined variable: {node.id}")
        return variables[node.id]
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left_val = _eval_node(node.left, variables)
        right_val = _eval_node(node.right, variables)
        return _ALLOWED_OPERATORS[type(node.op)](left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand_val = _eval_node(node.operand, variables)
        return _ALLOWED_OPERATORS[type(node.op)](operand_val)
    elif isinstance(node, ast.Expression): # The root node from ast.parse(mode='eval')
        return _eval_node(node.body, variables)
    else:
        # This case should ideally be caught by the initial isinstance check
        raise ValueError(f"Unexpected/Unhandled AST node type: {type(node).__name__}")


def evaluate_expression_safely(expression: str, variables: Dict[str, Union[int, float]]) -> str:
    """
    Safely evaluates a mathematical expression string using AST.

    Args:
        expression: The mathematical expression string.
        variables: A dictionary mapping variable names to their numeric values.

    Returns:
        The computed result as a string.

    Raises:
        ValueError: If the expression is invalid, contains disallowed operations,
                    or if computation fails (e.g., division by zero, undefined variable).
    """
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")
    try:
        # Parse the expression in 'eval' mode, which expects a single expression
        parsed_ast = ast.parse(expression, mode='eval')
        
        # The body of an ast.Expression node is the actual expression content
        result = _eval_node(parsed_ast.body, variables) # Pass parsed_ast.body directly
        return str(result)
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {e}")
    except ZeroDivisionError:
        raise ValueError("Division by zero.")
    except Exception as e: # Catch other potential errors from _eval_node or ast.parse
        # Re-raise as ValueError to provide a consistent error type to the caller
        raise ValueError(f"Error during expression evaluation: {e}")


if __name__ == '__main__':
    # Example Usage
    expr1 = "3 + 4 * (2 - 1) / x ^ 2"
    tokens1 = parse_expression(expr1)
    print(f"Expression: '{expr1}'")
    print(f"Tokens: {tokens1}")

    expr2 = "  var1*2.5 - (another_var / 3)  "
    tokens2 = parse_expression(expr2)
    print(f"Expression: '{expr2}'")
    print(f"Tokens: {tokens2}")

    expr3 = "sin(x) + cos(y)" # Note: 'sin' and 'cos' will be treated as variables here
    tokens3 = parse_expression(expr3)
    print(f"Expression: '{expr3}'")
    print(f"Tokens: {tokens3}")
    
    expr4 = "a+b-c*d/e^f"
    tokens4 = parse_expression(expr4)
    print(f"Expression: '{expr4}'")
    print(f"Tokens: {tokens4}")

    expr5 = "(1.0 + 2) * 3.14"
    tokens5 = parse_expression(expr5)
    print(f"Expression: '{expr5}'")
    print(f"Tokens: {tokens5}")

    # Example Usage for substitute_variables
    print("\n--- Variable Substitution Examples ---")
    expr_sub = "x * y + z / 2"
    tokens_sub = parse_expression(expr_sub)
    print(f"Original Expression: '{expr_sub}'")
    print(f"Original Tokens: {tokens_sub}")
    
    var_values = {"x": 5, "y": 10, "z": 4}
    substituted_tokens_sub = substitute_variables(tokens_sub, var_values)
    print(f"Variable Values: {var_values}")
    print(f"Substituted Tokens: {substituted_tokens_sub}")

    expr_sub2 = "radius * pi * radius"
    tokens_sub2 = parse_expression(expr_sub2)
    print(f"\nOriginal Expression: '{expr_sub2}'")
    print(f"Original Tokens: {tokens_sub2}")

    var_values2 = {"radius": 2.5, "pi": 3.14159}
    substituted_tokens_sub2 = substitute_variables(tokens_sub2, var_values2)
    print(f"Variable Values: {var_values2}")
    print(f"Substituted Tokens: {substituted_tokens_sub2}")

    # Example with a variable not in the dictionary
    expr_sub3 = "a + b * c"
    tokens_sub3 = parse_expression(expr_sub3)
    print(f"\nOriginal Expression: '{expr_sub3}'")
    print(f"Original Tokens: {tokens_sub3}")
    var_values3 = {"a": 1, "b": 2} # 'c' is missing
    substituted_tokens_sub3 = substitute_variables(tokens_sub3, var_values3)
    print(f"Variable Values: {var_values3}")
    print(f"Substituted Tokens (c not replaced): {substituted_tokens_sub3}")

    # Example Usage for evaluate_expression_safely
    print("\n--- Safe Evaluation Examples ---")
    eval_vars = {"x": 3, "y": 4, "z": 2}
    
    expressions_to_test = [
        ("x + y * z", eval_vars),
        (" (x + y) * z ", eval_vars),
        ("x / (y - z*2)", eval_vars), # Test division and precedence
        ("x ^ z", eval_vars), # Test power
        ("-x + y", eval_vars), # Test unary minus
        ("x / 0", eval_vars), # Test division by zero
        ("x + undefined_var", eval_vars), # Test undefined variable
        ("1.5 * x", eval_vars),
        ("import os", eval_vars), # Test disallowed expression
        ("x + y; print('evil')", eval_vars), # Test disallowed statement
        ("__import__('os').system('echo evil')", eval_vars) # More complex disallowed
    ]

    for expr_str, current_vars in expressions_to_test:
        print(f"\nEvaluating: '{expr_str}' with variables {current_vars}")
        try:
            result = evaluate_expression_safely(expr_str, current_vars)
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")

    print("\nTesting empty expression:")
    try:
        evaluate_expression_safely("", {})
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\nTesting expression with only spaces:")
    try:
        evaluate_expression_safely("   ", {})
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting expression with unsupported operator (e.g., bitwise XOR):")
    try:
        # Note: The tokenizer might split this differently, but ast.parse would fail for `^` as bitwise XOR if not handled.
        # Our current `^` is ast.Pow. If we wanted to test a truly unsupported AST op, it's harder with simple strings.
        # Let's try an expression that would lead to an unsupported AST node if not careful.
        # The current setup is quite restrictive, so most complex Python syntax will fail at ast.parse(mode='eval')
        # or at the _eval_node type check.
        # Example: a function call like "print(x)"
        evaluate_expression_safely("print(x)", eval_vars)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting expression with unsupported constant type (string literal):")
    try:
        evaluate_expression_safely("'hello' + x", eval_vars)
    except ValueError as e:
        print(f"Error: {e}")
