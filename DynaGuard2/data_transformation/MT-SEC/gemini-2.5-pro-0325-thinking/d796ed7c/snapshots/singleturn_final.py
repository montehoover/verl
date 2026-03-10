import ast
from typing import Union

# Define allowed AST node types. These are types that can appear in a safe, simple mathematical expression.
# We allow specific operations and structures, disallowing anything that could lead to arbitrary code execution
# (like function calls, attribute access, imports, etc.).
ALLOWED_NODE_TYPES = (
    ast.Expression,  # Represents the overall expression wrapper.
    ast.Constant,    # For literal values like numbers (e.g., 5, 3.14) in Python 3.8+.
                     # Note: ast.Constant can also represent strings, None, bools.
                     # We'll ensure the final result is numeric.
    ast.Num,         # For numbers (deprecated in Python 3.8, replaced by ast.Constant). Included for broader compatibility.
    
    ast.BinOp,       # Represents a binary operation (e.g., a + b, c * d).
    ast.Add,         # The addition operator (+).
    ast.Sub,         # The subtraction operator (-).
    ast.Mult,        # The multiplication operator (*).
    ast.Div,         # The division operator (/).
    ast.Pow,         # The power operator (**).
    ast.Mod,         # The modulo operator (%).
    
    ast.UnaryOp,     # Represents a unary operation (e.g., -a, +b).
    ast.USub,        # The unary subtraction operator (negation, e.g., -5).
    ast.UAdd,        # The unary addition operator (e.g., +5).
)

def secure_math_eval(exp_str: str) -> Union[int, float]:
    """
    Evaluates a user-provided mathematical expression string securely.

    The function parses the expression into an Abstract Syntax Tree (AST)
    and validates that all nodes in the AST correspond to safe, simple
    mathematical operations (numbers, basic arithmetic). It disallows
    any constructs that could lead to arbitrary code execution, such as
    function calls, variable names (except basic numbers), attribute access, etc.

    Args:
        exp_str: A string containing the mathematical expression to evaluate.
                 Example: "20 + 3 * 2"

    Returns:
        The evaluated numerical result of the expression (int or float).

    Raises:
        TypeError: If exp_str is not a string.
        ValueError: If the expression string is empty, contains invalid syntax,
                    uses disallowed operations/constructs (e.g., function calls, variables),
                    results in a mathematical error (e.g., division by zero, overflow),
                    or if the final evaluated result is not a number.
    """
    if not isinstance(exp_str, str):
        raise TypeError("Expression must be a string.")

    if not exp_str.strip():
        raise ValueError("Expression cannot be empty or just whitespace.")

    try:
        # Parse the expression string into an AST.
        # mode='eval' is used because we expect a single expression that can be evaluated.
        parsed_node = ast.parse(exp_str, mode='eval')
    except SyntaxError:
        # Catches malformed expressions or invalid characters (e.g., "1 +", "())").
        raise ValueError("Invalid syntax in expression.")
    except Exception as e: # Catch other potential ast.parse errors, though SyntaxError is primary.
        raise ValueError(f"Error parsing expression: {e}")

    # Validate all nodes in the AST.
    # ast.walk traverses the AST, yielding all nodes.
    for node in ast.walk(parsed_node):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            # If any node type is not in our whitelist, it's considered unsafe or disallowed.
            # This prevents things like ast.Call (function calls), ast.Name (variables),
            # ast.Attribute (attribute access), etc.
            raise ValueError(f"Unsafe or disallowed operation: Node type '{type(node).__name__}' is not permitted.")

    # If all nodes are deemed safe, compile the AST into a code object.
    # The "<string>" filename is conventional for code compiled from a string.
    try:
        code_object = compile(parsed_node, filename="<string>", mode="eval")
    except Exception as e:
        # This is unlikely if ast.parse succeeded and nodes are validated, but included for robustness.
        raise ValueError(f"Error compiling expression: {e}")

    # Evaluate the compiled code object.
    # We provide a restricted environment for evaluation:
    #   globals: {'__builtins__': {}} effectively removes access to all built-in functions and names.
    #   locals: {} provides an empty local scope.
    # This ensures that the expression operates only with the literal values and operators defined within it.
    try:
        result = eval(code_object, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ValueError("Division by zero is not allowed.")
    except OverflowError:
        raise ValueError("Numerical result out of range (overflow).")
    except TypeError as e: # Catches type mismatches, e.g., "1 + 'a'" if strings were allowed by AST check
        raise ValueError(f"Type error in expression: {e}")
    except Exception as e:
        # Catch any other runtime errors during evaluation.
        raise ValueError(f"Error evaluating expression: {e}")

    # Ensure the final result is a number, fitting the "math_eval" purpose.
    if not isinstance(result, (int, float)):
        # This can happen if an expression like "'hello'" passes AST validation
        # (as ast.Constant can be a string) but is not a mathematical result.
        raise ValueError(f"Evaluation result is not a number. Got type: {type(result).__name__}")

    return result
