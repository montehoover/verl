import ast

def evaluate_expression(expr_string: str) -> str:
    """
    Evaluates a user-provided mathematical expression string.

    Args:
        expr_string: A string containing a basic mathematical expression.

    Returns:
        A string representing the result of the evaluation or "Computation Error!"
        if the input is suspicious, unsafe, or evaluation fails.
    """
    try:
        # Basic input validation: length limit to prevent potential DoS from overly long strings
        if len(expr_string) > 200:  # Arbitrary limit for expression length
            return "Computation Error!"

        # Parse the expression string into an Abstract Syntax Tree (AST)
        # 'eval' mode is used as we expect a single expression.
        parsed_ast = ast.parse(expr_string, mode='eval')

        # Validate the AST to ensure it only contains allowed operations/nodes.
        # We iterate through all nodes in the AST using ast.walk.
        valid_ast = True
        for node in ast.walk(parsed_ast):
            # Allow structural/container nodes; their children will be checked.
            if isinstance(node, (ast.Expression, ast.Interactive, ast.Module)):
                continue
            
            # Allow numeric constants (int, float, complex).
            # ast.Constant is for Python 3.8+
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, complex)):
                    valid_ast = False; break
            # ast.Num is for Python < 3.8
            elif isinstance(node, ast.Num): 
                if not isinstance(node.n, (int, float, complex)):
                    valid_ast = False; break
            # Allow binary operations with a specific set of operators.
            elif isinstance(node, ast.BinOp):
                allowed_bin_ops = (
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, 
                    ast.FloorDiv, ast.Mod
                )
                if not isinstance(node.op, allowed_bin_ops):
                    valid_ast = False; break
            # Allow unary operations with a specific set of operators.
            elif isinstance(node, ast.UnaryOp):
                allowed_unary_ops = (ast.UAdd, ast.USub) # Unary plus, unary minus
                if not isinstance(node.op, allowed_unary_ops):
                    valid_ast = False; break
            # Any other AST node type (e.g., ast.Name, ast.Call, ast.Attribute,
            # ast.List, ast.Dict, ast.Compare, string constants, etc.) is disallowed.
            else:
                valid_ast = False; break
        
        if not valid_ast:
            return "Computation Error!"

        # If the AST is deemed safe, compile it.
        # The filename '<user_expr>' is used in tracebacks if an error occurs during evaluation.
        compiled_code = compile(parsed_ast, filename='<user_expr>', mode='eval')

        # Evaluate the compiled code in a highly restricted environment.
        # globals={'__builtins__': {}} prevents access to Python's built-in functions.
        # locals={} ensures no predefined local variables are available.
        # Basic arithmetic operations (+, -, etc.) are handled by Python's opcodes
        # and do not require lookups in globals or locals.
        result = eval(compiled_code, {'__builtins__': {}}, {})

        # Ensure the result of the evaluation is a number before converting to string.
        # This is a safeguard, as AST validation should prevent non-numeric results.
        if not isinstance(result, (int, float, complex)):
            return "Computation Error!"
            
        return str(result)

    # Catch specific, common errors during evaluation.
    except ZeroDivisionError:
        return "Computation Error!"
    except OverflowError:  # E.g., result too large for float.
        return "Computation Error!"
    
    # Catch errors related to parsing or the structure of the expression.
    except SyntaxError:  # Malformed expression string.
        return "Computation Error!"
    except TypeError:    # E.g., operation on incompatible types not caught by AST check.
        return "Computation Error!"
    except ValueError:   # E.g., malformed numeric literals if not caught by parser.
        return "Computation Error!"
    
    # A generic catch-all for any other exceptions.
    # This includes unexpected issues or potential security-related exceptions
    # if the AST validation was somehow incomplete or bypassed.
    except Exception:
        return "Computation Error!"
