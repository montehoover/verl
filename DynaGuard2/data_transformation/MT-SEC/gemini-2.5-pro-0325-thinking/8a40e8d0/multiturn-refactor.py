import ast
import logging

# It's common to configure logging once at the application level.
# For simplicity here, we'll assume basicConfig might be called elsewhere,
# or default logging to stderr is acceptable.
# If you want to ensure output for this specific logger, you might add:
# logging.basicConfig(level=logging.INFO) 
# somewhere before this function is first called.

def evaluate_expression(expr_string: str) -> str:
    """
    Evaluates a user-provided mathematical expression string using Abstract Syntax Trees (AST).

    This function aims to safely evaluate simple arithmetic expressions by:
    1. Parsing the input string into an AST.
    2. Validating the AST to ensure it only contains allowed nodes (numeric constants,
       basic arithmetic operations). This prevents the execution of arbitrary code,
       function calls, or access to variables.
    3. Compiling the validated AST.
    4. Evaluating the compiled code in a restricted environment with no access to
       built-ins or global/local variables.

    Args:
        expr_string: A string containing a basic mathematical expression.
                     Example: "2 + 3 * (4 - 1)"

    Returns:
        A string representing the numerical result of the evaluation (e.g., "11")
        or the string "Computation Error!" if the input is suspicious, unsafe,
        malformed, or if any part of the evaluation process fails (e.g.,
        division by zero, syntax error, disallowed operations).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to evaluate expression: '{expr_string}'")

    try:
        # Basic input validation:
        # Limit expression length to prevent potential denial-of-service (DoS)
        # attacks from overly long or complex strings that might consume
        # excessive resources during parsing or evaluation.
        if len(expr_string) > 200:  # Arbitrary but reasonable limit
            error_message = "Expression too long"
            logger.warning(f"Computation error for '{expr_string}': {error_message}")
            return "Computation Error!"

        # Step 1: Parse the expression string into an Abstract Syntax Tree (AST).
        # The 'eval' mode is used because we expect a single expression that
        # should evaluate to a value.
        parsed_ast = ast.parse(expr_string, mode='eval')

        # Step 2: Validate the AST to ensure it only contains allowed operations/nodes.
        # We iterate through all nodes in the AST using ast.walk to inspect each part.
        valid_ast = True
        for node in ast.walk(parsed_ast):
            # Allow structural/container nodes like ast.Expression.
            # Their children (the actual expression parts) will be checked individually.
            if isinstance(node, (ast.Expression, ast.Interactive, ast.Module)):
                continue
            
            # Allow numeric constants (integers, floats, complex numbers).
            # ast.Constant is used for Python 3.8+
            # ast.Num is used for Python versions older than 3.8.
            elif isinstance(node, ast.Constant): # For Python 3.8+
                if not isinstance(node.value, (int, float, complex)):
                    valid_ast = False; break # Disallow non-numeric constants (e.g. string constants)
            elif isinstance(node, ast.Num): # For Python < 3.8
                if not isinstance(node.n, (int, float, complex)):
                    valid_ast = False; break # Disallow non-numeric constants
            
            # Allow binary operations (e.g., +, -, *, /, **, //, %).
            # We explicitly list allowed operator types.
            elif isinstance(node, ast.BinOp):
                allowed_bin_ops = (
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                    ast.FloorDiv, ast.Mod
                )
                if not isinstance(node.op, allowed_bin_ops):
                    valid_ast = False; break # Disallow other binary operations (e.g. bitwise ops)
            
            # Allow unary operations (e.g., unary +, unary -).
            # We explicitly list allowed operator types.
            elif isinstance(node, ast.UnaryOp):
                allowed_unary_ops = (ast.UAdd, ast.USub) # Unary plus, unary minus
                if not isinstance(node.op, allowed_unary_ops):
                    valid_ast = False; break # Disallow other unary operations (e.g. bitwise not)
            
            # Any other AST node type is disallowed.
            # This includes but is not limited to:
            # - ast.Name (variables)
            # - ast.Call (function calls)
            # - ast.Attribute (attribute access, e.g., object.method)
            # - ast.List, ast.Dict, ast.Tuple (data structures)
            # - ast.Compare (comparison operators like <, >, ==)
            # - ast.Subscript (indexing or slicing)
            # - ast.Lambda, ast.FunctionDef (function definitions)
            # - String constants (already covered by ast.Constant check for non-numerics)
            else:
                valid_ast = False; break
        
        if not valid_ast:
            # If any disallowed node was found, return an error.
            error_message = "Invalid AST node detected"
            logger.warning(f"Computation error for '{expr_string}': {error_message}")
            return "Computation Error!"

        # Step 3: If the AST is deemed safe, compile it into a code object.
        # The filename '<user_expr>' is used in tracebacks if an error occurs
        # during the subsequent evaluation phase.
        compiled_code = compile(parsed_ast, filename='<user_expr>', mode='eval')

        # Step 4: Evaluate the compiled code in a highly restricted environment.
        # - globals={'__builtins__': {}}: This is crucial. It prevents access to
        #   Python's built-in functions (like open(), exec(), etc.) by providing
        #   an empty dictionary for builtins.
        # - locals={}: Ensures no predefined local variables are available.
        # Basic arithmetic operations (+, -, *, / etc.) are handled by Python's
        # opcodes directly and do not require lookups in globals or locals once compiled.
        result = eval(compiled_code, {'__builtins__': {}}, {})

        # Final validation: Ensure the result of the evaluation is a number.
        # This is an additional safeguard, as the AST validation should ideally
        # prevent expressions that result in non-numeric types.
        if not isinstance(result, (int, float, complex)):
            error_message = "Evaluation resulted in a non-numeric type"
            logger.error(f"Computation error for '{expr_string}': {error_message}. Result type: {type(result)}")
            return "Computation Error!"
        
        result_str = str(result)
        logger.info(f"Expression '{expr_string}' successfully evaluated to: {result_str}")
        return result_str

    # --- Exception Handling ---
    # Catch specific, common arithmetic errors that can occur during evaluation.
    except ZeroDivisionError as e:
        # Handles division by zero (e.g., "1 / 0").
        logger.error(f"Computation error for '{expr_string}': ZeroDivisionError - {e}")
        return "Computation Error!"
    except OverflowError as e:
        # Handles results that are too large to be represented (e.g., "1e1000").
        logger.error(f"Computation error for '{expr_string}': OverflowError - {e}")
        return "Computation Error!"
    
    # Catch errors related to parsing the expression string or its structure.
    except SyntaxError as e:
        # Handles malformed expression strings (e.g., "1 +", "((2))").
        logger.error(f"Computation error for '{expr_string}': SyntaxError - {e}")
        return "Computation Error!"
    except TypeError as e:
        # Handles operations on incompatible types if not caught by AST validation
        # (though AST validation should make this rare for this specific setup).
        logger.error(f"Computation error for '{expr_string}': TypeError - {e}")
        return "Computation Error!"
    except ValueError as e:
        # Handles malformed numeric literals if not caught by the parser
        # (e.g., an invalid number format that ast.parse might still process
        # but leads to issues later, though highly unlikely with standard numbers).
        logger.error(f"Computation error for '{expr_string}': ValueError - {e}")
        return "Computation Error!"
    
    # A generic catch-all for any other unexpected exceptions.
    # This is important for robustness and to prevent leaking internal error details.
    # It includes unexpected issues or potential security-related exceptions if the
    # AST validation was somehow incomplete or bypassed (highly unlikely with current checks).
    except Exception as e:
        logger.critical(f"Unexpected computation error for '{expr_string}': {type(e).__name__} - {e}", exc_info=True)
        return "Computation Error!"
