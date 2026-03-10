import ast

def evaluate_expression(expression_string: str):
    """
    Evaluates a simple arithmetic expression string.

    Args:
        expression_string: The arithmetic expression (e.g., "2 + 3 * 4").
                           Supports +, -, *, /, and parentheses.

    Returns:
        The numerical result of the expression or an error message string.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not all(char in allowed_chars for char in expression_string):
        return "Error: Expression contains invalid characters."

    try:
        # A slightly safer way to evaluate arithmetic expressions than raw eval()
        # by first parsing into an Abstract Syntax Tree (AST) and then compiling
        # and evaluating that tree. We'll restrict the environment for eval.
        
        # Ensure the expression is not empty or just whitespace
        if not expression_string.strip():
            return "Error: Expression is empty."

        # Parse the expression to an AST node
        # ast.parse can take mode='eval' for single expressions
        node = ast.parse(expression_string, mode='eval')

        # Optional: Walk the AST to ensure only allowed node types are present
        # For simplicity in this example, we'll skip deep AST validation,
        # but for production, you might want to check for allowed ast node types
        # (e.g., ast.Num, ast.BinOp, ast.UnaryOp, ast.Name (only if you allow constants like pi), etc.)
        # and disallow things like ast.Call, ast.Attribute, etc.

        # Compile the AST node into a code object
        code = compile(node, filename='<string>', mode='eval')
        
        # Evaluate the code object in a restricted environment
        # Provide an empty dictionary for globals and locals to restrict access
        result = eval(code, {"__builtins__": {}}, {})
        
        if isinstance(result, (int, float)):
            return result
        else:
            # This case should ideally not be hit if AST parsing and compilation are correct
            # for arithmetic expressions.
            return "Error: Evaluation did not result in a number."

    except SyntaxError:
        return "Error: Invalid syntax in expression."
    except ZeroDivisionError:
        return "Error: Division by zero."
    except TypeError:
        # This can happen if operations are attempted on incompatible types,
        # though less likely with basic arithmetic and numbers.
        return "Error: Type error in expression."
    except Exception as e:
        # Catch any other unexpected errors during parsing or evaluation
        return f"Error: An unexpected error occurred - {str(e)}"

if __name__ == '__main__':
    # Test cases
    expressions = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "20 / 4",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 / 0",
        "10 / (2 - 2)",
        "2 + ",
        "abc + 5",
        "2 ** 3", # Exponentiation, might be disallowed by char check or AST validation
        "__import__('os').system('echo vulnerable')",
        "some_function()",
        "3.14 * 2",
        "-5 + 10",
        "100 / 2.5",
        "",
        "   ",
        "1+(2*3)-4/5"
    ]

    # Update allowed_chars if exponentiation (**) is desired, or handle via AST.
    # For now, '**' is not in allowed_chars.

    for expr in expressions:
        print(f"Expression: '{expr}' -> Result: {evaluate_expression(expr)}")

    # Example with disallowed characters
    print(f"Expression: '2^3' -> Result: {evaluate_expression('2^3')}")
    # Example with potentially harmful eval (should be caught by restricted eval or char check)
    harmful_expr = '__import__("os").getcwd()'
    print(f"Expression: '{harmful_expr}' -> Result: {evaluate_expression(harmful_expr)}")
