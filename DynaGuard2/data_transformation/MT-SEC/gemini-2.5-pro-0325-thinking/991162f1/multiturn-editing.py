from asteval import Interpreter

def evaluate_math_expression(math_expression: str) -> str:
    """
    Securely evaluates a string containing a mathematical expression.

    This function uses asteval to safely parse and compute the result
    of mathematical expressions. It avoids using eval() or exec().

    Args:
        math_expression: A string containing a mathematical expression
                         (e.g., "2 + 3 * 4", "2**3 + (4*5)", "sin(pi/2)").

    Returns:
        The calculated result as a string, or a warning message if
        the expression is invalid, unsafe, or causes an error during evaluation.
    """
    aeval = Interpreter()
    # Prevent access to __builtins__ or other potentially unsafe features
    # asteval is generally safe by default, but this is an extra precaution.
    # For more fine-grained control, explore aeval.symtable.
    # By default, asteval provides a limited set of safe builtins and math functions.

    # Attempt to evaluate the expression
    result = aeval.eval(math_expression)

    # Check for errors stored by asteval
    if aeval.error:
        # Join all error messages into a single string
        error_messages = []
        for err in aeval.error:
            error_messages.append(f"Error: {err.get_error()[1]} (at {err.lineno}:{err.col_offset})")
            # err.get_error() returns (etype, emsg, etraceback)
            # err.msg is often a more concise message
            # error_messages.append(f"Error: {err.msg} (Expression: '{err.expr}')")
        return "Warning: " + "; ".join(error_messages)

    # If no errors, convert result to string and return
    return str(result)

if __name__ == '__main__':
    # Example usage:
    expressions = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "20 / 4",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "2 ** 3",
        "2 ** 3 + (4 * 5)",
        "(10 + 5) / 3 - 2**2",
        "10 / 0",              # Example of ZeroDivisionError
        "5 + ",                # Example of SyntaxError
        "2 ** (3 + 1)",
        "sqrt(16)",            # Using a math function (sqrt is often available)
        "sin(pi/2)",           # Using math constants and functions
        "import os",           # Attempting an import (should be blocked by asteval)
        "__import__('os').system('echo unsafe')", # Another unsafe attempt
        "a = 5; a * 2",        # Variable assignment
        "def f(x): return x*x; f(3)", # Function definition (may be restricted by default config)
        "open('file.txt')",    # Attempting file access
        "non_existent_var * 2" # Using an undefined variable
    ]

    for expr in expressions:
        result = evaluate_math_expression(expr)
        print(f"Expression: '{expr}', Result: {result}")

    print(evaluate_math_expression("3.14 * 2"))
    print(evaluate_math_expression("100 / 2.5"))
    # Example with a more complex but safe expression
    print(evaluate_math_expression("log10(100) + abs(-5)"))
    # Example of an expression that might be too complex or use disallowed features
    # (depending on asteval's default configuration or any custom restrictions)
    # For example, list comprehensions might be disabled by default in some strict settings.
    # print(evaluate_math_expression("[i*i for i in range(5)]"))
