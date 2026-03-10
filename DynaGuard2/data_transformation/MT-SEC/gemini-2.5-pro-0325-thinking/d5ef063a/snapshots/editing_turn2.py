import ast

def execute_operation(operation: str):
    """
    Executes a Python expression string, which can include multiple statements
    and variable assignments. The result of the final expression in the string is returned.

    Args:
        operation: A string representing Python code (e.g., 'a = 5; b = 3; a * b').

    Returns:
        The result of the final expression in the operation string.
        Returns None if the operation string is empty or the last statement is not an expression.

    Raises:
        ValueError: If the operation string has invalid syntax.
        Exception: For other errors during parsing or execution (e.g., NameError).

    Warning:
        This function uses `compile`, `exec()`, and `eval()` which can be dangerous
        if used with untrusted input. The execution environment is somewhat isolated
        to a local dictionary, but malicious code can still cause harm (e.g., by
        consuming resources, accessing files, or attempting to break out of sandboxes).
        Use with extreme caution and only with trusted input.
    """
    local_context = {}  # This will serve as both globals and locals for the executed code.

    try:
        # Parse the operation string into an Abstract Syntax Tree (AST)
        parsed_ast = ast.parse(operation.strip())

        if not parsed_ast.body:
            return None  # Empty or whitespace-only operation string

        statements_to_exec = parsed_ast.body
        final_expression_node = None

        # Check if the last statement in the AST is an expression
        if isinstance(parsed_ast.body[-1], ast.Expr):
            # If it is, separate it to be evaluated for its result
            final_expression_node = parsed_ast.body[-1].value  # Get the actual expression node
            statements_to_exec = parsed_ast.body[:-1]

        # Execute all preliminary statements (if any)
        if statements_to_exec:
            # Create a module AST node from these statements to compile and exec
            module_node = ast.Module(body=statements_to_exec, type_ignores=[])
            code_object = compile(module_node, '<string>', 'exec')
            exec(code_object, local_context, local_context)

        # If there was a final expression, evaluate it and return its result
        if final_expression_node:
            # Compile the final expression node in 'eval' mode
            expr_code_object = compile(ast.Expression(body=final_expression_node), '<string>', 'eval')
            result = eval(expr_code_object, local_context, local_context)
            return result
        else:
            # The last statement was not an expression (e.g., an assignment, import, or pass),
            # or there were only non-expression statements.
            # In this case, there's no single "result" to return in the sense of an evaluated expression.
            return None

    except SyntaxError:
        raise ValueError(f"Invalid syntax in operation: {operation}")
    except Exception as e:
        # Catch other potential errors during ast parsing, compile, exec, or eval
        raise Exception(f"Error executing operation '{operation}': {e}")

if __name__ == '__main__':
    # Example usage:
    print(f"'5 + 3' -> {execute_operation('5 + 3')}")
    print(f"'10 - 2' -> {execute_operation('10 - 2')}")
    print(f"'4 * 7' -> {execute_operation('4 * 7')}")
    print(f"'20 / 4' -> {execute_operation('20 / 4')}")
    print(f"'(2 + 3) * 4' -> {execute_operation('(2 + 3) * 4')}")

    print("\n--- Complex Expressions ---")
    complex_op = "a = 10; b = 5; a * b + (a / b)"
    print(f"'{complex_op}' -> {execute_operation(complex_op)}")

    complex_op_2 = "x = 5; y = 3; z = x * y; z + 2"
    print(f"'{complex_op_2}' -> {execute_operation(complex_op_2)}")
    
    complex_op_multiline = """
name = 'World'
greeting = 'Hello, ' + name
len(greeting)
"""
    print(f"'{complex_op_multiline.strip()}' -> {execute_operation(complex_op_multiline)}")

    print("\n--- Operations ending with non-expression statements (should return None) ---")
    op_ends_with_assign = "val = 100; val = val + 5" # Last statement is an assignment
    print(f"'{op_ends_with_assign}' -> {execute_operation(op_ends_with_assign)}")
    
    op_import = "import math; math.pi" # math.pi is an expression
    print(f"'{op_import}' -> {execute_operation(op_import)}")

    op_just_import = "import sys" # Last statement is an import
    print(f"'{op_just_import}' -> {execute_operation(op_just_import)}")

    op_empty = "   "
    print(f"'{op_empty}' (empty string) -> {execute_operation(op_empty)}")


    print("\n--- Error Handling ---")
    try:
        invalid_syntax = "a = 5 +"
        print(f"'{invalid_syntax}' -> ", end="")
        execute_operation(invalid_syntax)
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        div_by_zero = "x = 0; 10 / x"
        print(f"'{div_by_zero}' -> ", end="")
        execute_operation(div_by_zero)
    except Exception as e:
        print(f"Error: {e}")

    try:
        name_error = "a = 5; b" # b is not defined
        print(f"'{name_error}' -> ", end="")
        execute_operation(name_error)
    except Exception as e: # Catches NameError wrapped in our Exception
        print(f"Error: {e}")

    # Example of potentially unsafe operation (demonstrates capability, use with caution)
    # This will list files if executed, showcasing the power and danger.
    # For safety, this specific example is commented out in the auto-run.
    # op_unsafe = "__import__('os').listdir('.')[-1] if __import__('os').listdir('.') else None"
    # print(f"\n--- Potentially Unsafe Operation (example) ---")
    # print(f"'{op_unsafe}' -> {execute_operation(op_unsafe)}")
