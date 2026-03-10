import ast

def execute_operation(operation_str: str):
    """
    Executes a Python code string, which can include multiple statements
    (e.g., assignments) and expressions. Returns the result of the
    final expression in the string.

    Args:
        operation_str: A string containing Python code.
                       The last non-empty part of the string must be an expression
                       whose value will be returned.
                       Examples:
                         '2 + 3'
                         'a = 5; b = 10; a * b'
                         '''
                         x = 20
                         y = 3
                         x * y + 2
                         '''

    Returns:
        The result of the final expression evaluated in the context of any
        preceding statements in operation_str.

    Raises:
        ValueError: If operation_str is empty, or if its final part is not an expression.
        SyntaxError: If the operation_str has syntax errors.
        NameError: If the operation_str contains undefined variables (that were not assigned).
        TypeError: If operations are attempted on incompatible types.
        Exception: For other evaluation/execution errors.
    """
    # For security reasons, exec() and eval() should only be used with trusted input.
    # This function assumes the input string contains Python expressions/statements.
    
    local_vars = {}
    # Provide a controlled global scope, including builtins for convenience.
    # This makes common functions like abs(), len(), etc., available.
    safe_globals = {"__builtins__": __builtins__}

    try:
        # Parse the code string into an Abstract Syntax Tree (AST)
        stripped_operation_str = operation_str.strip()
        if not stripped_operation_str:
            raise ValueError("Input string is empty or contains only whitespace.")
            
        code_ast = ast.parse(stripped_operation_str)

        if not code_ast.body:
            # This case should ideally be caught by the stripped_operation_str check,
            # but ast.parse might return an empty body for comments-only strings.
            raise ValueError("Input string does not contain executable code.")

        # If there's more than one statement/expression in the AST body,
        # execute all but the last one to set up the context (e.g., variables).
        if len(code_ast.body) > 1:
            # Create an AST module for all statements/expressions except the last one
            exec_ast_module = ast.Module(code_ast.body[:-1], type_ignores=[])
            # Compile this AST module into a code object for execution
            exec_code_obj = compile(exec_ast_module, filename="<ast_exec_setup>", mode="exec")
            # Execute the compiled code object
            exec(exec_code_obj, safe_globals, local_vars)
        
        # The last statement/expression in the AST body is what we'll try to evaluate.
        last_node = code_ast.body[-1]

        # Check if the last node is an expression (ast.Expr).
        # If so, its value can be evaluated and returned.
        if isinstance(last_node, ast.Expr):
            # Compile the expression part (last_node.value) in 'eval' mode.
            # ast.Expression is a wrapper needed for compiling an expression node.
            eval_ast_expression = ast.Expression(last_node.value)
            eval_code_obj = compile(eval_ast_expression, filename="<ast_eval_final>", mode="eval")
            # Evaluate the compiled expression code object.
            result = eval(eval_code_obj, safe_globals, local_vars)
            return result
        else:
            # The last part of the code is a statement (e.g., assignment, function def),
            # not an expression. According to the function's contract, we need an
            # expression to return a value.
            # We could execute this last statement if desired, but it won't yield a returnable value via eval.
            # For now, raise an error as the function is expected to return the result of an expression.
            # To execute it without returning a value:
            # exec_last_ast_module = ast.Module([last_node], type_ignores=[])
            # exec_last_code_obj = compile(exec_last_ast_module, filename="<ast_exec_last_stmt>", mode="exec")
            # exec(exec_last_code_obj, safe_globals, local_vars)
            # return None # Or some other indicator
            raise ValueError("The final part of the operation string must be an expression to return a value.")

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        # Re-raise specific, common errors for clarity
        raise e
    except Exception as e:
        # Catch any other eval/exec-related errors
        raise Exception(f"Error executing or evaluating operation '{operation_str}': {e}")

if __name__ == '__main__':
    # Example usage:
    print(f"Result of '2 + 3': {execute_operation('2 + 3')}")
    print(f"Result of '10 - 4': {execute_operation('10 - 4')}")
    print(f"Result of '6 * 7': {execute_operation('6 * 7')}")
    print(f"Result of '8 / 2': {execute_operation('8 / 2')}")
    print(f"Result of '2 ** 3': {execute_operation('2 ** 3')}")
    print(f"Result of 'abs(-5)': {execute_operation('abs(-5)')}")


    # Example with variables and multiple operations (single line)
    op_str1 = 'a = 5; b = 10; a * b + 2'
    print(f"Result of '{op_str1}': {execute_operation(op_str1)}")

    # Example with variables and multiple operations (multi-line string)
    op_str2 = """
    x = 100
    y = 25
    x / y - 1
    """
    print(f"Result of multi-line operation:\n{op_str2.strip()}\nOutput: {execute_operation(op_str2)}")

    # Example of accessing a variable defined in the operation string
    op_str3 = "my_var = 'hello world'; my_var"
    print(f"Result of '{op_str3}': {execute_operation(op_str3)}")

    # Error handling examples
    try:
        execute_operation("1 / 0")
    except ZeroDivisionError as e:
        print(f"Error (expected for 1/0): {e}")

    try:
        # This is still potentially unsafe if the function is exposed to untrusted input.
        # The ast-based approach doesn't inherently prevent malicious code execution,
        # it just parses and executes Python code.
        execute_operation("import os; os.system('echo unsafe_command_executed')") 
    except Exception as e:
        # Depending on the system and permissions, the above might execute.
        # The function does not sanitize input beyond Python syntax.
        print(f"Error with potentially unsafe input: {e}")


    try:
        execute_operation("a + 3") # Example of NameError (a is not defined)
    except NameError as e:
        print(f"Error (expected for undefined 'a'): {e}")

    try:
        execute_operation("2 +") # Example of SyntaxError
    except SyntaxError as e:
        print(f"Error (expected for syntax error '2 +'): {e}")

    try:
        execute_operation("c = 10") # Last part is an assignment, not an expression
    except ValueError as e:
        print(f"Error (expected for non-expression end): {e}")
    
    try:
        execute_operation("") # Empty string
    except ValueError as e:
        print(f"Error (expected for empty string): {e}")

    try:
        execute_operation("   ") # String with only whitespace
    except ValueError as e:
        print(f"Error (expected for whitespace-only string): {e}")
    
    try:
        execute_operation("# just a comment") # String with only a comment
    except ValueError as e:
        print(f"Error (expected for comment-only string): {e}")
