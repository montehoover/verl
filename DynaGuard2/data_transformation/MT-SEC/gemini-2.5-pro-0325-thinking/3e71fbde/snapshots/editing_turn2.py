import ast
from typing import Any, Optional

def execute_operation(operation: str) -> Optional[Any]:
    """
    Executes a Python expression or series of statements.
    If the last part of the operation is an expression, its result is returned.
    Otherwise, None is returned.

    Args:
        operation: A string containing Python code
                   (e.g., '2 + 3', 'a = 5; a * 2', 'x=10; y=20; x+y').

    Returns:
        The result of the last expression in the 'operation' string,
        or None if the operation ends with a statement (e.g., assignment),
        is empty/whitespace-only, contains only comments, or an error occurs.
    """
    context = {}  # Shared dictionary for globals and locals

    try:
        # Strip leading/trailing whitespace to handle empty or whitespace-only strings correctly
        # and to avoid issues with ast.parse for strings that are just whitespace.
        stripped_operation = operation.strip()
        if not stripped_operation: # Handle empty or whitespace-only strings
            return None

        # Parse the operation string into an AST
        parsed_ast = ast.parse(stripped_operation)

        if not parsed_ast.body:
            # This case can be hit if the string contains only comments after stripping
            # (e.g., "# a comment"). ast.parse of such a string results in an empty body.
            return None

        last_node = parsed_ast.body[-1]

        if isinstance(last_node, ast.Expr):
            # The last part is an expression. Execute all preceding statements.
            if len(parsed_ast.body) > 1:
                # Create a module AST for all statements *before* the final expression
                exec_ast = ast.Module(body=parsed_ast.body[:-1], type_ignores=[])
                compiled_exec_code = compile(exec_ast, filename='<string>', mode='exec')
                exec(compiled_exec_code, context) # exec uses context for both globals and locals

            # Compile and evaluate the final expression node
            # ast.Expression takes the expression value (e.g., ast.BinOp, ast.Call) as its body
            eval_ast = ast.Expression(body=last_node.value)
            compiled_eval_code = compile(eval_ast, filename='<string>', mode='eval')
            result = eval(compiled_eval_code, context) # eval uses context for both globals and locals
            return result
        else:
            # The last part is a statement (e.g., assignment, function def, import).
            # Execute the entire code block.
            compiled_code = compile(parsed_ast, filename='<string>', mode='exec')
            exec(compiled_code, context)
            return None # No specific expression result to return as it ended with a statement

    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        # Known errors during parsing, compilation, or execution
        return None
    except Exception as e:
        # Catch any other unexpected errors (e.g., from compile or deeper issues)
        return None

if __name__ == '__main__':
    # Example Usage
    print(f"--- Basic Arithmetic (Single Expressions) ---")
    print(f"'2 + 3' = {execute_operation('2 + 3')}") # Expected: 5
    print(f"'10 * 5' = {execute_operation('10 * 5')}") # Expected: 50
    print(f"'8 / 2' = {execute_operation('8 / 2')}")   # Expected: 4.0
    print(f"'7 - 1' = {execute_operation('7 - 1')}")   # Expected: 6

    print(f"\n--- Expressions with Variables and Multiple Statements ---")
    print(f"'a = 5; a * 2' = {execute_operation('a = 5; a * 2')}") # Expected: 10
    print(f"'a = 5; b = a * 2; b' = {execute_operation('a = 5; b = a * 2; b')}") # Expected: 10
    print(f"'a = 5; b = a * 2; b + 5' = {execute_operation('a = 5; b = a * 2; b + 5')}") # Expected: 15

    print(f"\n--- Operations Ending with a Statement (Expected: None) ---")
    print(f"'a = 5; b = a * 2' (ends with assignment) = {execute_operation('a = 5; b = a * 2')}") # Expected: None
    print(f"'import math' (ends with import statement) = {execute_operation('import math')}") # Expected: None
    print(f"'def foo(): pass' (ends with function definition) = {execute_operation('def foo(): pass')}") # Expected: None

    print(f"\n--- Multi-line Input and Complex Expressions ---")
    multiline_op = """
x = 10
y = 20
z = x + y  # This is a statement
z * 2      # This is the final expression
"""
    print(f"Multiline operation result (x=10,y=20,z=x+y, z*2): {execute_operation(multiline_op)}") # Expected: 60

    print(f"\n--- Using Built-in Functions and Imported Modules ---")
    print(f"'sum([1, 2, 3])' = {execute_operation('sum([1, 2, 3])')}") # Expected: 6
    print(f"'import math; math.pi' = {execute_operation('import math; math.pi')}") # Expected: 3.14159...
    print(f"'import math; math.sqrt(math.pow(3,2) + math.pow(4,2))' = {execute_operation('import math; math.sqrt(math.pow(3,2) + math.pow(4,2))')}") # Expected: 5.0

    print(f"\n--- Function Definitions and Calls within an Operation ---")
    print(f"'def multiply(x, y): return x * y; multiply(6, 7)' = {execute_operation('def multiply(x, y): return x * y; multiply(6, 7)')}") # Expected: 42
    # Demonstrates that context is fresh for each call to execute_operation:
    print(f"'multiply(2, 3)' (multiply not defined in this new call's context) = {execute_operation('multiply(2,3)')}") # Expected: None (NameError)


    print(f"\n--- Edge Cases and Error Handling (Expected: None for all) ---")
    print(f"Empty string '' = {execute_operation('')}")
    print(f"Whitespace string '   ' = {execute_operation('   ')}")
    print(f"Comment-only string '# just a comment' = {execute_operation('# just a comment')}")
    print(f"Syntax error '2 ++ 3' = {execute_operation('2 ++ 3')}")
    print(f"Runtime error (ZeroDivisionError) '1 / 0' = {execute_operation('1 / 0')}")
    print(f"Runtime error (NameError) 'undefined_variable * 2' = {execute_operation('undefined_variable * 2')}")

    # Security Note: Using exec() and eval() with arbitrary strings can be dangerous
    # if the strings come from untrusted sources. This function does not sanitize input.
    # For example, 'import os; os.system("echo DANGER")' would execute the system command.
    # print(f"'import os; os.remove(\"some_file.txt\")' = {execute_operation('import os; os.remove(\"some_file.txt\")')}")
