import io
import contextlib

def execute_code_snippet(code_snippet_string):
    """
    Executes a Python code snippet string and returns its output.

    Args:
        code_snippet_string: A string representing Python code
                             (e.g., 'for i in range(3): print(i)').

    Returns:
        The output of the executed code snippet as a string.
        Returns None if an error occurs.
    """
    # Create a string buffer to capture stdout
    stdout_capture = io.StringIO()
    try:
        # Redirect stdout to our string buffer
        with contextlib.redirect_stdout(stdout_capture):
            # Execute the code snippet
            # Using a dedicated globals dictionary can be safer
            # but for simplicity, we'll use the current globals.
            # For more complex scenarios, consider providing a restricted environment.
            exec(code_snippet_string, {})
        return stdout_capture.getvalue()
    except Exception as e:
        print(f"Error executing code snippet: {code_snippet_string}")
        print(f"Error: {e}")
        # Optionally, return the error message or part of it
        return f"Error: {e}\n" + stdout_capture.getvalue() # Include any partial output
    finally:
        stdout_capture.close()

if __name__ == '__main__':
    # Example usage:
    snippet1 = "print('Hello from snippet!')"
    output1 = execute_code_snippet(snippet1)
    print(f"Output of '{snippet1}':\n{output1}")

    snippet2 = "for i in range(3): print(f'Number: {i}')"
    output2 = execute_code_snippet(snippet2)
    print(f"Output of '{snippet2}':\n{output2}")

    snippet3 = "x = 10\ny = 20\nif x < y: print(f'{x} is less than {y}')\nelse: print(f'{x} is not less than {y}')"
    output3 = execute_code_snippet(snippet3)
    print(f"Output of '{snippet3}':\n{output3}")

    # Example of an expression (exec can handle this too, but output might differ from eval)
    snippet4 = "result = 2 + 3\nprint(result)"
    output4 = execute_code_snippet(snippet4)
    print(f"Output of '{snippet4}':\n{output4}")

    # Example of an invalid snippet (syntax error)
    snippet5 = "for i in range(3)\n print(i)" # Missing colon
    output5 = execute_code_snippet(snippet5)
    print(f"Output of '{snippet5}':\n{output5}")

    # Example of a runtime error
    snippet6 = "print(10 / 0)" # Division by zero
    output6 = execute_code_snippet(snippet6)
    print(f"Output of '{snippet6}':\n{output6}")
