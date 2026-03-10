import ast

def setup_execution_environment():
    """
    Initializes a restricted Python script execution environment.

    Returns:
        dict: A dictionary to be used as globals for script execution,
              containing a restricted set of built-ins and no import
              capabilities.
    """
    # Whitelisted built-in functions and constants
    # Based on https://docs.python.org/3/library/builtins.html
    safe_builtins_dict = {
        'abs': abs,
        'all': all,
        'any': any,
        'ascii': ascii,
        'bin': bin,
        'bool': bool,
        'bytearray': bytearray,
        'bytes': bytes,
        'callable': callable,
        'chr': chr,
        'complex': complex,
        'dict': dict,
        'dir': dir, # Can be revealing, but useful for introspection
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hash': hash,
        'hex': hex,
        'id': id,
        'int': int,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'object': object,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print, # Note: print will go to standard output unless redirected
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }

    # Whitelisted safe exception types
    safe_exceptions = [
        BaseException, Exception, StopIteration, StopAsyncIteration,
        ArithmeticError, AssertionError, AttributeError, BufferError, EOFError,
        FloatingPointError, GeneratorExit, IndexError, KeyError, LookupError,
        MemoryError, NameError, NotImplementedError, OverflowError,
        ReferenceError, RuntimeError, SyntaxError, SystemError, TypeError,
        UnboundLocalError, UnicodeDecodeError, UnicodeEncodeError, UnicodeError,
        ValueError, ZeroDivisionError
    ]

    for exc in safe_exceptions:
        safe_builtins_dict[exc.__name__] = exc

    # The environment for the script execution
    script_globals = {
        "__builtins__": safe_builtins_dict
        # No other globals are defined by default.
        # Import is blocked because __import__ is not in safe_builtins_dict.
        # eval and exec are blocked because they are not in safe_builtins_dict.
    }

    return script_globals

if __name__ == '__main__':
    # Example usage (outside the platform, for testing this function)
    restricted_env_globals = setup_execution_environment()

    # Test case 1: Safe operations
    safe_script = """
result = []
for i in range(5):
    result.append(i * 2)
print(result)
s = "hello"
print(len(s))
print(max(1, 5, 2))
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught expected error")
"""
    print("--- Running safe_script ---")
    try:
        exec(safe_script, restricted_env_globals, {}) # Using a fresh local dict
    except Exception as e:
        print(f"Error in safe_script: {type(e).__name__}: {e}")

    # Test case 2: Attempting to import
    import_attempt_script = """
try:
    import os
    print("os imported") # Should not happen
except Exception as e:
    print(f"Import attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running import_attempt_script ---")
    try:
        # Create a new local scope for each execution if desired, or reuse
        exec(import_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in import_attempt_script: {type(e).__name__}: {e}")

    # Test case 3: Attempting to use eval
    eval_attempt_script = """
try:
    eval("1+1")
    print("eval worked") # Should not happen
except Exception as e:
    print(f"Eval attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running eval_attempt_script ---")
    try:
        exec(eval_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in eval_attempt_script: {type(e).__name__}: {e}")

    # Test case 4: Attempting to use open
    open_attempt_script = """
try:
    f = open("test.txt", "w")
    print("open worked") # Should not happen
except Exception as e:
    print(f"Open attempt failed as expected: {type(e).__name__}: {e}")
"""
    print("\n--- Running open_attempt_script ---")
    try:
        exec(open_attempt_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in open_attempt_script: {type(e).__name__}: {e}")

    # Test case 5: Accessing __builtins__ directly (should be the restricted dict)
    builtins_access_script = """
print(type(__builtins__))
print('print' in __builtins__)
print('eval' in __builtins__)
print('__import__' in __builtins__)
"""
    print("\n--- Running builtins_access_script ---")
    try:
        exec(builtins_access_script, restricted_env_globals, {})
    except Exception as e:
        print(f"Error in builtins_access_script: {type(e).__name__}: {e}")
