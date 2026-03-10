def exec_calculation(calc_string: str) -> str:
    """
    Safely evaluates a Python mathematical expression string and returns the result as a string.
    Returns "Computation Error!" for empty input, unsafe code, or any computation errors.
    """
    if not calc_string:
        return "Computation Error!"

    try:
        # Use a restricted environment for eval to enhance safety.
        # Globals are restricted to prevent access to most built-in functions.
        # Locals are empty, meaning only literals and operators in the string are processed.
        # Allowed operations include basic arithmetic (+, -, *, /, //, %, **),
        # numbers (int, float), and potentially simple data structures if typed out
        # (e.g. "[1,2]" would result in "[1, 2]").
        result = eval(calc_string, {"__builtins__": {}}, {})
        return str(result)
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        # These are common errors expected during evaluation of arbitrary strings.
        return "Computation Error!"
    except Exception:
        # Catch any other unexpected errors during evaluation.
        return "Computation Error!"

if __name__ == '__main__':
    test_cases = {
        "1+2*3": "7",
        "10 / 5 - 2": "0.0",
        "2 * (3 + 4)": "14",
        "2**3": "8",
        "-5 + 2": "-3",
        "1.5 + 3.0": "4.5",
        "10 // 3": "3",
        "10 % 3": "1",
        "1 / 0": "Computation Error!",
        "1 +": "Computation Error!",  # SyntaxError
        "alpha + 1": "Computation Error!",  # NameError
        "": "Computation Error!",  # Empty input
        "__import__('os').system('clear')": "Computation Error!",  # NameError for __import__
        "print('hello')": "Computation Error!",  # NameError for print
        "[1,2,3]": "[1, 2, 3]",  # eval returns list, str() converts
        "'hello' + 'world'": "helloworld",  # eval returns string
        "1 < 2": "True",  # eval returns boolean
        "1.5.5 + 3": "Computation Error!",  # SyntaxError
        "5 %%% 2": "Computation Error!",  # SyntaxError
        "eval('1+1')": "Computation Error!", # NameError for 'eval'
        "object()": "Computation Error!", # NameError for 'object'
        "float('inf')": "Computation Error!", # NameError for 'float'
        "abs(-5)": "Computation Error!", # NameError for 'abs'
    }

    print("Running test cases for exec_calculation:")
    all_passed = True
    for expr, expected in test_cases.items():
        actual = exec_calculation(expr)
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"Input: '{expr}', Output: '{actual}', Expected: '{expected}' --- {status}")

    # Specific test for a slightly more complex but safe expression
    print(f"\nTesting complex safe expression: '(2 + 3) * 4 / 2 - 1'")
    expr_complex = "(2 + 3) * 4 / 2 - 1"
    actual_complex = exec_calculation(expr_complex)
    expected_complex = "9.0" # (5 * 4 / 2) - 1 = (20 / 2) - 1 = 10.0 - 1 = 9.0
    status_complex = "PASS" if actual_complex == expected_complex else "FAIL"
    if actual_complex != expected_complex:
        all_passed = False
    print(f"Input: '{expr_complex}', Output: '{actual_complex}', Expected: '{expected_complex}' --- {status_complex}")

    if all_passed:
        print("\nAll test cases passed!")
    else:
        print("\nSome test cases FAILED.")
