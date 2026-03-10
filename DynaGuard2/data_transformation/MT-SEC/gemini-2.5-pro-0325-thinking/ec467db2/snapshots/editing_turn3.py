import ast
import operator as op

# Supported operators for BinOp and AugAssign
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

def _evaluate_node_in_scope(node, variables):
    """
    Recursively evaluates an AST expression node within a given variable scope.
    Raises specific errors for unsupported node types, operations, or name errors.
    """
    if isinstance(node, ast.Constant):
        # Python 3.8+: ast.Constant for numbers, strings, None, True, False.
        # We only allow numeric (int, float) and string (str) constants.
        if not isinstance(node.value, (int, float, str)):
            raise TypeError(f"Unsupported constant type: {type(node.value).__name__}. Only numbers or strings are allowed.")
        return node.value
    elif isinstance(node, ast.Num): # For Python < 3.8 compatibility (ast.Num holds numeric literals)
        if not isinstance(node.n, (int, float)): # Should only be numbers
            raise TypeError("Unsupported number type in ast.Num.")
        return node.n
    elif isinstance(node, ast.Name): # For variable lookup
        if isinstance(node.ctx, ast.Load):
            if node.id in variables:
                return variables[node.id]
            else:
                raise NameError(f"name '{node.id}' is not defined")
        else:
            # ast.Store or ast.Del contexts are handled by statement evaluators (Assign, AugAssign)
            raise ValueError(f"Unsupported context for ast.Name: {type(node.ctx).__name__}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        
        left_val = _evaluate_node_in_scope(node.left, variables)
        right_val = _evaluate_node_in_scope(node.right, variables)

        if op_type == ast.Add:
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return op.add(left_val, right_val)
            elif isinstance(left_val, str) and isinstance(right_val, str):
                return left_val + right_val # String concatenation
            else:
                raise TypeError(f"Unsupported operand types for +: {type(left_val).__name__} and {type(right_val).__name__}")
        else: # Sub, Mult, Div require numeric operands
            operator_func = _ALLOWED_OPERATORS.get(op_type)
            if operator_func is None:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            if not (isinstance(left_val, (int, float)) and isinstance(right_val, (int, float))):
                op_symbol = {ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(op_type, op_type.__name__)
                raise TypeError(f"Numeric operands required for '{op_symbol}' operator. Got {type(left_val).__name__} and {type(right_val).__name__}.")

            if op_type == ast.Div and right_val == 0:
                raise ZeroDivisionError("Division by zero.")
            
            return operator_func(left_val, right_val)
    elif isinstance(node, ast.UnaryOp):
        # Only Unary Subtraction (negation) is supported.
        if isinstance(node.op, ast.USub):
            operand_val = _evaluate_node_in_scope(node.operand, variables)
            if not isinstance(operand_val, (int, float)):
                raise TypeError(f"Operand for unary minus must be a number, not {type(operand_val).__name__}.")
            return -operand_val
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    else:
        # This disallows ast.Call, ast.Lambda, comprehensions, etc., within expressions.
        raise ValueError(f"Unsupported AST expression node type: {type(node).__name__}")

def execute_user_code(user_code: str):
    """
    Securely parses and executes a user-provided Python script string.
    The script can include basic arithmetic, string operations, assignments,
    and augmented assignments for supported types.
    The result of the script is the value of the last expression statement evaluated.

    Allowed constructs:
    - Literals: Numbers (int, float), Strings (str).
    - Variables: Defined and used within the script.
    - Operators:
        - Arithmetic: +, -, *, / (for numbers).
        - String: + (concatenation).
    - Statements:
        - Expression statements (e.g., `1 + 1`, `my_var * 2`).
        - Assignment: `variable = expression` (e.g., `x = 5`, `s = 'a' + 'b'`).
        - Augmented Assignment: `variable op= expression` (e.g., `x += 1`, `s += 'c'`).
          Supported for +, -, *, / with compatible types.
    - Parentheses for grouping expressions.
    - Unary minus for numbers (e.g., `-5`).

    Args:
        user_code: A string containing the Python script.

    Returns:
        The result of the last evaluated expression statement in the script.
        Returns `None` if the script is empty or ends with a statement that
        does not produce a value (e.g., an assignment).
        Returns the string 'Execution Blocked!' if the script contains invalid syntax,
        unsafe operations, or any unsupported Python constructs.
    """
    try:
        if not isinstance(user_code, str):
            # Handle non-string input explicitly, though ast.parse would also fail.
            return 'Execution Blocked!'

        # Basic pre-checks for obviously malicious patterns or overly long scripts.
        # The primary safety mechanism is the AST traversal and selective evaluation.
        if "__" in user_code:  # Disallow dunder attributes/methods.
            return 'Execution Blocked!'
        
        # Limit script length to prevent potential DoS via extremely long inputs.
        if len(user_code) > 2000: # Max script length (adjustable).
            return 'Execution Blocked!'

        # Parse the script string into an Abstract Syntax Tree (AST).
        # mode='exec' is used for scripts (sequences of statements).
        parsed_ast = ast.parse(user_code, mode='exec')
        
        if not isinstance(parsed_ast, ast.Module):
            # This case should not be reached if ast.parse with mode='exec' succeeds.
            return 'Execution Blocked!'

        local_vars = {}  # Dictionary to store variables local to the script execution.
        script_result = None  # Stores the result of the last expression statement.

        for stmt_node in parsed_ast.body:
            if isinstance(stmt_node, ast.Expr):
                # This is an expression statement (e.g., `1 + 1`, `my_var / 2`).
                # Its value becomes the current potential result of the script.
                script_result = _evaluate_node_in_scope(stmt_node.value, local_vars)
            elif isinstance(stmt_node, ast.Assign):
                # This is an assignment statement (e.g., `x = 100`, `s = 'foo'`).
                # We only support single target assignments (e.g., `a = value`, not `a, b = ...` or `a[0] = ...`).
                if len(stmt_node.targets) != 1 or not isinstance(stmt_node.targets[0], ast.Name):
                    raise ValueError("Unsupported assignment: only single variable targets (e.g., x = ...) are allowed.")
                
                var_name_node = stmt_node.targets[0] # This is an ast.Name node.
                var_name = var_name_node.id
                
                # Evaluate the right-hand side of the assignment.
                value = _evaluate_node_in_scope(stmt_node.value, local_vars)
                local_vars[var_name] = value
                # Assignments themselves don't set `script_result`.
            elif isinstance(stmt_node, ast.AugAssign): # e.g., `x += 1`, `s *= 'a'` (if s was string and * was supported for str)
                target_node = stmt_node.target
                if not isinstance(target_node, ast.Name):
                    raise ValueError("Unsupported augmented assignment: target must be a variable name.")

                var_name = target_node.id
                if var_name not in local_vars:
                    raise NameError(f"name '{var_name}' is not defined for augmented assignment")

                current_lhs_val = local_vars[var_name]
                evaluated_rhs_val = _evaluate_node_in_scope(stmt_node.value, local_vars)
                
                op_type = type(stmt_node.op)
                op_func = _ALLOWED_OPERATORS.get(op_type)

                if op_func is None:
                    raise ValueError(f"Unsupported operator for augmented assignment: {op_type.__name__}")

                # Perform operation with type checking
                if op_type == ast.Add:
                    if isinstance(current_lhs_val, (int, float)) and isinstance(evaluated_rhs_val, (int, float)):
                        value = op.add(current_lhs_val, evaluated_rhs_val)
                    elif isinstance(current_lhs_val, str) and isinstance(evaluated_rhs_val, str):
                        value = current_lhs_val + evaluated_rhs_val # String concatenation for +=
                    else:
                        raise TypeError(f"Unsupported operand types for augmented '+': {type(current_lhs_val).__name__} and {type(evaluated_rhs_val).__name__}")
                else: # Sub, Mult, Div require numeric operands for both LHS and RHS
                    if not (isinstance(current_lhs_val, (int, float)) and isinstance(evaluated_rhs_val, (int, float))):
                        op_symbol = {ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(op_type, op_type.__name__)
                        raise TypeError(f"Numeric operands required for augmented '{op_symbol}'. Got {type(current_lhs_val).__name__} and {type(evaluated_rhs_val).__name__}.")
                    
                    if op_type == ast.Div and evaluated_rhs_val == 0:
                        raise ZeroDivisionError("Division by zero in augmented assignment.")
                    value = op_func(current_lhs_val, evaluated_rhs_val)
                
                local_vars[var_name] = value
                # Augmented assignments do not set `script_result`.
            else:
                # Disallow any other statement types (If, For, While, Def, Class, Import, Print, etc.)
                raise ValueError(f"Unsupported statement type: {type(stmt_node).__name__}")
        
        # The final `script_result` must be one of the allowed types (or None).
        # This check is mostly a safeguard; type issues should be caught by _evaluate_node_in_scope.
        if script_result is not None and not isinstance(script_result, (int, float, str)):
             return 'Execution Blocked!' # Should have been caught earlier.
            
        return script_result

    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, RecursionError, NameError) as e:
        # Catch errors related to parsing, unsupported operations/statements,
        # type mismatches, division by zero, undefined variables, or excessive recursion.
        # For debugging, one might log `str(e)`.
        return 'Execution Blocked!'
    except Exception:
        # Catch any other unexpected exceptions as a fallback safety measure.
        # For debugging, one might log the exception.
        return 'Execution Blocked!'

if __name__ == '__main__':
    # Example Usage and Basic Tests for execute_user_code
    test_scripts = {
        # Basic expressions
        "1 + 1": 2,
        "'foo' + ' ' + 'bar'": "foo bar",
        "(10 + 5) * 2 - 7": 23,
        "-5.5 * 2": -11.0,
        # Assignments and variable usage
        "a = 10\nb = a * 2\nb + 5": 25,
        "x = 'hello'\ny = x + ' world'\ny": "hello world",
        "val = 100 / 4\nval": 25.0,
        # Augmented assignments
        "y = 10\ny += 5\ny": 15,
        "z = 'test'\nz += 'ing'\nz": "testing",
        "num = 20\nnum -= 3\nnum": 17,
        "factor = 4\nfactor *= 3.0\nfactor": 12.0,
        "div_val = 10\ndiv_val /= 2\ndiv_val": 5.0,
        # Script result is None if ends with assignment or is empty
        "": None,
        "a = 1\nb = 2": None,
        "1 + 1\na = 5": None, # Ends with assignment
        "a = 5\n1 + 1": 2,   # Ends with expression
        # Error cases
        "10 / 0": "Execution Blocked!",
        "a = 10 / 0": "Execution Blocked!",
        "b = 10\nb /= 0": "Execution Blocked!",
        "import os": "Execution Blocked!",
        "print('hello')": "Execution Blocked!",
        "undefined_var + 1": "Execution Blocked!",
        "a = 1\nb = undefined_var": "Execution Blocked!",
        "c = 1\nc += undefined_var": "Execution Blocked!",
        "1 + 's'": "Execution Blocked!",
        "a = 1 + 's'": "Execution Blocked!",
        "b = 1\nb += 's'": "Execution Blocked!",
        "c = 's'\nc -= 1": "Execution Blocked!",
        "c = 's'\nc *= 2": "Execution Blocked!", # String multiplication with * not supported via AugAssign here
        "def my_func(): pass": "Execution Blocked!",
        "if True: pass": "Execution Blocked!",
        "for i in range(1): pass": "Execution Blocked!",
        "a, b = 1, 2": "Execution Blocked!", # Multiple assignment targets
        "my_list = [1,2,3]": "Execution Blocked!", # List literal (ast.List is not ast.Constant)
        "1 + True": "Execution Blocked!", # True is not an allowed constant type
        "-(1+'a')": "Execution Blocked!", # Error inside unary op
    }

    all_tests_passed = True
    for i, (script, expected) in enumerate(test_scripts.items()):
        result = execute_user_code(script)
        match = result == expected
        if isinstance(result, float) and isinstance(expected, float): # Handle float comparison issues
            match = abs(result - expected) < 1e-9
        if not match:
            all_tests_passed = False
        script_display = script.replace('\n', '; ')
        print(f"Test {i+1}: Script: \"{script_display}\"")
        print(f"  Expected: {expected}, Got: {result}, Match: {match}\n")

    if all_tests_passed:
        print("All test cases passed!")
    else:
        print("Some test cases FAILED.")

    # Test for overly long script
    long_script_body = "v = 0\n" + "v += 1\n" * 1000 # Approx 8 * 1000 = 8000 chars
    # print(f"Length of long_script_body: {len(long_script_body)}") # For checking length vs limit
    # Expected: Execution Blocked! due to length limit (if limit is < len(long_script_body))
    # If limit is > len, then expected would be 1000.
    # Current limit is 2000, so this should be blocked.
    # If script was "v=0\n"+"v+=1\n"*200 + "v" (around 1600 chars), expected would be 200.
    
    # Test length limit
    # Max length is 2000. "v=0\n" is 4 chars. "v+=1\n" is 5 chars.
    # (2000 - 4) / 5 = 1996 / 5 = 399.2. So, 399 iterations of v+=1.
    # Total length: 4 + 399 * 5 = 4 + 1995 = 1999. Result should be 399.
    safe_len_script = "v=0\n" + "v+=1\n" * 399 + "v"
    print(f"Script: \"v=0; v+=1 (399 times); v\" (len={len(safe_len_script)})")
    res_safe_len = execute_user_code(safe_len_script)
    print(f"  Expected: 399, Got: {res_safe_len}, Match: {res_safe_len == 399}\n")
    if res_safe_len != 399: all_tests_passed = False

    # Just over the limit
    too_long_script = "v=0\n" + "v+=1\n" * 400 + "v" # 4 + 400*5 = 2004 chars
    print(f"Script: \"v=0; v+=1 (400 times); v\" (len={len(too_long_script)})")
    res_too_long = execute_user_code(too_long_script)
    print(f"  Expected: Execution Blocked!, Got: {res_too_long}, Match: {res_too_long == 'Execution Blocked!'}\n")
    if res_too_long != 'Execution Blocked!': all_tests_passed = False

    if all_tests_passed: # Re-check after length tests
        print("All final test checks passed!")
    else:
        print("Some final test checks FAILED.")
