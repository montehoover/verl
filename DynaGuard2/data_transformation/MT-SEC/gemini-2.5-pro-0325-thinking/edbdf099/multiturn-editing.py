import ast

def _is_ast_safe(node_to_check):
    """
    Helper function to validate the AST of the user script.
    Only allows a restricted set of AST nodes and operations.
    """
    # Whitelist of AST node types allowed in the script
    allowed_node_types = {
        ast.Module,     # Root node of a script
        ast.Expr,       # Expression statement (e.g., "1 + 2" on a line)
        ast.Constant,   # Literals: numbers, strings, True, False, None
        ast.Name,       # Variable names (e.g., x)
        ast.Assign,     # Assignment (e.g., x = 1)
        ast.BinOp,      # Binary operations (e.g., +, -, *, /)
        ast.UnaryOp,    # Unary operations (e.g., -x)
        # ast.Load, ast.Store are contexts for ast.Name, not separate node types
        # walked by ast.walk in this manner. ast.Name itself is allowed.
    }
    # Whitelist of operator types allowed for BinOp and UnaryOp
    allowed_operator_types = {
        ast.Add, ast.Sub, ast.Mult, ast.Div,  # For BinOp
        ast.USub, ast.UAdd                    # For UnaryOp
    }

    for node in ast.walk(node_to_check):
        node_type = type(node)

        if node_type not in allowed_node_types:
            return False  # Node type itself is not allowed

        # If node type is allowed, check specific operator types for BinOp/UnaryOp
        if isinstance(node, ast.BinOp):
            if type(node.op) not in allowed_operator_types:
                return False
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_operator_types:
                return False
    return True

def process_user_code(code_script: str):
    """
    Securely parses and executes a user-provided Python script string
    containing basic arithmetic or string operations.

    Args:
        code_script: The Python script as a string.

    Returns:
        The result of the script execution, or an error message string
        (e.g., "Execution Blocked!", "Error: Invalid syntax.").
    """
    if not code_script.strip():
        return "Error: Script is empty."

    try:
        module_node = ast.parse(code_script, mode='exec')
    except SyntaxError:
        return "Error: Invalid syntax."

    if not _is_ast_safe(module_node):
        return "Execution Blocked!"

    restricted_globals = {"__builtins__": {}}
    local_scope = {}

    try:
        # Compile the entire AST module once
        compiled_script = compile(module_node, '<user_script>', 'exec')
        exec(compiled_script, restricted_globals, local_scope)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except TypeError as e:
        return f"Error: Type error during execution ({str(e)})."
    except NameError as e:
        return f"Error: Name error during execution ({str(e)})."
    except Exception as e:
        # Catch any other unexpected errors during the initial exec
        return f"Error: An unexpected error occurred during execution - {str(e)}"

    # Determine the result after successful execution of the whole script
    if module_node.body:
        last_statement_node = module_node.body[-1]
        
        # If the last statement was an expression, evaluate it in the final scope
        if isinstance(last_statement_node, ast.Expr):
            # The value of an Expr statement is its expression node.
            expr_to_eval_node = last_statement_node.value
            try:
                # Compile this specific expression for eval
                code_obj_expr = compile(ast.Expression(body=expr_to_eval_node), '<last_expr>', 'eval')
                return eval(code_obj_expr, restricted_globals, local_scope)
            except Exception as e: # Catch errors during this specific eval too
                return f"Error: Could not evaluate final expression - {str(e)}"
        
        # If the last statement was an assignment, the result is the value of the (first) target
        elif isinstance(last_statement_node, ast.Assign):
            if last_statement_node.targets and isinstance(last_statement_node.targets[0], ast.Name):
                var_name = last_statement_node.targets[0].id
                if var_name in local_scope:
                    return local_scope[var_name]
                else:
                    # This case should ideally not be hit if exec was successful
                    return "Error: Assigned variable not found in final scope."
            else:
                # Multiple assignment targets or complex targets (e.g., a[0]=1, which is disallowed by AST check)
                # For simple allowed assignments, this path shouldn't be common.
                return None # No clear single result for complex assignments
        else:
            # Last statement is not Expr or Assign (e.g., pass, if allowed), no specific result.
            return None
    else:
        # Script was empty or only comments (already handled by strip check or parse)
        return None


if __name__ == '__main__':
    test_scripts = {
        "Simple arithmetic": "1 + 2",
        "Order of operations": "2 * (3 + 4) / 2",
        "String concatenation": "'hello' + ' ' + 'world'",
        "String repetition": "'abc' * 3",
        "Variable assignment and use": "x = 10\ny = x * 2\ny + 5",
        "Result is last assignment": "val = 100 / 4",
        "Result is last expression": "a = 'foo'\nb = 'bar'\na + b",
        "Unary minus": "x = 5\n-x",
        "Empty script": "",
        "Whitespace script": "   \n  ",
        "Syntax error": "1 +",
        "Zero division": "1 / 0",
        "Type error": "1 + 'a'",
        "Name error (undefined var in expr)": "x + 1",
        "Name error (undefined var in assignment)": "y = z",
        "Unsafe import": "__import__('os').system('ls')",
        "Unsafe function call": "print('hello')",
        "Unsafe attribute access": "x = ().__class__",
        "Unsafe eval call": "eval('1+1')",
        "Unsafe subscript": "my_list = [1,2]\my_list[0]",
        "Unsafe power op": "2 ** 3", # Disallowed by current _is_ast_safe op whitelist
        "Allowed multi-line script ending in expr": "a=1\nb=2\na+b",
        "Allowed multi-line script ending in assign": "a=1\nb=2\nc=a+b",
        "Script with only assignments": "a=1\nb=2", # Should return value of b
        "Script with None": "None",
        "Script with just a variable (name error)": "my_var"
    }

    for description, script in test_scripts.items():
        print(f"Test: {description}\nScript:\n```\n{script}\n```")
        result = process_user_code(script)
        print(f"Result: {result}\n---")

    # Example of a script that is safe but has no specific return value by current logic
    print("Test: Safe script with no specific return (e.g. only assignments not last)")
    script = "x = 1\ny = 2" # process_user_code will return value of y (last assignment)
    # If it was "x=1\npass", it would be None (if pass was allowed)
    print(f"Script:\n```\n{script}\n```")
    result = process_user_code(script)
    print(f"Result: {result}\n---")
