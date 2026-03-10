import re
import operator

def substitute_variables(expression: str, variables_dict: dict) -> str:
    """
    Substitutes variables in a mathematical expression string with their values from a dictionary.

    Args:
        expression: The string containing the mathematical expression.
        variables_dict: A dictionary where keys are variable names (strings)
                        and values are their corresponding numerical values.

    Returns:
        A new string with variables substituted by their values.
        If a variable in the expression is not found in the dictionary,
        it remains unchanged in the output string.
    """
    def replace_match(match):
        variable_name = match.group(0)
        # Check if the variable_name is a key in the dictionary
        if variable_name in variables_dict:
            return str(variables_dict[variable_name])
        # If not found, return the original variable name (or handle as an error/warning)
        return variable_name

    # Regex to find standalone variable names (alphanumeric, starting with a letter or underscore)
    # This regex looks for words, which is a common way to represent variables.
    # It avoids matching parts of numbers or function names if they are not keys in variables_dict.
    return re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', replace_match, expression)


def evaluate_expression(expression: str) -> str:
    """
    Evaluates a simple arithmetic expression string and returns the result as a string.
    Handles +, -, *, / operators with standard precedence (*, / before +, -).
    Handles negative numbers.

    Args:
        expression: The string containing the mathematical expression.
                     Assumes variables are already substituted with numerical values.

    Returns:
        A string representing the computed result, or an error message.
    """
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    # Tokenize into numbers (float) and operators (str)
    # Regex handles numbers (including negative) and operators +, -, *, /
    tokens_str = re.findall(r"(-?\d+\.?\d*|[+\-*/])", expression)
    
    if not tokens_str:
        return "Error: Empty expression"

    processed_tokens = []
    for t_str in tokens_str:
        if t_str in ops:
            processed_tokens.append(t_str)
        else:
            try:
                processed_tokens.append(float(t_str))
            except ValueError:
                return f"Error: Invalid token '{t_str}'"
    
    tokens = processed_tokens

    # Pass 1: Multiplication and Division
    temp_tokens_pass1 = list(tokens)
    idx = 0
    while idx < len(temp_tokens_pass1):
        token = temp_tokens_pass1[idx]
        if isinstance(token, str) and (token == '*' or token == '/'):
            if idx == 0 or idx == len(temp_tokens_pass1) - 1 or \
               not isinstance(temp_tokens_pass1[idx-1], float) or \
               not isinstance(temp_tokens_pass1[idx+1], float):
                return f"Error: Invalid expression structure for '{token}' operator"

            left_val = temp_tokens_pass1[idx-1]
            right_val = temp_tokens_pass1[idx+1]
            
            if token == '/' and right_val == 0:
                return "Error: Division by zero"
            
            result = ops[token](left_val, right_val)
            temp_tokens_pass1 = temp_tokens_pass1[:idx-1] + [result] + temp_tokens_pass1[idx+2:]
            idx -= 1 # Adjust index due to list modification
        else:
            idx += 1
    
    # Pass 2: Addition and Subtraction (left to right)
    if not temp_tokens_pass1:
        return "Error: Expression resolved to empty after '*/' processing"

    # Check if the first token is a number, it could be the only token (final result)
    if not isinstance(temp_tokens_pass1[0], float):
        if len(temp_tokens_pass1) == 1 and temp_tokens_pass1[0] in ops: # e.g. expression was just "+"
             return "Error: Expression is just an operator"
        return "Error: Expression does not start with a number after '*/' processing"

    current_val = temp_tokens_pass1[0]
    idx = 1
    while idx < len(temp_tokens_pass1):
        op_token = temp_tokens_pass1[idx]
        if not isinstance(op_token, str) or op_token not in ['+', '-']:
            return "Error: Invalid token sequence or missing operator after '*/' processing"

        if idx + 1 >= len(temp_tokens_pass1) or not isinstance(temp_tokens_pass1[idx+1], float):
            return f"Error: Operator '{op_token}' not followed by a number"
        
        right_val = temp_tokens_pass1[idx+1]
        current_val = ops[op_token](current_val, right_val)
        idx += 2
    
    # Format to string, show as int if it's a whole number
    if current_val == int(current_val):
        return str(int(current_val))
    return str(current_val)

if __name__ == '__main__':
    # Example Usage
    expr1 = "x + y * 2 - z"
    vars1 = {"x": 10, "y": 5, "z": 3}
    print(f"Expression: {expr1}, Variables: {vars1}")
    print(f"Substituted: {substitute_variables(expr1, vars1)}") # Expected: 10 + 5 * 2 - 3

    expr2 = "a * b + c / a"
    vars2 = {"a": 4, "b": 2} # 'c' is missing
    print(f"\nExpression: {expr2}, Variables: {vars2}")
    print(f"Substituted: {substitute_variables(expr2, vars2)}") # Expected: 4 * 2 + c / 4

    expr3 = "var1 - var2 + unknown_var"
    vars3 = {"var1": 100, "var2": 50}
    print(f"\nExpression: {expr3}, Variables: {vars3}")
    print(f"Substituted: {substitute_variables(expr3, vars3)}") # Expected: 100 - 50 + unknown_var

    expr4 = "sin(x) + cos(y)" # Assuming sin and cos are function names, not variables
    vars4 = {"x": 0.5, "y": 0.8}
    # The current regex might match 'sin' and 'cos' if they are not in vars4.
    # If 'sin' or 'cos' were keys in vars4, they would be replaced.
    # This behavior is acceptable if function names are not expected to be in the variables_dict.
    print(f"\nExpression: {expr4}, Variables: {vars4}")
    print(f"Substituted: {substitute_variables(expr4, vars4)}") # Expected: sin(0.5) + cos(0.8)

    expr5 = "val1 + 2 * val1 - 3"
    vars5 = {"val1": 7}
    print(f"\nExpression: {expr5}, Variables: {vars5}")
    print(f"Substituted: {substitute_variables(expr5, vars5)}") # Expected: 7 + 2 * 7 - 3

    print("\n--- Evaluate Expression Examples ---")
    expressions_to_evaluate = [
        "10 + 20",                  # Expected: 30
        "10 - 5 * 2",               # Expected: 0
        "100 / 10 * 2",             # Expected: 20
        "5 * 2 + 10 / 2 - 1",       # Expected: 14 (10 + 5 - 1)
        "-5 + 10",                  # Expected: 5
        "10 * -2",                  # Expected: -20
        "10 / 0",                   # Expected: Error: Division by zero
        "10 + ",                    # Expected: Error
        "* 2",                      # Expected: Error
        "10 + 2 *",                 # Expected: Error
        "5.5 + 2.5",                # Expected: 8.0 (or 8)
        "7",                        # Expected: 7
        "-7",                       # Expected: -7
        "10 / 4",                   # Expected: 2.5
        "2 * 3 + 4 / 2 - 1 * 5",    # Expected: 6 + 2 - 5 = 3
        "10 - -2",                  # Expected: 12
        "10 + -2",                  # Expected: 8
        "10 * 2 / 4 * 3",           # Expected: 5 / 4 * 3 = 1.25 * 3 = 3.75 -> 20/4*3 = 5*3 = 15
                                    # Correct: (10*2)/4*3 = 20/4*3 = 5*3 = 15
    ]
    for expr_eval in expressions_to_evaluate:
        substituted_expr = substitute_variables(expr_eval, {}) # Assuming no variables for these direct evaluations
        # Or, if expressions are already numbers:
        # result = evaluate_expression(expr_eval)
        # For now, let's assume evaluate_expression takes fully numeric strings
        result = evaluate_expression(expr_eval)
        print(f"Expression: \"{expr_eval}\", Evaluated: {result}")
