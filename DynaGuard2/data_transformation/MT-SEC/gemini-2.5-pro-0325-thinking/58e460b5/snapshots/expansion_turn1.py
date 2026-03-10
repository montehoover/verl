import re

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
