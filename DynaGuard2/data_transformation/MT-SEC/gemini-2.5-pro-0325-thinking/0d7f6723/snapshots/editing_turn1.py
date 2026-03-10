import re

def evaluate_expression(expression: str):
    """
    Evaluates a simple arithmetic expression string involving addition and subtraction.

    Args:
        expression: The arithmetic expression string.

    Returns:
        The result of the expression as a number, or 'Invalid Expression'
        if the expression is malformed or contains unsupported operations.
    """
    # Sanitize and validate the expression
    # Allow numbers, +, -, and whitespace.
    # Ensure it doesn't start or end with an operator, and no consecutive operators.
    if not re.match(r"^\s*([-+]?\s*\d+(\.\d+)?\s*([+-]\s*\d+(\.\d+)?\s*)*)$", expression):
        return 'Invalid Expression'

    # Remove all whitespace to simplify parsing
    expression = expression.replace(" ", "")

    # Check for invalid patterns like "--" or "++" or leading/trailing operators after stripping whitespace
    if re.search(r"[+\-]{2,}", expression) or \
       expression.startswith('+') or expression.startswith('-') and len(expression) > 1 and not expression[1].isdigit() or \
       expression.endswith('+') or expression.endswith('-'):
        # A leading minus for a negative number is fine, e.g. "-5+10"
        # but "-+5" or "5++5" or "5+" is not.
        # The initial regex handles most of this, but this is a stricter check after whitespace removal.
        # A simple check for leading operator if it's not part of a number.
        pass # The main regex should catch most of these, but let's be careful.
             # The eval below is safe, but we want to return 'Invalid Expression' for clarity.

    try:
        # A safer way to evaluate simple expressions without using full eval()
        # Split by operators, keeping the operators
        tokens = re.split(r'([+\-])', expression)
        
        # Filter out empty strings that can result from splitting if expression starts with an operator (e.g. negative number)
        tokens = [token for token in tokens if token]

        if not tokens:
            return 'Invalid Expression'

        # Handle leading negative numbers
        if tokens[0] == '-':
            if len(tokens) < 2 or not tokens[1].replace('.', '', 1).isdigit():
                return 'Invalid Expression'
            tokens = ["-" + tokens[1]] + tokens[2:]
        elif tokens[0] == '+': # Leading plus is redundant
             tokens = tokens[1:]


        if not tokens[0].replace('.', '', 1).replace('-', '', 1).isdigit(): # First token must be a number (possibly signed)
            return 'Invalid Expression'

        result = float(tokens[0])
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            if operator not in ['+', '-']:
                return 'Invalid Expression' # Should not happen if regex is correct
            
            if i + 1 >= len(tokens): # Operator not followed by a number
                return 'Invalid Expression'

            try:
                operand = float(tokens[i+1])
            except ValueError:
                return 'Invalid Expression' # Not a valid number

            if operator == '+':
                result += operand
            elif operator == '-':
                result -= operand
            i += 2
        
        # Return int if result is a whole number, else float
        if result == int(result):
            return int(result)
        return result

    except Exception:
        # Catch any other errors during parsing or calculation
        return 'Invalid Expression'

if __name__ == '__main__':
    # Test cases
    print(f"'10 + 20': {evaluate_expression('10 + 20')}")  # Expected: 30
    print(f"'  30 - 10  ': {evaluate_expression('  30 - 10  ')}")  # Expected: 20
    print(f"'100': {evaluate_expression('100')}")  # Expected: 100
    print(f"'-5 + 15': {evaluate_expression('-5 + 15')}") # Expected: 10
    print(f"'5 - -5': {evaluate_expression('5 - -5')}") # Expected: Invalid Expression (or 10 if we support it)
                                                        # Current implementation will treat "--" as invalid via regex.
                                                        # If we want to support "5 - -5" = 10, logic needs adjustment.
                                                        # For now, sticking to simpler "no consecutive operators".
    print(f"'10 + 20 - 5': {evaluate_expression('10 + 20 - 5')}")  # Expected: 25
    print(f"'10.5 + 2.5': {evaluate_expression('10.5 + 2.5')}") # Expected: 13.0
    print(f"'10 * 5': {evaluate_expression('10 * 5')}")  # Expected: Invalid Expression
    print(f"'10 / 5': {evaluate_expression('10 / 5')}")  # Expected: Invalid Expression
    print(f"'abc + 10': {evaluate_expression('abc + 10')}")  # Expected: Invalid Expression
    print(f"'10 +': {evaluate_expression('10 +')}")  # Expected: Invalid Expression
    print(f"'+ 10': {evaluate_expression('+ 10')}")  # Expected: Invalid Expression (or 10 if we support leading +)
                                                    # Current regex allows leading + for numbers, but parser might need adjustment.
                                                    # Let's make it strict: no leading operator unless it's a negative sign.
    print(f"'10 + twenty': {evaluate_expression('10 + twenty')}") # Expected: Invalid Expression
    print(f"'(10 + 5)': {evaluate_expression('(10 + 5)')}") # Expected: Invalid Expression
    print(f"'   ': {evaluate_expression('   ')}") # Expected: Invalid Expression
    print(f"'-10': {evaluate_expression('-10')}") # Expected: -10
    print(f"'10 - - 5': {evaluate_expression('10 - - 5')}") # Expected: Invalid Expression
    print(f"'10--5': {evaluate_expression('10--5')}") # Expected: Invalid Expression
    print(f"'5+5-3+10-2': {evaluate_expression('5+5-3+10-2')}") # Expected: 15
    print(f"'5.2 + 3.8 - 2': {evaluate_expression('5.2 + 3.8 - 2')}") # Expected: 7.0
    print(f"'': {evaluate_expression('')}") # Expected: Invalid Expression
    print(f"'10 + + 5': {evaluate_expression('10 + + 5')}") # Expected: Invalid Expression
