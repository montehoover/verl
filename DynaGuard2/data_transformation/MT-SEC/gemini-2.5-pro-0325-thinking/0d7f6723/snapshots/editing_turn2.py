import re

def evaluate_expression(expression_str: str):
    """
    Evaluates a simple arithmetic expression string involving addition and subtraction,
    or a string concatenation expression.

    Args:
        expression_str: The arithmetic or string expression string.

    Returns:
        The result of the expression (number for arithmetic, string for concatenation),
        or 'Invalid Expression' if malformed or unsupported.
    """
    expression = expression_str.strip()
    if not expression:
        return 'Invalid Expression'

    # Try to parse as a string expression first
    # String literal pattern (captures content): '...'
    string_literal_capture_pattern = r"'((?:[^'\\]|\\.)*)'"
    # String literal pattern (non-capturing, for structure matching): '...'
    string_literal_no_capture_pattern = r"'(?:[^'\\]|\\.)*'"
    
    # Full string expression pattern: a string literal, optionally followed by '+' and more string literals
    full_string_expr_pattern = (
        r"^\s*" + string_literal_no_capture_pattern + r"\s*"
        r"(?:\+\s*" + string_literal_no_capture_pattern + r"\s*)*$"
    )

    if re.fullmatch(full_string_expr_pattern, expression):
        # It's a string expression. Extract all string parts and concatenate.
        parts_contents = re.findall(string_literal_capture_pattern, expression)
        # Unescape \' and \\
        unescaped_parts = [p.replace("\\'", "'").replace("\\\\", "\\") for p in parts_contents]
        return "".join(unescaped_parts)

    # If not a string expression, try numeric evaluation
    # Original numeric validation regex:
    numeric_validation_regex = r"^\s*([-+]?\s*\d+(\.\d+)?\s*([+-]\s*\d+(\.\d+)?\s*)*)$"
    
    if not re.fullmatch(numeric_validation_regex, expression):
        # If it doesn't match the numeric pattern either (and wasn't a string expression)
        return 'Invalid Expression'

    # --- Existing numeric evaluation logic ---
    # Remove all internal whitespace to simplify parsing for the numeric tokenizer
    expression_no_whitespace = "".join(expression.split())


    # The original code had a 'pass' block for some checks here, which are effectively
    # covered by the numeric_validation_regex if it's comprehensive.
    # Example: re.search(r"[+\-]{2,}", expression_no_whitespace) ... pass

    try:
        # Split by operators, keeping the operators
        tokens = re.split(r'([+\-])', expression_no_whitespace)
        
        # Filter out empty strings that can result from splitting
        tokens = [token for token in tokens if token]

        if not tokens: # Should be caught by numeric_validation_regex
            return 'Invalid Expression'

        # Handle leading negative/positive signs for the first number
        if tokens[0] == '-':
            if len(tokens) < 2 or not tokens[1].replace('.', '', 1).isdigit():
                return 'Invalid Expression' # e.g., just "-" or "-+"
            tokens = ["-" + tokens[1]] + tokens[2:]
        elif tokens[0] == '+': # Leading plus is redundant if followed by a number
            if len(tokens) < 2 or not tokens[1].replace('.', '', 1).isdigit():
                return 'Invalid Expression' # e.g., just "+" or "+-"
            tokens = tokens[1:]

        # First token must be a valid number string
        try:
            # Validate that the first token (which should now be a complete number string) is a float
            result = float(tokens[0])
        except (ValueError, IndexError): # IndexError if tokens became empty
            return 'Invalid Expression'


        i = 1
        while i < len(tokens):
            operator = tokens[i]
            # Operator must be + or - (already guaranteed by split pattern, but good check)
            if operator not in ['+', '-']:
                return 'Invalid Expression' 
            
            if i + 1 >= len(tokens): # Operator not followed by a number
                return 'Invalid Expression'

            try:
                operand_str = tokens[i+1]
                operand = float(operand_str)
            except ValueError:
                return 'Invalid Expression' # Subsequent token not a valid number

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
    # Test cases for numeric operations
    print(f"Numeric Ops:")
    print(f"  '10 + 20': {evaluate_expression('10 + 20')}")  # Expected: 30
    print(f"  '  30 - 10  ': {evaluate_expression('  30 - 10  ')}")  # Expected: 20
    print(f"  '100': {evaluate_expression('100')}")  # Expected: 100
    print(f"  '-5 + 15': {evaluate_expression('-5 + 15')}") # Expected: 10
    print(f"  '10 + 20 - 5': {evaluate_expression('10 + 20 - 5')}")  # Expected: 25
    print(f"  '10.5 + 2.5': {evaluate_expression('10.5 + 2.5')}") # Expected: 13.0
    print(f"  '-10': {evaluate_expression('-10')}") # Expected: -10
    print(f"  '+10 - 5': {evaluate_expression('+10 - 5')}") # Expected: 5
    print(f"  '5+5-3+10-2': {evaluate_expression('5+5-3+10-2')}") # Expected: 15
    print(f"  '5.2 + 3.8 - 2': {evaluate_expression('5.2 + 3.8 - 2')}") # Expected: 7.0
    
    # Test cases for string operations
    print(f"\nString Ops:")
    print(f'''  "'hello' + ' world'": {evaluate_expression("'hello' + ' world'")})''') # Expected: "hello world"
    print(f'''  "'justastring'": {evaluate_expression("'justastring'")})''') # Expected: "justastring"
    print(f'''  "'  leading space' + 'trailing space  '": {evaluate_expression("'  leading space' + 'trailing space  '")})''') # Expected: "  leading spacetrailing space  "
    print(f'''  "'str1' + 'str2' + 'str3'": {evaluate_expression("'str1' + 'str2' + 'str3'")})''') # Expected: "str1str2str3"
    print(f'''  "'escaped \\'quote\\'' + ' and \\\\backslash'": {evaluate_expression("'escaped \\'quote\\'' + ' and \\\\backslash'")})''') # Expected: "escaped 'quote' and \\backslash"
    print(f'''  "'' + 'nonempty'": {evaluate_expression("'' + 'nonempty'")})''') # Expected: "nonempty"
    print(f'''  "'nonempty' + ''": {evaluate_expression("'nonempty' + ''")})''') # Expected: "nonempty"
    print(f'''  "'' + ''": {evaluate_expression("'' + ''")})''') # Expected: ""

    # Test cases for invalid expressions
    print(f"\nInvalid Ops:")
    print(f"  '5 - -5': {evaluate_expression('5 - -5')}") # Expected: Invalid Expression (by current numeric regex)
    print(f"  '10 * 5': {evaluate_expression('10 * 5')}")  # Expected: Invalid Expression
    print(f"  '10 / 5': {evaluate_expression('10 / 5')}")  # Expected: Invalid Expression
    print(f"  'abc + 10': {evaluate_expression('abc + 10')}")  # Expected: Invalid Expression
    print(f"  '10 +': {evaluate_expression('10 +')}")  # Expected: Invalid Expression
    print(f"  '+ 10': {evaluate_expression('+ 10')}") # Expected: 10 (Handled by numeric parser)
    print(f"  '++10': {evaluate_expression('++10')}") # Expected: Invalid Expression
    print(f"  '10 + twenty': {evaluate_expression('10 + twenty')}") # Expected: Invalid Expression
    print(f"  '(10 + 5)': {evaluate_expression('(10 + 5)')}") # Expected: Invalid Expression
    print(f"  '   ': {evaluate_expression('   ')}") # Expected: Invalid Expression
    print(f"  '': {evaluate_expression('')}") # Expected: Invalid Expression
    print(f"  '10 - - 5': {evaluate_expression('10 - - 5')}") # Expected: Invalid Expression
    print(f"  '10--5': {evaluate_expression('10--5')}") # Expected: Invalid Expression
    print(f"  '10 + + 5': {evaluate_expression('10 + + 5')}") # Expected: Invalid Expression
    print(f'''  "'hello' - 'world'": {evaluate_expression("'hello' - 'world'")})''') # Expected: Invalid Expression
    print(f'''  "'hello' + 5": {evaluate_expression("'hello' + 5")})''') # Expected: Invalid Expression
    print(f'''  "5 + 'hello'": {evaluate_expression("5 + 'hello'")})''') # Expected: Invalid Expression
    print(f'''  "'unterminated": {evaluate_expression("'unterminated")}")''') # Expected: Invalid Expression
    print(f'''  "'a' + 'b' +": {evaluate_expression("'a' + 'b' +")})''') # Expected: Invalid Expression
    print(f'''  "'a' 'b'": {evaluate_expression("'a' 'b'")})''') # Expected: Invalid Expression
