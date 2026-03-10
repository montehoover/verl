import re

# Helper function to tokenize the expression string
def tokenize(expression: str) -> list:
    """Converts an infix expression string into a list of tokens (numbers as floats, operators as strings)."""
    token_specification = [
        ('NUMBER',   r'\d+\.\d*|\.\d+|\d+'),  # Integer or decimal number
        ('OPERATOR', r'[+\-*/()]'),          # Arithmetic operators or parentheses
        ('SPACE',    r'\s+'),                # Spaces
        ('MISMATCH', r'.'),                  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, expression):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            tokens.append(float(value))
        elif kind == 'OPERATOR':
            tokens.append(value)
        elif kind == 'SPACE':
            continue  # Ignore spaces
        elif kind == 'MISMATCH':
            raise ValueError(f"Invalid character in expression: {value}")
    return tokens

# Helper function to convert infix tokens to RPN using Shunting-yard algorithm
def shunting_yard(tokens: list) -> list:
    """Converts a list of infix tokens to Reverse Polish Notation (RPN)."""
    output_queue = []
    operator_stack = []
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    # All supported operators (+, -, *, /) are left-associative.

    for token in tokens:
        if isinstance(token, float):  # Token is a number
            output_queue.append(token)
        elif token in precedence:  # Token is an operator
            while (operator_stack and 
                   operator_stack[-1] != '(' and
                   precedence.get(operator_stack[-1], 0) >= precedence[token]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':  # Token is an opening parenthesis
            operator_stack.append(token)
        elif token == ')':  # Token is a closing parenthesis
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if not operator_stack or operator_stack[-1] != '(':
                raise ValueError("Mismatched parentheses: missing '(' or misplaced ')'")
            operator_stack.pop()  # Pop the '('
        # No else needed here as tokenizer should ensure valid tokens or raise error

    while operator_stack:
        op = operator_stack.pop()
        if op == '(':
            # This implies mismatched parentheses, e.g., unclosed '('
            raise ValueError("Mismatched parentheses: unclosed '('")
        output_queue.append(op)
    
    return output_queue

# Helper function to evaluate an RPN token list
def evaluate_rpn(rpn_tokens: list) -> float:
    """Evaluates a list of tokens in RPN and returns the result."""
    operand_stack = []
    for token in rpn_tokens:
        if isinstance(token, float):  # Token is a number
            operand_stack.append(token)
        else:  # Token is an operator
            if len(operand_stack) < 2:
                raise ValueError("Invalid expression: insufficient operands for operator")
            num2 = operand_stack.pop()
            num1 = operand_stack.pop()
            
            if token == '+':
                operand_stack.append(num1 + num2)
            elif token == '-':
                operand_stack.append(num1 - num2)
            elif token == '*':
                operand_stack.append(num1 * num2)
            elif token == '/':
                if num2 == 0:
                    raise ValueError("Cannot divide by zero")
                operand_stack.append(num1 / num2)
            # No else needed as shunting_yard should only output valid operators
    
    if len(operand_stack) != 1:
        # Handles cases like empty RPN list or malformed expressions (e.g. "1 2")
        raise ValueError("Invalid expression: malformed RPN evaluation or empty expression")
    return operand_stack[0]

def basic_calculate(expression: str) -> float:
    """
    Evaluates a simple arithmetic expression string with support for parentheses.
    Handles addition, subtraction, multiplication, and division.

    Args:
        expression: The arithmetic expression string (e.g., "4 + 3 * (2 - 1)").

    Returns:
        The result of the expression as a float.

    Raises:
        ValueError: If the expression is invalid, contains invalid characters,
                    has mismatched parentheses, involves division by zero, or is malformed.
    """
    if not expression.strip():
        raise ValueError("Expression cannot be empty or contain only whitespace")
        
    tokens = tokenize(expression)
    if not tokens: # Should be caught by the strip check, but as a safeguard for tokenizer logic
        raise ValueError("Expression results in no tokens (e.g. only spaces, or empty after processing)")

    rpn_tokens = shunting_yard(tokens)
    if not rpn_tokens and any(isinstance(t, float) for t in tokens): # e.g. expression was just "()" or "(()))"
        pass # This will be caught by evaluate_rpn if rpn_tokens is empty
    elif not rpn_tokens and not any(isinstance(t, float) for t in tokens): # e.g. expression was just "()"
         raise ValueError("Invalid expression: contains only operators/parentheses without numbers")


    result = evaluate_rpn(rpn_tokens)
    return result
