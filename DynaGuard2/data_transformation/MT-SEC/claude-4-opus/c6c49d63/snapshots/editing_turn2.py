import re

def basic_calculate(expression):
    """
    Perform basic arithmetic operations on a string expression.
    
    Args:
        expression (str): Arithmetic expression like '4 + 3' or '(2 + 3) * 4'
    
    Returns:
        float: Result of the operation
    """
    # Remove all whitespace
    expression = expression.replace(' ', '')
    
    # Validate expression contains only valid characters
    if not re.match(r'^[\d\+\-\*\/\(\)\.]+$', expression):
        raise ValueError("Invalid characters in expression")
    
    # Helper function to find matching closing parenthesis
    def find_matching_paren(expr, start):
        count = 1
        i = start + 1
        while i < len(expr) and count > 0:
            if expr[i] == '(':
                count += 1
            elif expr[i] == ')':
                count -= 1
            i += 1
        return i - 1
    
    # Helper function to evaluate expression without parentheses
    def evaluate_simple(expr):
        # First handle multiplication and division
        while '*' in expr or '/' in expr:
            # Find first * or /
            mul_pos = expr.find('*')
            div_pos = expr.find('/')
            
            if mul_pos == -1:
                pos = div_pos
                op = '/'
            elif div_pos == -1:
                pos = mul_pos
                op = '*'
            else:
                if mul_pos < div_pos:
                    pos = mul_pos
                    op = '*'
                else:
                    pos = div_pos
                    op = '/'
            
            # Find the numbers around the operator
            # Find start of left number
            left_start = pos - 1
            while left_start > 0 and (expr[left_start-1].isdigit() or expr[left_start-1] == '.'):
                left_start -= 1
            
            # Handle negative numbers
            if left_start > 0 and expr[left_start-1] == '-':
                if left_start == 1 or expr[left_start-2] in '+-*/(':
                    left_start -= 1
            
            # Find end of right number
            right_end = pos + 1
            if right_end < len(expr) and expr[right_end] == '-':
                right_end += 1
            while right_end < len(expr) and (expr[right_end].isdigit() or expr[right_end] == '.'):
                right_end += 1
            
            left_num = float(expr[left_start:pos])
            right_num = float(expr[pos+1:right_end])
            
            if op == '*':
                result = left_num * right_num
            else:
                if right_num == 0:
                    raise ValueError("Division by zero is not allowed")
                result = left_num / right_num
            
            expr = expr[:left_start] + str(result) + expr[right_end:]
        
        # Then handle addition and subtraction
        # Convert expression to handle negative numbers
        tokens = []
        current_num = ''
        i = 0
        
        while i < len(expr):
            if expr[i] in '+-':
                if current_num:
                    tokens.append(float(current_num))
                    current_num = ''
                if i == 0 or expr[i-1] in '+-':
                    current_num = expr[i]
                else:
                    tokens.append(expr[i])
            else:
                current_num += expr[i]
            i += 1
        
        if current_num:
            tokens.append(float(current_num))
        
        # Evaluate tokens
        if not tokens:
            return 0.0
        
        result = tokens[0] if isinstance(tokens[0], (int, float)) else 0.0
        i = 1
        while i < len(tokens):
            if tokens[i] == '+':
                result += tokens[i+1]
                i += 2
            elif tokens[i] == '-':
                result -= tokens[i+1]
                i += 2
            else:
                i += 1
        
        return result
    
    # Main evaluation function
    def evaluate(expr):
        # Handle parentheses first
        while '(' in expr:
            start = expr.find('(')
            end = find_matching_paren(expr, start)
            inner_result = evaluate(expr[start+1:end])
            expr = expr[:start] + str(inner_result) + expr[end+1:]
        
        return evaluate_simple(expr)
    
    return float(evaluate(expression))
