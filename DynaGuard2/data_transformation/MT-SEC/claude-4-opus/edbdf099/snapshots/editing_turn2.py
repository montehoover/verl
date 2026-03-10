def evaluate_expression(expression):
    """
    Evaluate a simple arithmetic expression or string concatenation given as a string.
    
    Args:
        expression (str): The expression to evaluate (arithmetic or string concatenation)
        
    Returns:
        float, str, or error message: The result of the expression or an error message
    """
    try:
        # Check for empty expression
        if not expression.strip():
            return "Error: Empty expression"
        
        # Try to evaluate the expression
        result = eval(expression)
        
        return result
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except NameError:
        return "Error: Undefined variable or invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


# Test the function
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3",
        "10 - 5",
        "4 * 6",
        "15 / 3",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "10 / 0",
        "2 + + 3",
        "hello",
        "",
        "3.14 * 2",
        "100 / 4 - 5",
        "'hello' + ' ' + 'world'",
        "'foo' + 'bar'",
        "'test' * 3",
        "'a' + 'b' + 'c'",
        "5 + 'hello'",
        "'The answer is: ' + str(42)",
        "len('hello')",
        "'HELLO'.lower()",
        "'hello'.upper()",
        "' spaces '.strip()"
    ]
    
    for expr in test_expressions:
        result = evaluate_expression(expr)
        print(f"{expr} = {result}")
