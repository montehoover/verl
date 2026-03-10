def run_user_query(query):
    try:
        # Evaluate the expression
        result = eval(query)
        
        # Return as string
        return str(result)
    except:
        return "Error!"
