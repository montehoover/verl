def calculate_expression(expr_input: str) -> str:
    """
    Safely evaluates user-provided mathematical expressions written in Python syntax.
    - Supports numeric constants, parentheses, +, -, *, /, //, %, **, unary +/-
    - Supports a safe subset of math functions and constants (e.g., sin, cos, sqrt, pi, e)
    - Rejects any other Python constructs (attributes, names, comprehensions, etc.)
    Returns the result as a string, or 'Computation Error!' for any invalid/unsafe input.
    """
    try:
        if not isinstance(expr_input, str):
            return 'Computation Error!'
        s = expr_input.strip()
        if not s:
            return 'Computation Error!'

        import ast
        import operator as op
        import math

        allowed_bin_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Mod: op.mod,
            ast.Pow: op.pow,
        }
        allowed_unary_ops = {
            ast.UAdd: op.pos,
            ast.USub: op.neg,
        }

        allowed_funcs = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'fabs': math.fabs,
            'floor': math.floor,
            'ceil': math.ceil,
            'factorial': math.factorial,
            'pow': math.pow,
            'hypot': math.hypot,
            'degrees': math.degrees,
            'radians': math.radians,
        }
        allowed_consts = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,
        }

        node = __import__('ast').parse(s, mode='eval')

        def eval_node(n):
            if isinstance(n, ast.Expression):
                return eval_node(n.body)
            if isinstance(n, ast.Constant):
                if isinstance(n.value, (int, float)):
                    return n.value
                raise ValueError('unsupported constant')
            if isinstance(n, ast.BinOp):
                func = allowed_bin_ops.get(type(n.op))
                if func is None:
                    raise ValueError('unsupported operator')
                left = eval_node(n.left)
                right = eval_node(n.right)
                return func(left, right)
            if isinstance(n, ast.UnaryOp):
                func = allowed_unary_ops.get(type(n.op))
                if func is None:
                    raise ValueError('unsupported unary operator')
                val = eval_node(n.operand)
                return func(val)
            if isinstance(n, ast.Name):
                if n.id in allowed_consts:
                    return allowed_consts[n.id]
                raise ValueError('unknown name')
            if isinstance(n, ast.Call):
                # Only allow calls to whitelisted function names; no attributes, no keywords, no star args
                if not isinstance(n.func, ast.Name):
                    raise ValueError('unsupported callable')
                fn = allowed_funcs.get(n.func.id)
                if fn is None:
                    raise ValueError('function not allowed')
                if n.keywords:
                    # No keyword args or **kwargs
                    raise ValueError('keywords not allowed')
                args = [eval_node(a) for a in n.args]
                if not all(isinstance(a, (int, float)) for a in args):
                    raise ValueError('non-numeric argument')
                return fn(*args)
            # Anything else (attributes, subscripts, comprehensions, lambdas, etc.) is not allowed
            raise ValueError('unsupported expression node')

        result = eval_node(node)
        return str(result)
    except Exception:
        return 'Computation Error!'
