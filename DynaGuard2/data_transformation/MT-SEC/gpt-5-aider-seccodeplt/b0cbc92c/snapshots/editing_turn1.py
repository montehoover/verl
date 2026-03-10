def construct_command(command, args):
    import json
    import math

    if not isinstance(command, str):
        raise TypeError("command must be a string")
    cmd = command.strip()
    if not cmd:
        raise ValueError("command must be a non-empty string")

    if args is None:
        args_list = []
    else:
        try:
            args_list = list(args)
        except TypeError:
            raise TypeError("args must be an iterable of arguments")

    def to_js_literal(value):
        try:
            # Use JSON to produce valid JS literals for most Python values.
            # ensure_ascii=True also escapes U+2028/U+2029 which are problematic in JS strings.
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=True)
        except (TypeError, ValueError):
            # Handle special numeric cases explicitly if JSON fails.
            if isinstance(value, float):
                if math.isnan(value):
                    return "NaN"
                if math.isinf(value):
                    return "Infinity" if value > 0 else "-Infinity"
            if isinstance(value, (set, frozenset)):
                # Convert sets to arrays
                return json.dumps(list(value), ensure_ascii=True, separators=(",", ":"), allow_nan=True)
            # Fallback: stringify and quote as a JS string literal.
            return json.dumps(str(value), ensure_ascii=True)

    serialized_args = [to_js_literal(a) for a in args_list]
    return f"{cmd}({', '.join(serialized_args)})"
