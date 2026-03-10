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


def execute_command(command, args):
    import subprocess
    import json

    try:
        expr = construct_command(command, args)
    except Exception as e:
        return f"Execution error: {e}"

    node_script = """
(function () {
  const expr = %s;

  function toStringValue(v) {
    if (typeof v === 'string') return v;
    if (v === undefined) return '';
    try {
      const s = JSON.stringify(v);
      if (typeof s === 'string') return s;
    } catch (_) {}
    try { return String(v); } catch (_) { return ''; }
  }

  function print(v) {
    try {
      process.stdout.write(toStringValue(v));
    } catch (e) {
      // As a last resort, coerce and write something
      try { process.stdout.write(String(v)); } catch (_) {}
    }
  }

  function printError(err) {
    try {
      const msg = (err && err.stack) ? String(err.stack) : String(err);
      process.stderr.write(msg);
    } catch (_) {
      try { process.stderr.write('Unknown error'); } catch (_) {}
    }
  }

  try {
    const res = eval(expr);
    if (res && typeof res.then === 'function') {
      Promise.resolve(res).then(
        (v) => { print(v); },
        (err) => { printError(err); process.exitCode = 1; }
      );
    } else {
      print(res);
    }
  } catch (err) {
    printError(err);
    process.exitCode = 1;
  }
})();
""" % json.dumps(expr)

    try:
        completed = subprocess.run(
            ["node", "-e", node_script],
            capture_output=True,
            text=True
        )
        if completed.returncode == 0:
            return completed.stdout
        else:
            return completed.stderr or completed.stdout or f"Execution failed with exit code {completed.returncode}"
    except FileNotFoundError:
        return "Execution error: Node.js runtime not found. Please install Node.js to execute JavaScript commands."
    except Exception as e:
        return f"Execution error: {e}"
