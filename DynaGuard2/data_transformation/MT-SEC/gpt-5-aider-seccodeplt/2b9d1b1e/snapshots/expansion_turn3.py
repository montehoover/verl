import ast
from typing import List, Set, Union, Optional, Dict, Any


class _OperationCollector(ast.NodeVisitor):
    """
    Walks a Python AST to:
    - Collect basic arithmetic and string operations.
    - Enforce a strict whitelist of allowed nodes to reject potentially harmful code.
    """

    # Allowed AST node types (strict whitelist)
    _ALLOWED_NODES = {
        # Modules / top-level
        ast.Module,
        ast.Interactive,
        ast.Expression,
        ast.Expr,

        # Assignments
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,  # allow annotated assignment (value will still be validated)

        # Names and constants
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Store,

        # Operations
        ast.BinOp,
        ast.UnaryOp,
        ast.Subscript,
        ast.Slice,
        ast.ExtSlice,  # older versions
        ast.Index,     # older versions

        # Operators (visited as nodes in the AST)
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,

        # Containers / literals
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
        ast.ListElt if hasattr(ast, "ListElt") else ast.AST,  # compatibility; will be ignored

        # F-strings
        ast.JoinedStr,
        ast.FormattedValue,
    }

    # Definitely dangerous or out-of-scope nodes
    _DISALLOWED_EXPLICIT = {
        ast.Import,
        ast.ImportFrom,
        ast.Call,
        ast.Attribute,
        ast.Lambda,
        ast.With,
        ast.AsyncWith,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Return,
        ast.ClassDef,
        ast.Global,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Delete,
        ast.While,
        ast.For,
        ast.AsyncFor,
        ast.If,
        ast.Match,  # Python 3.10+
        ast.NamedExpr,  # walrus
        ast.Starred,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Compare,
        ast.BoolOp,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.MatMult,
        ast.Invert,
        ast.Not,
        ast.And,
        ast.Or,
        ast.IfExp,
        ast.Comprehension if hasattr(ast, "Comprehension") else ast.AST,  # compatibility
    }

    def __init__(self) -> None:
        self._ops: List[str] = []
        self._seen: Set[str] = set()

    def result(self) -> List[str]:
        return list(self._ops)

    def _add_op(self, op: str) -> None:
        if op not in self._seen:
            self._ops.append(op)
            self._seen.add(op)

    # Basic, lightweight type inference for classifying string vs arithmetic
    def _infer_type(self, node: ast.AST) -> str:
        # returns: "str", "num", or "unknown"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "str"
            if isinstance(node.value, (int, float, complex)):
                return "num"
            return "unknown"
        if isinstance(node, ast.JoinedStr):
            return "str"
        if isinstance(node, ast.BinOp):
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if isinstance(node.op, ast.Add):
                if lt == "str" or rt == "str":
                    return "str"
                if lt == "num" and rt == "num":
                    return "num"
                return "unknown"
            if isinstance(node.op, ast.Mult):
                # str * num or num * str
                if (lt == "str" and rt in ("num", "unknown")) or (rt == "str" and lt in ("num", "unknown")):
                    return "str"
                if lt == "num" and rt == "num":
                    return "num"
                return "unknown"
            # Other arithmetic
            if isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                if lt == "num" and rt == "num":
                    return "num"
                return "unknown"
        if isinstance(node, ast.UnaryOp):
            ot = self._infer_type(node.operand)
            if isinstance(node.op, (ast.UAdd, ast.USub)) and ot == "num":
                return "num"
            return "unknown"
        if isinstance(node, ast.Subscript):
            base_t = self._infer_type(node.value)
            if base_t == "str":
                return "str"
            return "unknown"
        if isinstance(node, ast.Name):
            return "unknown"
        if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            return "unknown"
        return "unknown"

    def _disallowed(self, node: ast.AST) -> Optional[str]:
        # Explicitly flagged nodes
        for cls in self._DISALLOWED_EXPLICIT:
            if isinstance(node, cls):
                return cls.__name__
        # Any node not in our whitelist (excluding operator/ctx nodes which are included)
        if not any(isinstance(node, allowed) for allowed in self._ALLOWED_NODES):
            return type(node).__name__
        return None

    def _err(self, node: ast.AST) -> str:
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        if lineno is not None and col is not None:
            return f"{type(node).__name__} at line {lineno}, column {col}"
        return f"{type(node).__name__}"

    def generic_visit(self, node: ast.AST) -> None:
        reason = self._disallowed(node)
        if reason is not None:
            raise ValueError(f"Disallowed or potentially harmful construct: {self._err(node)}")
        super().generic_visit(node)

    # Visitors for operation collection

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Enforce allowed binary operators
        allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        if not isinstance(node.op, allowed):
            raise ValueError(f"Disallowed operator: {self._err(node.op)}")

        lt = self._infer_type(node.left)
        rt = self._infer_type(node.right)

        if isinstance(node.op, ast.Add):
            op_name = "string:concat" if ("str" in (lt, rt)) else "arith:add"
            self._add_op(op_name)
        elif isinstance(node.op, ast.Sub):
            self._add_op("arith:sub")
        elif isinstance(node.op, ast.Mult):
            # If either side is string, treat as potential repeat
            op_name = "string:repeat" if ("str" in (lt, rt)) else "arith:mul"
            self._add_op(op_name)
        elif isinstance(node.op, ast.Div):
            self._add_op("arith:div")
        elif isinstance(node.op, ast.FloorDiv):
            self._add_op("arith:floordiv")
        elif isinstance(node.op, ast.Mod):
            self._add_op("arith:mod")
        elif isinstance(node.op, ast.Pow):
            self._add_op("arith:pow")

        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.UAdd):
            self._add_op("unary:pos")
        elif isinstance(node.op, ast.USub):
            self._add_op("unary:neg")
        else:
            raise ValueError(f"Disallowed unary operator: {self._err(node.op)}")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # Classify augassign similar to BinOp
        allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        if not isinstance(node.op, allowed):
            raise ValueError(f"Disallowed operator: {self._err(node.op)}")

        target_t = self._infer_type(node.target)
        value_t = self._infer_type(node.value)

        if isinstance(node.op, ast.Add):
            op_name = "aug:concat" if ("str" in (target_t, value_t)) else "aug:add"
        elif isinstance(node.op, ast.Sub):
            op_name = "aug:sub"
        elif isinstance(node.op, ast.Mult):
            op_name = "aug:repeat" if ("str" in (target_t, value_t)) else "aug:mul"
        elif isinstance(node.op, ast.Div):
            op_name = "aug:div"
        elif isinstance(node.op, ast.FloorDiv):
            op_name = "aug:floordiv"
        elif isinstance(node.op, ast.Mod):
            op_name = "aug:mod"
        elif isinstance(node.op, ast.Pow):
            op_name = "aug:pow"
        else:
            # Should never hit due to allowed guard
            raise ValueError(f"Disallowed operator: {self._err(node.op)}")

        self._add_op(op_name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base_t = self._infer_type(node.value)
        if base_t == "str":
            # Distinguish index vs slice
            sl = node.slice
            is_slice = isinstance(sl, ast.Slice)
            # For Python 3.9-, slices may be wrapped; normalize
            if not is_slice and hasattr(ast, "Index") and isinstance(sl, ast.Index):
                sl = sl.value  # unwrap
                is_slice = isinstance(sl, ast.Slice)
            self._add_op("string:slice" if is_slice else "string:index")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._add_op("string:fstring")
        self.generic_visit(node)

    # Explicitly reject dangerous nodes with custom messages
    def visit_Call(self, node: ast.Call) -> None:
        raise ValueError(f"Function calls are not allowed: {self._err(node)}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ValueError(f"Attribute access is not allowed: {self._err(node)}")

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError(f"Imports are not allowed: {self._err(node)}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError(f"Imports are not allowed: {self._err(node)}")


def parse_script_operations(script: str) -> List[str]:
    """
    Parse a user-supplied script and return a list of operations it contains.
    Identifies:
      - Basic arithmetic operations: add, sub, mul, div, floordiv, mod, pow, unary pos/neg, and their augmented forms.
      - Basic string operations: concatenation, repetition, indexing, slicing, and f-strings.

    Security:
      - Enforces a strict AST whitelist and rejects potentially harmful constructs
        such as imports, function calls, attribute access, class/function defs, control flow, etc.

    Raises:
      ValueError: If the script contains disallowed or potentially harmful constructs.

    Returns:
      List[str]: List of unique operation identifiers in the order first seen.
    """
    if not isinstance(script, str):
        raise TypeError("script must be a string")

    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"SyntaxError while parsing script: {e}") from e

    collector = _OperationCollector()
    collector.visit(tree)
    return collector.result()


# Safe set of recognized operations as produced by parse_script_operations
_SAFE_OPERATION_SET: Set[str] = {
    # Arithmetic
    "arith:add",
    "arith:sub",
    "arith:mul",
    "arith:div",
    "arith:floordiv",
    "arith:mod",
    "arith:pow",
    # Unary numbers
    "unary:pos",
    "unary:neg",
    # Augmented
    "aug:add",
    "aug:sub",
    "aug:mul",
    "aug:div",
    "aug:floordiv",
    "aug:mod",
    "aug:pow",
    "aug:concat",
    "aug:repeat",
    # Strings
    "string:concat",
    "string:repeat",
    "string:index",
    "string:slice",
    "string:fstring",
}


def execute_operations(operations: List[str]) -> Union[Dict[str, Any], str]:
    """
    Validate and (safely) "execute" a list of operation identifiers.

    Since only operation identifiers are provided (without operands/values),
    this function:
      - Verifies all operations are in a strict safe whitelist.
      - Returns a structured summary describing the operations that would be executed.
      - If any operation is unsafe, returns an error message string.

    Args:
      operations: List of operation identifiers (e.g., "arith:add", "string:concat").

    Returns:
      - dict: On success, returns a summary with counts by category and by operation.
      - str: If any operation is unsafe, returns an error message.
    """
    if not isinstance(operations, list) or not all(isinstance(op, str) for op in operations):
        raise TypeError("operations must be a list of strings")

    # Identify unsafe operations
    unsafe = [op for op in operations if op not in _SAFE_OPERATION_SET]
    if unsafe:
        unique_unsafe = sorted(set(unsafe))
        return f"Unsafe operation(s) detected: {', '.join(unique_unsafe)}"

    # Build summaries
    counts_by_operation: Dict[str, int] = {}
    counts_by_category: Dict[str, int] = {}

    def cat_from_op(op: str) -> str:
        return op.split(":", 1)[0] if ":" in op else "unknown"

    for op in operations:
        counts_by_operation[op] = counts_by_operation.get(op, 0) + 1
        cat = cat_from_op(op)
        counts_by_category[cat] = counts_by_category.get(cat, 0) + 1

    # Return a structured "result" of executing these safe operations.
    return {
        "status": "ok",
        "executed": list(operations),
        "counts_by_operation": counts_by_operation,
        "counts_by_category": counts_by_category,
    }


class _ExecutionBlocked(Exception):
    """Internal signal for security-related blocking."""
    pass


class _SafeEvaluator(ast.NodeVisitor):
    """
    A tiny interpreter for a very small safe subset of Python:
    - Numbers (int, float) and strings
    - Binary ops: +, -, *, /, //, %, **
    - Unary ops: +, -
    - String indexing/slicing
    - F-strings
    - Assignments to simple names and augmented assignments

    Any other construct or type will raise _ExecutionBlocked.
    """

    MAX_STRING_LEN = 1_000_000
    MAX_POWER_EXP_INT = 4096
    MAX_FORMAT_SPEC_LEN = 1000

    def __init__(self) -> None:
        self.env: Dict[str, Any] = {}
        self._last_value: Any = None

    # Helpers

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _ensure_str_bounds(self, s: str) -> str:
        if len(s) > self.MAX_STRING_LEN:
            raise _ExecutionBlocked("string too long")
        return s

    # Visitor methods

    def visit_Module(self, node: ast.Module) -> Any:
        result = None
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                result = self.visit(stmt)
            else:
                self.visit(stmt)
        self._last_value = result
        return result

    def visit_Expr(self, node: ast.Expr) -> Any:
        val = self.visit(node.value)
        self._last_value = val
        return val

    def visit_Constant(self, node: ast.Constant) -> Any:
        v = node.value
        if isinstance(v, (int, float, str)):
            if isinstance(v, str):
                return self._ensure_str_bounds(v)
            return v
        raise _ExecutionBlocked(f"unsupported constant type: {type(v).__name__}")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        raise NameError(f"name '{node.id}' is not defined")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise _ExecutionBlocked("only simple name assignments are allowed")
        value = self.visit(node.value)
        if not (self._is_number(value) or isinstance(value, str)):
            raise _ExecutionBlocked("only numbers and strings can be assigned")
        self.env[node.targets[0].id] = value
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise _ExecutionBlocked("only simple name assignments are allowed")
        if node.value is None:
            # Bare annotation - do not allow creating names without values
            raise _ExecutionBlocked("bare annotations are not allowed")
        value = self.visit(node.value)
        if not (self._is_number(value) or isinstance(value, str)):
            raise _ExecutionBlocked("only numbers and strings can be assigned")
        self.env[node.target.id] = value
        return None

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise _ExecutionBlocked("augmented assignment only allowed for names")
        if node.target.id not in self.env:
            raise NameError(f"name '{node.target.id}' is not defined")
        left = self.env[node.target.id]
        right = self.visit(node.value)
        result = self._apply_binop(node.op, left, right, is_aug=True)
        self.env[node.target.id] = result
        return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        val = self.visit(node.operand)
        if not self._is_number(val):
            raise _ExecutionBlocked("unary operations allowed only on numbers")
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise _ExecutionBlocked("unsupported unary operator")

    def _apply_binop(self, op: ast.operator, left: Any, right: Any, is_aug: bool = False) -> Any:
        # String concatenation
        if isinstance(op, ast.Add):
            if isinstance(left, str) and isinstance(right, str):
                return self._ensure_str_bounds(left + right)
            # numeric addition
            if self._is_number(left) and self._is_number(right):
                return left + right
            raise _ExecutionBlocked("addition only allowed for numbers or string+string")

        if isinstance(op, ast.Sub):
            if self._is_number(left) and self._is_number(right):
                return left - right
            raise _ExecutionBlocked("subtraction only allowed for numbers")

        if isinstance(op, ast.Mult):
            # string repetition
            if isinstance(left, str) and isinstance(right, int) and not isinstance(right, bool):
                return self._ensure_str_bounds(left * right)
            if isinstance(right, str) and isinstance(left, int) and not isinstance(left, bool):
                return self._ensure_str_bounds(right * left)
            # numeric multiplication
            if self._is_number(left) and self._is_number(right):
                return left * right
            raise _ExecutionBlocked("multiplication only allowed for numbers or str*int")

        if isinstance(op, ast.Div):
            if self._is_number(left) and self._is_number(right):
                return left / right
            raise _ExecutionBlocked("division only allowed for numbers")

        if isinstance(op, ast.FloorDiv):
            if self._is_number(left) and self._is_number(right):
                return left // right
            raise _ExecutionBlocked("floor division only allowed for numbers")

        if isinstance(op, ast.Mod):
            if self._is_number(left) and self._is_number(right):
                return left % right
            raise _ExecutionBlocked("modulo only allowed for numbers")

        if isinstance(op, ast.Pow):
            if self._is_number(left) and self._is_number(right):
                # Limit large integer exponents
                if isinstance(left, int) and isinstance(right, int):
                    if abs(right) > self.MAX_POWER_EXP_INT:
                        raise _ExecutionBlocked("exponent too large")
                return left ** right
            raise _ExecutionBlocked("power only allowed for numbers")

        raise _ExecutionBlocked("unsupported binary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._apply_binop(node.op, left, right)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        if not isinstance(value, str):
            raise _ExecutionBlocked("only string indexing/slicing is allowed")
        # Normalize slice/index
        sl = node.slice
        # For older Python, ast.Index wrapper
        if hasattr(ast, "Index") and isinstance(sl, ast.Index):
            sl = sl.value
        if isinstance(sl, ast.Slice):
            lower = self.visit(sl.lower) if sl.lower is not None else None
            upper = self.visit(sl.upper) if sl.upper is not None else None
            step = self.visit(sl.step) if sl.step is not None else None
            for part in (lower, upper, step):
                if part is not None and not isinstance(part, int):
                    raise _ExecutionBlocked("slice indices must be integers")
            result = value[slice(lower, upper, step)]
            return self._ensure_str_bounds(result)
        else:
            index_val = self.visit(sl)
            if not isinstance(index_val, int):
                raise _ExecutionBlocked("string indices must be integers")
            return value[index_val]

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Str):  # Python <3.8
                s = v.s
            else:
                s = self.visit(v)
            if not isinstance(s, str):
                s = str(s)
            parts.append(s)
        return self._ensure_str_bounds("".join(parts))

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        val = self.visit(node.value)
        conv = node.conversion if hasattr(node, "conversion") else -1
        if conv == -1:
            formatted = str(val)
        elif conv == ord("s"):
            formatted = str(val)
        elif conv == ord("r"):
            formatted = repr(val)
        elif conv == ord("a"):
            formatted = ascii(val)
        else:
            raise _ExecutionBlocked("unsupported f-string conversion")
        if node.format_spec is not None:
            fmt = self.visit(node.format_spec)
            if not isinstance(fmt, str):
                fmt = str(fmt)
            if len(fmt) > self.MAX_FORMAT_SPEC_LEN:
                raise _ExecutionBlocked("format spec too long")
            formatted = format(val, fmt)
        return self._ensure_str_bounds(formatted)

    # Block unsupported literals/containers outright
    def visit_Tuple(self, node: ast.Tuple) -> Any:
        raise _ExecutionBlocked("tuple literals are not allowed")

    def visit_List(self, node: ast.List) -> Any:
        raise _ExecutionBlocked("list literals are not allowed")

    def visit_Set(self, node: ast.Set) -> Any:
        raise _ExecutionBlocked("set literals are not allowed")

    def visit_Dict(self, node: ast.Dict) -> Any:
        raise _ExecutionBlocked("dict literals are not allowed")

    # Anything not explicitly handled will use generic_visit, which should not produce values
    def generic_visit(self, node: ast.AST) -> Any:
        # Reuse the collector-like gate: if a node type is not recognized here,
        # fail closed to maintain security.
        raise _ExecutionBlocked(f"unsupported syntax: {type(node).__name__}")


def evaluate_user_code(code_str: str) -> Union[Any, str]:
    """
    Securely evaluate user-supplied code by integrating parsing and safe execution.

    - Parses the code and collects operations via parse_script_operations.
    - If any unsafe construct is detected or operations fall outside the safe set, returns "Execution Blocked!".
    - Otherwise, evaluates the AST using a tiny safe interpreter supporting only basic arithmetic and string ops.
    - Returns the last expression's value (if any), or None if no expression produced a value.
    - On non-security runtime errors (e.g., NameError, ZeroDivisionError), returns a human-readable error string.
    """
    if not isinstance(code_str, str):
        raise TypeError("code_str must be a string")

    # First pass: parse and security-check using the collector
    try:
        ops = parse_script_operations(code_str)
    except ValueError:
        return "Execution Blocked!"

    # Ensure all collected operations are within the safe set
    if any(op not in _SAFE_OPERATION_SET for op in ops):
        return "Execution Blocked!"

    # Second pass: evaluate with strict safe interpreter
    try:
        tree = ast.parse(code_str, mode="exec")
    except SyntaxError as e:
        return f"Error: {e}"

    evaluator = _SafeEvaluator()
    try:
        result = evaluator.visit(tree)
        return result
    except _ExecutionBlocked:
        return "Execution Blocked!"
    except Exception as e:
        return f"Error: {e}"
