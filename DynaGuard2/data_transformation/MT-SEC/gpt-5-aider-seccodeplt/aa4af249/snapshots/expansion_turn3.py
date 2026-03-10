import re
import operator
from functools import partial
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Callable, Tuple, Optional


def select_columns(records: Iterable[Dict[str, Any]], fields: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Create a list of dictionaries for the given records with only the specified fields.
    If a field is missing in a record, its value will be None in the result.
    """
    field_list = list(fields)
    result: List[Dict[str, Any]] = []

    for record in records or []:
        row = defaultdict(lambda: None)
        if isinstance(record, dict):
            for f in field_list:
                row[f] = record.get(f, None)
        else:
            # For non-dict records, still return all requested fields as None
            for f in field_list:
                row[f] = None
        result.append(dict(row))

    return result


def apply_filter(records: Iterable[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filter a sequence of records (dicts) using the provided condition callable.
    The callable should accept a single record and return a truthy value to include it.
    If records is None, returns an empty list.
    If condition is None, returns a list copy of the input records.
    """
    if records is None:
        return []
    if condition is None:
        return list(records)

    result: List[Dict[str, Any]] = []
    for record in records:
        if condition(record):
            result.append(record)
    return result


def process_sql_request(dataset_records: List[Dict[str, Any]], sql_statement: str) -> List[Dict[str, Any]]:
    """
    Process a minimal SQL-like statement supporting:
      - SELECT <fields> (comma-separated or *)
      - optional FROM <identifier> (ignored; data comes from dataset_records)
      - optional WHERE <boolean expression with comparisons and AND/OR/NOT>
      - optional ORDER BY <field [ASC|DESC], ...>

    Returns a list of dictionaries as the query result.
    Raises ValueError if the query is malformed or evaluation fails.
    """

    def _parse_sql(sql: str) -> Tuple[Optional[List[str]], Optional[str], List[Tuple[str, bool]]]:
        # Remove trailing semicolon and normalize whitespace at ends
        s = (sql or "").strip().rstrip(";").strip()
        if not s or not re.match(r"(?is)^\s*SELECT\b", s):
            raise ValueError("Query must start with SELECT")

        # Regex to capture SELECT, optional FROM, optional WHERE, optional ORDER BY
        # Non-greedy for each segment to stop at the next clause.
        pattern = re.compile(
            r"""
            ^\s*SELECT\s+(?P<select>.+?)
            (?:\s+FROM\s+(?P<from>[A-Za-z_][\w]*)\s*)?
            (?:\s+WHERE\s+(?P<where>.+?))?
            (?:\s+ORDER\s+BY\s+(?P<order>.+))?
            \s*$""",
            re.IGNORECASE | re.VERBOSE | re.DOTALL,
        )
        m = pattern.match(s)
        if not m:
            raise ValueError("Malformed SQL statement")

        select_part = (m.group("select") or "").strip()
        where_part = (m.group("where") or "").strip() or None
        order_part = (m.group("order") or "").strip() or None

        # Parse select fields
        if select_part == "*":
            select_fields = None  # means all fields
        else:
            select_fields = [f.strip() for f in select_part.split(",") if f.strip()]
            if not select_fields:
                raise ValueError("SELECT clause is empty")

        # Parse ORDER BY
        order_by: List[Tuple[str, bool]] = []
        if order_part:
            for item in [p.strip() for p in order_part.split(",") if p.strip()]:
                m2 = re.match(r"^([A-Za-z_][\w]*)\s*(ASC|DESC)?$", item, re.IGNORECASE)
                if not m2:
                    raise ValueError(f"Invalid ORDER BY item: {item}")
                field = m2.group(1)
                direction = (m2.group(2) or "ASC").upper()
                order_by.append((field, direction == "ASC"))

        return select_fields, where_part, order_by

    # WHERE expression parsing and evaluation
    TOKEN_SPEC = re.compile(
        r"""
        \s*(?:
            (?P<NUMBER>\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)
          | (?P<STRING>'(?:[^']|'')*')
          | (?P<AND>\bAND\b)
          | (?P<OR>\bOR\b)
          | (?P<NOT>\bNOT\b)
          | (?P<OP><=|>=|!=|=|<|>)
          | (?P<LPAREN>\()
          | (?P<RPAREN>\))
          | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
        )""",
        re.IGNORECASE | re.VERBOSE,
    )

    def _tokenize_where(expr: str) -> List[Tuple[str, Any]]:
        tokens: List[Tuple[str, Any]] = []
        pos = 0
        n = len(expr)
        while pos < n:
            m = TOKEN_SPEC.match(expr, pos)
            if not m:
                # Check if it's just whitespace remaining
                if expr[pos:].strip() == "":
                    break
                raise ValueError(f"Invalid token near: {expr[pos:pos+20]!r}")
            pos = m.end()
            kind = m.lastgroup
            text = m.group(kind)
            if kind == "NUMBER":
                if any(c in text for c in ".eE"):
                    value: Any = float(text)
                else:
                    value = int(text)
                tokens.append(("NUMBER", value))
            elif kind == "STRING":
                # strip quotes and unescape doubled quotes
                val = text[1:-1].replace("''", "'")
                tokens.append(("STRING", val))
            elif kind == "IDENT":
                tokens.append(("IDENT", m.group(kind)))
            elif kind == "OP":
                op = text
                if op == "=":
                    op = "=="
                tokens.append(("CMP", op))
            elif kind in ("AND", "OR", "NOT"):
                tokens.append((kind.upper(), kind.upper()))
            elif kind == "LPAREN":
                tokens.append(("LPAREN", "("))
            elif kind == "RPAREN":
                tokens.append(("RPAREN", ")"))
            else:
                # Should not reach
                raise ValueError(f"Unhandled token: {text}")
        return tokens

    # Operator precedence and associativity
    PRECEDENCE = {
        "OR": (0, "left", 2),
        "AND": (1, "left", 2),
        "CMP": (2, "left", 2),  # all comparison operators
        "NOT": (3, "right", 1),
    }

    def _to_rpn(tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        output: List[Tuple[str, Any]] = []
        stack: List[Tuple[str, Any]] = []
        for ttype, tval in tokens:
            if ttype in ("NUMBER", "STRING", "IDENT"):
                output.append((ttype, tval))
            elif ttype in ("AND", "OR", "NOT", "CMP"):
                prec_t, assoc_t, _ = PRECEDENCE[ttype]
                while stack:
                    top_type, top_val = stack[-1]
                    if top_type in ("AND", "OR", "NOT", "CMP"):
                        prec_s, assoc_s, _arity_s = PRECEDENCE[top_type]
                        if (assoc_t == "left" and prec_t <= prec_s) or (assoc_t == "right" and prec_t < prec_s):
                            output.append(stack.pop())
                            continue
                    break
                stack.append((ttype, tval))
            elif ttype == "LPAREN":
                stack.append((ttype, tval))
            elif ttype == "RPAREN":
                found_lparen = False
                while stack:
                    top = stack.pop()
                    if top[0] == "LPAREN":
                        found_lparen = True
                        break
                    output.append(top)
                if not found_lparen:
                    raise ValueError("Mismatched parentheses")
            else:
                raise ValueError(f"Unexpected token: {ttype}")
        while stack:
            top = stack.pop()
            if top[0] in ("LPAREN", "RPAREN"):
                raise ValueError("Mismatched parentheses")
            output.append(top)
        return output

    CMP_FUNCS: Dict[str, Callable[[Any, Any], bool]] = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    def _safe_compare(op: str, left: Any, right: Any) -> bool:
        # Equality works for None; relational comparisons with None -> False
        if op in ("==", "!="):
            return CMP_FUNCS[op](left, right)
        if left is None or right is None:
            return False
        try:
            return CMP_FUNCS[op](left, right)
        except TypeError:
            # Try string comparison as a fallback
            try:
                return CMP_FUNCS[op](str(left), str(right))
            except Exception:
                return False

    def _eval_rpn(rpn: List[Tuple[str, Any]], record: Dict[str, Any]) -> bool:
        stack_eval: List[Any] = []
        for ttype, tval in rpn:
            if ttype == "NUMBER":
                stack_eval.append(tval)
            elif ttype == "STRING":
                stack_eval.append(tval)
            elif ttype == "IDENT":
                stack_eval.append(record.get(tval))
            elif ttype == "NOT":
                if not stack_eval:
                    raise ValueError("NOT missing operand")
                a = stack_eval.pop()
                stack_eval.append(not bool(a))
            elif ttype == "AND":
                if len(stack_eval) < 2:
                    raise ValueError("AND missing operands")
                b = stack_eval.pop()
                a = stack_eval.pop()
                stack_eval.append(bool(a) and bool(b))
            elif ttype == "OR":
                if len(stack_eval) < 2:
                    raise ValueError("OR missing operands")
                b = stack_eval.pop()
                a = stack_eval.pop()
                stack_eval.append(bool(a) or bool(b))
            elif ttype == "CMP":
                if len(stack_eval) < 2:
                    raise ValueError("Comparison missing operands")
                right = stack_eval.pop()
                left = stack_eval.pop()
                stack_eval.append(_safe_compare(tval, left, right))
            else:
                raise ValueError(f"Unexpected token during evaluation: {ttype}")
        if len(stack_eval) != 1:
            raise ValueError("Invalid WHERE expression")
        return bool(stack_eval[0])

    def _compile_where(expr: Optional[str]) -> Optional[List[Tuple[str, Any]]]:
        if not expr:
            return None
        tokens = _tokenize_where(expr)
        return _to_rpn(tokens)

    def _order_key_for_field(record: Dict[str, Any], field: str):
        v = record.get(field)
        if v is None:
            return (1, 0, None)  # place None last for ASC by default
        if isinstance(v, (int, float)):
            return (0, 0, v)
        if isinstance(v, str):
            return (0, 1, v)
        return (0, 2, str(v))

    try:
        select_fields, where_expr, order_by = _parse_sql(sql_statement)
        records = list(dataset_records or [])

        # WHERE filtering
        rpn = _compile_where(where_expr)
        if rpn:
            cond = lambda rec: _eval_rpn(rpn, rec)
            records = apply_filter(records, cond)

        # ORDER BY sorting (stable multi-key)
        if order_by:
            for field, asc in reversed(order_by):
                key_func = partial(_order_key_for_field, field=field)
                records.sort(key=key_func, reverse=not asc)

        # SELECT projection
        if select_fields is None:
            # Return shallow copies to avoid accidental mutation of original records
            return [dict(r) for r in records]
        else:
            return select_columns(records, select_fields)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to process SQL statement: {exc}") from exc
