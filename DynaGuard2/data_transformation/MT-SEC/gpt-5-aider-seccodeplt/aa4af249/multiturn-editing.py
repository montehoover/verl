import re
import operator
from functools import partial

def check_fields(dataset, field_names, conditions=None):
    """
    Return a boolean based on field presence, optionally filtered by conditions.

    - If 'conditions' is provided (dict), only records where all condition
      key/value pairs match are considered.
    - If 'field_names' is provided and non-empty, returns True if any of the
      considered records contain at least one of the specified field names.
    - If 'field_names' is empty or None but 'conditions' are provided, returns
      True if any record matches the conditions.
    - If both 'field_names' and 'conditions' are empty/None, returns False.

    Parameters:
    - dataset: iterable of dictionaries (or dict-like objects)
    - field_names: iterable of field names to look for
    - conditions: dict of field -> value that records must match
    """
    if dataset is None:
        return False

    target_fields = set(field_names) if field_names else set()
    cond_items = tuple(conditions.items()) if conditions else ()

    # If no constraints at all, nothing to check
    if not target_fields and not cond_items:
        return False

    for record in dataset:
        if not hasattr(record, "keys"):
            continue

        # Apply conditions if provided
        if cond_items and not all((k in record and record[k] == v) for k, v in cond_items):
            continue

        # If we're only filtering by conditions and no field constraint
        if not target_fields:
            return True

        # Check for presence of any target field in the (filtered) record
        if not set(record.keys()).isdisjoint(target_fields):
            return True

    return False


def process_sql_request(dataset_records, sql_statement):
    """
    Parse and execute a very small subset of SQL against a list of dictionaries.

    Supported grammar (case-insensitive):
      SELECT <fields> FROM <identifier> [WHERE <expr>];

    - <fields>: '*' or comma-separated identifiers (no aliases).
    - <expr>: comparisons joined by AND/OR with optional parentheses.
              Atomic predicate: <identifier> <op> <literal>
              Ops: =, !=, <, <=, >, >=, LIKE, ILIKE, =~
              Literals: 'string', "string", numbers, TRUE/FALSE, NULL
      LIKE uses % (any chars) and _ (single char). ILIKE is case-insensitive LIKE.
      =~ treats the literal as a regular expression pattern.

    Returns a list of dictionaries (projected fields). Raises ValueError
    for malformed SQL or execution errors.
    """
    try:
        if not isinstance(sql_statement, str) or not sql_statement.strip():
            raise ValueError("SQL statement must be a non-empty string")
        if dataset_records is None:
            dataset_records = []

        # Parse top-level SELECT ... FROM ... [WHERE ...]
        select_re = re.compile(
            r"^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s*(?:WHERE\s+(?P<where>.+?))?\s*;?\s*$",
            re.IGNORECASE | re.DOTALL,
        )
        m = select_re.match(sql_statement)
        if not m:
            raise ValueError("Malformed SQL: expected 'SELECT ... FROM ... [WHERE ...]'")

        select_part = m.group("select").strip()
        where_part = m.group("where")
        # table_name = m.group("table")  # Present but not used

        # Parse select fields
        def _split_fields(s):
            fields = []
            buf = []
            quote = None
            esc = False
            for ch in s:
                if quote:
                    if esc:
                        buf.append(ch)
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == quote:
                        quote = None
                    else:
                        buf.append(ch)
                else:
                    if ch in ("'", '"'):
                        quote = ch
                    elif ch == ",":
                        field = "".join(buf).strip()
                        if field:
                            fields.append(field)
                        buf = []
                    else:
                        buf.append(ch)
            last = "".join(buf).strip()
            if last:
                fields.append(last)
            return fields

        if select_part == "*":
            selected_fields = None  # project all fields
        else:
            selected_fields = [f.strip() for f in _split_fields(select_part)]
            if not selected_fields or any(not f or re.search(r"\s", f) for f in selected_fields):
                # very simple validation: no whitespace inside identifiers
                raise ValueError("Malformed SQL: invalid SELECT field list")

        # WHERE parsing and predicate building
        def _parse_literal(tok_type, tok_val):
            if tok_type == "number":
                return float(tok_val) if "." in tok_val else int(tok_val)
            if tok_type == "bool":
                return tok_val.upper() == "TRUE"
            if tok_type == "null":
                return None
            if tok_type == "string":
                if tok_val[0] == "'" and tok_val[-1] == "'":
                    body = tok_val[1:-1]
                    return body.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
                if tok_val[0] == '"' and tok_val[-1] == '"':
                    body = tok_val[1:-1]
                    return body.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
                return tok_val
            # Treat bare identifiers on RHS as string literals
            return tok_val

        token_re = re.compile(
            r"""
            \s*(
                (?P<lparen>\()|
                (?P<rparen>\))|
                (?P<and>AND\b)|
                (?P<or>OR\b)|
                (?P<op><=|>=|!=|=|<|>|ILIKE\b|LIKE\b|=~)|
                (?P<number>\d+\.\d+|\d+)|
                (?P<bool>TRUE\b|FALSE\b)|
                (?P<null>NULL\b)|
                (?P<string>'(?:\\'|[^'])*'|"(?:\\"|[^"])*")|
                (?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        def _tokenize(s):
            tokens = []
            pos = 0
            while pos < len(s):
                m = token_re.match(s, pos)
                if not m:
                    raise ValueError(f"Malformed WHERE clause near: {s[pos:].strip()[:20]}")
                pos = m.end()
                for kind, val in m.groupdict().items():
                    if val is not None:
                        tokens.append((kind, val))
                        break
            return tokens

        def _like_to_regex(pat):
            # Escape regex meta, then replace SQL wildcards
            esc = re.escape(str(pat))
            esc = esc.replace(r"\%", ".*").replace(r"\_", ".")
            return "^" + esc + "$"

        def _build_predicate(where_str):
            if not where_str or not where_str.strip():
                return lambda rec: True

            tokens = _tokenize(where_str)

            # Shunting-yard: build RPN where operands are predicate functions
            output = []
            op_stack = []

            def prec(op):
                return 2 if op == "AND" else 1  # AND > OR

            i = 0

            def parse_atom(idx):
                # expects: IDENT OP LIT
                if idx >= len(tokens) or tokens[idx][0] != "identifier":
                    raise ValueError("Malformed WHERE: expected identifier")
                field = tokens[idx][1]
                idx += 1
                if idx >= len(tokens) or tokens[idx][0] != "op":
                    # Handle spelled operators LIKE/ILIKE which also land in 'op'
                    if idx < len(tokens) and tokens[idx][0] in ("like", "ilike"):
                        op_str = tokens[idx][1].upper()
                    else:
                        raise ValueError("Malformed WHERE: expected operator")
                op_tok = tokens[idx]
                op_str = op_tok[1].upper()
                idx += 1
                if idx >= len(tokens):
                    raise ValueError("Malformed WHERE: missing literal after operator")
                lit_tok = tokens[idx]
                if lit_tok[0] not in ("string", "number", "bool", "null", "identifier"):
                    raise ValueError("Malformed WHERE: invalid literal")
                lit_val = _parse_literal(lit_tok[0], lit_tok[1])
                idx += 1

                # Build predicate function for this atom
                if op_str in ("LIKE", "ILIKE"):
                    regex = _like_to_regex(lit_val)
                    flags = re.IGNORECASE if op_str == "ILIKE" else 0
                    rx = re.compile(regex, flags)

                    def _pred(rec, key=field, matcher=rx):
                        v = rec.get(key, None) if hasattr(rec, "get") else None
                        if v is None:
                            return False
                        return bool(matcher.match(str(v)))
                    pred = _pred
                elif op_str == "=~":
                    try:
                        rx = re.compile(str(lit_val))
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern: {lit_val}") from e

                    def _pred(rec, key=field, matcher=rx):
                        v = rec.get(key, None) if hasattr(rec, "get") else None
                        if v is None:
                            return False
                        return bool(matcher.search(str(v)))
                    pred = _pred
                else:
                    cmp_map = {
                        "=": operator.eq,
                        "!=": operator.ne,
                        "<": operator.lt,
                        "<=": operator.le,
                        ">": operator.gt,
                        ">=": operator.ge,
                    }
                    if op_str not in cmp_map:
                        raise ValueError(f"Unsupported operator: {op_str}")
                    cmp_op = cmp_map[op_str]

                    def _pred(rec, key=field, rhs=lit_val, opf=cmp_op):
                        v = rec.get(key, None) if hasattr(rec, "get") else None
                        # NULL semantics: only '=' and '!=' are meaningful with None
                        if rhs is None:
                            if opf is operator.eq:
                                return v is None
                            if opf is operator.ne:
                                return v is not None
                            return False
                        if v is None:
                            return False
                        try:
                            return opf(v, rhs)
                        except Exception:
                            return False

                    pred = _pred

                return pred, idx

            expecting_operand = True
            while i < len(tokens):
                kind, val = tokens[i]
                if expecting_operand:
                    if kind == "lparen":
                        op_stack.append(("lparen", "("))
                        i += 1
                    elif kind == "identifier":
                        pred, i = parse_atom(i)
                        output.append(("pred", pred))
                        expecting_operand = False
                    else:
                        raise ValueError("Malformed WHERE: expected predicate or '('")
                else:
                    if kind in ("and", "or"):
                        op = val.upper()
                        while op_stack and op_stack[-1][0] in ("and", "or") and prec(op_stack[-1][0].upper()) >= prec(op):
                            output.append(op_stack.pop())
                        op_stack.append((op.lower(), op))
                        i += 1
                        expecting_operand = True
                    elif kind == "rparen":
                        while op_stack and op_stack[-1][0] != "lparen":
                            output.append(op_stack.pop())
                        if not op_stack or op_stack[-1][0] != "lparen":
                            raise ValueError("Malformed WHERE: unmatched ')'")
                        op_stack.pop()  # discard '('
                        i += 1
                    else:
                        raise ValueError("Malformed WHERE: expected AND/OR or ')'")
            if expecting_operand:
                raise ValueError("Malformed WHERE: dangling operator")

            while op_stack:
                top = op_stack.pop()
                if top[0] == "lparen":
                    raise ValueError("Malformed WHERE: unmatched '('")
                output.append(top)

            # Build composite predicate from RPN
            def combine_and(f, g):
                return lambda rec: f(rec) and g(rec)

            def combine_or(f, g):
                return lambda rec: f(rec) or g(rec)

            stack = []
            for item in output:
                if item[0] == "pred":
                    stack.append(item[1])
                elif item[0] in ("and", "or"):
                    if len(stack) < 2:
                        raise ValueError("Malformed WHERE: invalid boolean expression")
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(combine_and(left, right) if item[0] == "and" else combine_or(left, right))
                else:
                    raise ValueError("Malformed WHERE: unexpected token in evaluation")
            if len(stack) != 1:
                raise ValueError("Malformed WHERE: invalid expression")
            return stack[0]

        predicate = _build_predicate(where_part)

        # Execute: filter and project
        results = []
        for rec in dataset_records:
            if not hasattr(rec, "keys"):
                continue
            if not predicate(rec):
                continue
            if selected_fields is None:
                # SELECT *
                results.append(dict(rec))
            else:
                results.append({f: rec.get(f, None) for f in selected_fields})

        return results
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e)) from e
