from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial # Included as requested

def select_fields(records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of records.

    Args:
        records: A list of dictionaries.
        fields: A list of field names (strings) to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a field is not
        present in a record, it is omitted from the result for that record.
    """
    selected_records = []
    for record in records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        selected_records.append(new_record)
    return selected_records

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries.
        condition: A callable that takes a record (dictionary) and
                   returns True if the record satisfies the condition,
                   False otherwise.

    Returns:
        A list of dictionaries that satisfy the condition.
    """
    filtered_records = []
    for record in records:
        if condition(record):
            filtered_records.append(record)
    return filtered_records

def run_sql_query(records: List[Dict], command: str) -> List[Dict]:
    """
    Processes SQL-like queries (SELECT, WHERE, ORDER BY) on a list of records.

    Args:
        records: A list of dictionaries.
        command: A string with the SQL-like statement.

    Returns:
        A list of dictionaries with the query results.

    Raises:
        ValueError: If the query is malformed or fails during execution.
    """

    def _parse_condition_str(condition_str: str) -> Callable[[Dict], bool]:
        # Helper to parse WHERE clause conditions like "field op value"
        # Handles: field op 'string', field op "string", field op number, field IS NULL, field IS NOT NULL
        # Ops: =, !=, >, <, >=, <=, CONTAINS

        # IS [NOT] NULL must be checked first due to its specific syntax
        m_is_null = re.match(r"(\w+)\s+(IS\s+NOT\s+NULL|IS\s+NULL)\s*$", condition_str, re.IGNORECASE)
        if m_is_null:
            field = m_is_null.group(1)
            op_str_is_null = m_is_null.group(2).upper()
            if op_str_is_null == "IS NULL":
                return lambda r: r.get(field) is None
            else:  # IS NOT NULL
                return lambda r: r.get(field) is not None

        # field op 'string_value' or field op "string_value"
        m_str_val = re.match(r"(\w+)\s*([<>=!]+|CONTAINS)\s*(['\"])(.*?)\3\s*$", condition_str, re.IGNORECASE)
        if m_str_val:
            field, op_str, _, value_str = m_str_val.groups()
            value = value_str  # Value is a string
        else:
            # field op numeric_value
            # CONTAINS is typically not used with numbers in this simple context
            m_num_val = re.match(r"(\w+)\s*([<>=!]+)\s*(\d+\.?\d*)\s*$", condition_str, re.IGNORECASE)
            if m_num_val:
                field, op_str, value_str = m_num_val.groups()
                try:
                    value = float(value_str) if '.' in value_str else int(value_str)
                except ValueError:
                    raise ValueError(f"Invalid numeric value in WHERE condition: {value_str}")
            else:
                raise ValueError(f"Malformed WHERE condition: '{condition_str}'. Expected format: field op value (e.g., name = 'test', age > 30, id IS NULL).")

        # Normalize operator (e.g., '=' to '==') for the operator map
        normalized_op_str = op_str.upper()
        if normalized_op_str == '=':
            normalized_op_str = '=='
        
        op_map = {
            '==': operator.eq, '!=': operator.ne,
            '>': operator.gt, '<': operator.lt,
            '>=': operator.ge, '<=': operator.le,
        }

        if normalized_op_str == 'CONTAINS':
            if not isinstance(value, str):
                raise ValueError("CONTAINS operator requires a string value for comparison.")
            return lambda r: field in r and isinstance(r.get(field), str) and value in r[field]
        
        if normalized_op_str not in op_map:
            raise ValueError(f"Unsupported operator in WHERE condition: {op_str}")
        
        op_func = op_map[normalized_op_str]

        def condition_checker(record: Dict) -> bool:
            if field not in record or record.get(field) is None:
                # SQL behavior: comparisons with NULL (other than IS NULL/IS NOT NULL) are false/unknown.
                return False
            
            record_value = record[field]
            
            # Type checking for comparison:
            # Allow numeric comparison (int/float with int/float)
            # Allow string comparison (str with str)
            # Otherwise, types are considered incompatible for this simple checker.
            if isinstance(value, (int, float)) and isinstance(record_value, (int, float)):
                return op_func(record_value, value)
            elif isinstance(value, str) and isinstance(record_value, str):
                return op_func(record_value, value)
            else:
                # Mismatched types (e.g., comparing string '10' with number 5) are false.
                return False
        return condition_checker

    # Main regex to parse SELECT, optional WHERE, optional ORDER BY
    query_pattern = re.compile(
        r"SELECT\s+(?P<select_fields>.+?)"
        r"(?:\s+WHERE\s+(?P<where_clause>.+?))?"
        r"(?:\s+ORDER BY\s+(?P<orderby_clause>.+?))?$",
        re.IGNORECASE
    )

    match = query_pattern.match(command.strip())
    if not match:
        raise ValueError("Malformed SQL query. Expected: SELECT fields [WHERE condition] [ORDER BY field [ASC|DESC]]")

    parts = match.groupdict()
    select_fields_str = parts["select_fields"].strip()
    where_clause_str = parts["where_clause"].strip() if parts["where_clause"] else None
    orderby_clause_str = parts["orderby_clause"].strip() if parts["orderby_clause"] else None
    
    # If WHERE clause is present, but it also captured ORDER BY (due to regex structure)
    # we need to separate them. This happens if orderby_clause is None but where_clause contains "ORDER BY"
    if where_clause_str and orderby_clause_str is None and "ORDER BY" in where_clause_str.upper():
        orderby_keyword_index = where_clause_str.upper().rfind("ORDER BY")
        # Check if "ORDER BY" is not within quotes
        # This is a simplification; proper parsing would be more robust.
        # For now, assume "ORDER BY" is a clause separator if not obviously part of a string literal.
        # A quick check: count quotes before. If even, it's likely a separator.
        # This is still fragile. The main regex should ideally handle this better.
        # The current regex `.+?` is non-greedy, so it should stop before the next optional group.
        # If `orderby_clause` is None, it means `ORDER BY` was not matched as a separate clause.
        # So, if `where_clause_str` contains `ORDER BY`, it must be part of the where condition itself,
        # or the query is malformed in a way the regex didn't catch (e.g. `SELECT * WHERE x ORDER BY y = 10`)
        # For now, trust the regex groups. If `ORDER BY` is in `where_clause_str` when `orderby_clause` is None,
        # it's treated as part of the WHERE condition, which `_parse_condition_str` will likely reject.

    current_records = list(records)  # Work on a copy

    # --- WHERE clause processing ---
    if where_clause_str:
        try:
            # If WHERE clause itself contains "ORDER BY" and orderby_clause is also set,
            # it implies the where_clause was not terminated correctly by the regex.
            # Example: SELECT * WHERE name = 'test' ORDER BY id ORDER BY date
            # This is unlikely with the current regex structure.
            if orderby_clause_str and "ORDER BY" in where_clause_str[len(where_clause_str) - len(orderby_clause_str) - 10:].upper(): # Heuristic
                 # This suggests `where_clause` might have incorrectly captured an `ORDER BY` meant for the `orderby_clause`.
                 # This situation should ideally be handled by a more sophisticated parser or a clearer regex.
                 # For now, we assume the regex correctly separates clauses.
                 pass

            filter_func = _parse_condition_str(where_clause_str)
            current_records = filter_data(current_records, filter_func)
        except ValueError as e:
            # Re-raise with more context if needed, or let it propagate.
            raise ValueError(f"Error processing WHERE clause '{where_clause_str}': {e}")

    # --- ORDER BY clause processing ---
    if orderby_clause_str:
        m_orderby = re.match(r"(\w+)(?:\s+(ASC|DESC))?\s*$", orderby_clause_str, re.IGNORECASE)
        if not m_orderby:
            raise ValueError(f"Malformed ORDER BY clause: '{orderby_clause_str}'. Expected: field [ASC|DESC]")
        
        orderby_field = m_orderby.group(1)
        direction_str = m_orderby.group(2)
        sort_desc = bool(direction_str and direction_str.upper() == "DESC")

        try:
            # Sort key: (is_None, value) ensures Nones are grouped and sorted consistently.
            # For ASC (sort_desc=False): Nones last. (False, val) < (True, None)
            # For DESC (sort_desc=True): Nones first. Reversed: (True, None) < (False, val)
            current_records.sort(
                key=lambda r: (r.get(orderby_field) is None, r.get(orderby_field)),
                reverse=sort_desc
            )
        except TypeError:
            # Occurs if trying to compare incompatible types (e.g., int and str) for sorting.
            raise ValueError(f"Incompatible data types for sorting on field '{orderby_field}'.")

    # --- SELECT fields processing (projection) ---
    if select_fields_str == "*":
        final_records = current_records  # Return all fields of the processed records
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',') if f.strip()]
        if not fields_to_select: # Handles cases like "SELECT ," or "SELECT  "
             raise ValueError("No fields specified in SELECT clause or invalid field list.")
        if any(not f.isidentifier() for f in fields_to_select): # Basic check for valid field names
            raise ValueError(f"Invalid field name in SELECT clause: {select_fields_str}")
        final_records = select_fields(current_records, fields_to_select)
        
    return final_records
