import re
import operator
from functools import partial
from collections import defaultdict
from typing import List, Dict, Any, Callable

def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        records: A list of dictionaries (representing records).
        fields: A list of field names (strings) to select.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a field is not
        present in a record, it is omitted from the new dictionary.
    """
    result_records: List[Dict[str, Any]] = []
    for record in records:
        selected_record: Dict[str, Any] = {}
        for field_name in fields:
            if field_name in record:
                selected_record[field_name] = record[field_name]
        result_records.append(selected_record)
    return result_records


def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries (representing records).
        condition: A callable that takes a record (dictionary) and
                   returns True if the record satisfies the condition,
                   False otherwise.

    Returns:
        A list of records that satisfy the condition.
    """
    filtered_records: List[Dict[str, Any]] = []
    for record in records:
        if condition(record):
            filtered_records.append(record)
    return filtered_records


def _parse_condition(condition_str: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Parses a condition string (e.g., "age > 30", "name == 'Alice'")
    and returns a callable that evaluates the condition for a record.
    """
    match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", condition_str.strip())
    if not match:
        raise ValueError(f"Invalid condition string: {condition_str}")

    field, op_str, value_str = match.groups()

    # Attempt to convert value to int, float, or keep as string
    try:
        value = int(value_str)
    except ValueError:
        try:
            value = float(value_str)
        except ValueError:
            # Handle string literals (e.g., 'Alice' or "Alice")
            if (value_str.startswith("'") and value_str.endswith("'")) or \
               (value_str.startswith('"') and value_str.endswith('"')):
                value = value_str[1:-1]
            else: # Treat as a field name if not a literal
                 # This part is tricky, as it could be a field name or an unquoted string.
                 # For simplicity, we'll assume unquoted non-numeric values are errors for now,
                 # or require explicit handling if they are meant to be field comparisons.
                 # For now, let's assume it's a literal string if not number.
                 # A more robust solution would differentiate field-to-field comparison.
                raise ValueError(f"Unsupported value format in condition: {value_str}. String literals must be quoted.")


    op_map = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }

    if op_str not in op_map:
        raise ValueError(f"Unsupported operator: {op_str}")

    op_func = op_map[op_str]

    def condition_checker(record: Dict[str, Any]) -> bool:
        if field not in record:
            return False  # Field not present, condition cannot be met
        record_value = record[field]
        # Ensure types are comparable, especially for numeric comparisons
        if isinstance(value, (int, float)) and not isinstance(record_value, (int, float)):
            try:
                record_value = type(value)(record_value) # Try to cast record_value to value's type
            except (ValueError, TypeError):
                return False # Cannot compare
        return op_func(record_value, value)

    return condition_checker


def execute_custom_query(data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Executes a SQL-like query with SELECT, WHERE, and ORDER BY capabilities.

    Args:
        data: A list of dictionaries representing the dataset.
        query: A string representing the SQL-like query.
               Example: "SELECT name, age FROM data_source WHERE age > 30 ORDER BY name ASC"
                        "SELECT * FROM data_source WHERE city == 'New York'"

    Returns:
        A list of dictionaries with the query results.

    Raises:
        ValueError: If the query is invalid or cannot be executed.
    """
    query = query.strip()
    processed_data = list(data) # Work on a copy

    # Parse SELECT clause
    select_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: Missing SELECT clause or FROM keyword.")
    select_fields_str = select_match.group(1).strip()
    
    # Remove SELECT part for further parsing
    query_remainder = query[select_match.end():].strip()
    # We expect "data_source" or similar after FROM, but we don't use it as data is passed in.
    # We'll just find the start of WHERE or ORDER BY.

    # Parse WHERE clause (optional)
    where_match = re.search(r"WHERE\s+(.+?)(?:\s+ORDER BY|$)", query_remainder, re.IGNORECASE)
    if where_match:
        condition_str = where_match.group(1).strip()
        # Basic support for multiple conditions with AND (more complex logic like OR or parentheses needs more advanced parsing)
        conditions = [c.strip() for c in re.split(r'\s+AND\s+', condition_str, flags=re.IGNORECASE)]
        
        for cond_str_part in conditions:
            try:
                condition_func = _parse_condition(cond_str_part)
                processed_data = filter_data(processed_data, condition_func)
            except ValueError as e:
                raise ValueError(f"Error parsing WHERE clause '{cond_str_part}': {e}")
        
        # Remove WHERE part for further parsing
        # Find where the WHERE clause ended to correctly locate ORDER BY
        # This regex finds the end of the matched WHERE part
        # We need to adjust query_remainder based on where WHERE clause ends
        # A simpler way is to look for ORDER BY in the original query_remainder
        # and if WHERE was present, ensure ORDER BY comes after it.

    # Parse ORDER BY clause (optional)
    orderby_match = re.search(r"ORDER BY\s+(.+?)(?:\s+(ASC|DESC))?$", query, re.IGNORECASE) # Search in original query for simplicity
    if where_match and orderby_match: # Ensure ORDER BY is after WHERE if both exist
        if query.upper().rfind("ORDER BY") < query.upper().rfind("WHERE"):
             # This check is a bit naive, better parsing would handle clause order more robustly
            orderby_match = None # Ignore ORDER BY if it appears before WHERE

    if orderby_match:
        orderby_field = orderby_match.group(1).strip()
        direction = orderby_match.group(2)
        descending = direction and direction.upper() == "DESC"

        # Check if orderby_field is valid (exists in at least one record or is a selected field)
        # This is a simple check; a more robust one would use schema or selected fields.
        if not any(orderby_field in record for record in processed_data):
             # If select_fields_str is not '*', check if orderby_field is in selected fields
            if select_fields_str != "*":
                selected_field_names = [f.strip() for f in select_fields_str.split(',')]
                if orderby_field not in selected_field_names:
                     raise ValueError(f"ORDER BY field '{orderby_field}' not found in data or selected fields.")
            # else if select_fields_str is '*', we assume it might exist.
            # A better check would be to see if it exists in the original data if selected_fields_str is '*'

        try:
            processed_data.sort(key=lambda x: (
                x.get(orderby_field) is None,  # Put None values last (or first if descending and None handling is reversed)
                x.get(orderby_field)
            ), reverse=descending)
        except TypeError:
            # This can happen if trying to sort mixed types (e.g. int and str)
            # A more robust sort would handle this, e.g. by converting to string or having specific type logic
            raise ValueError(f"Cannot sort by field '{orderby_field}' due to incompatible data types in records.")


    # Apply SELECT fields
    if select_fields_str == "*":
        # '*' means all fields from the (potentially filtered and sorted) records
        # No actual field selection needed if all fields are kept
        pass
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',')]
        processed_data = select_fields(processed_data, fields_to_select)

    return processed_data
