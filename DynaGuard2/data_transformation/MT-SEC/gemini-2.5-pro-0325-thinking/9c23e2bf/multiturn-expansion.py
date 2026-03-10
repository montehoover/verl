import re
import operator
from functools import partial
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable

# Helper for type conversion in WHERE clause
def _convert_value(value_str: str) -> Any:
    """Tries to convert a string value to int, float, or returns it as string."""
    value_str = value_str.strip().strip("'").strip('"') # Remove quotes
    if value_str.lower() == 'none':
        return None
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str

# Operator mapping for WHERE clause
OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    'CONTAINS': lambda a, b: b is not None and a is not None and b.lower() in a.lower(), # Case-insensitive contains
    'NOT CONTAINS': lambda a, b: b is not None and a is not None and b.lower() not in a.lower(),
    'IS NULL': lambda a, b: a is None,
    'IS NOT NULL': lambda a, b: a is not None,
}

def select_fields(data_records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Optional[Any]]]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        data_records: A list of dictionaries, where keys are strings and values can be of any type.
        fields: A list of field names (strings) to select.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields. If a field is not present in an original record,
        it will be included in the corresponding new record with a value of None.
    """
    selected_data: List[Dict[str, Optional[Any]]] = []
    for record in data_records:
        new_record: Dict[str, Optional[Any]] = {}
        for field in fields:
            new_record[field] = record.get(field, None)  # Handles missing fields gracefully with None
        selected_data.append(new_record)
    return selected_data

def filter_data(data_records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filters a list of records based on a given condition.

    Args:
        data_records: A list of dictionaries, where keys are strings and values can be of any type.
        condition: A callable that takes a record (dictionary) as input and
                   returns True if the record satisfies the condition, False otherwise.

    Returns:
        A list of dictionaries containing only the records that satisfy the condition.
    """
    return [record for record in data_records if condition(record)]

def _parse_where_clause(where_clause: str) -> Callable[[Dict[str, Any]], bool]:
    """Parses a WHERE clause string and returns a filter function."""
    if not where_clause:
        return lambda record: True # No filter if clause is empty

    # Normalize spaces around operators for easier splitting
    # Handle 'IS NOT NULL' and 'IS NULL' first as they contain spaces
    where_clause = where_clause.replace(" IS NOT NULL", " IS_NOT_NULL ")
    where_clause = where_clause.replace(" IS NULL", " IS_NULL ")
    # Handle 'NOT CONTAINS'
    where_clause = where_clause.replace(" NOT CONTAINS ", " NOT_CONTAINS ")


    conditions_str = re.split(r'\s+(AND|OR)\s+', where_clause, flags=re.IGNORECASE)
    
    parsed_conditions = []
    logical_ops = []

    i = 0
    while i < len(conditions_str):
        condition_part = conditions_str[i].strip()
        
        # Attempt to match field, operator, value
        # Regex to capture: field, operator (including multi-word), value (optional for IS NULL/IS NOT NULL)
        # Supports quoted and unquoted values.
        match = re.match(r"(\w+)\s*(!=|>=|<=|=|>|<|CONTAINS|NOT_CONTAINS|IS_NOT_NULL|IS_NULL)\s*(.*)", condition_part, flags=re.IGNORECASE)

        if not match:
            raise ValueError(f"Malformed condition: {condition_part}")

        field, op_str, value_str = match.groups()
        op_str = op_str.upper().replace("_", " ") # Normalize operator back

        if op_str not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op_str}")

        op_func = OPERATORS[op_str]
        
        # For IS NULL and IS NOT NULL, value_str is not used by the lambda, but we need to handle it
        if op_str in ("IS NULL", "IS NOT NULL"):
            # The lambda for these operators doesn't use the 'val' argument, so it can be anything (e.g. None)
            parsed_conditions.append(lambda record, f=field, op=op_func: op(record.get(f), None))
        else:
            if value_str is None or value_str.strip() == "":
                 raise ValueError(f"Value missing for operator {op_str} in condition: {condition_part}")
            value = _convert_value(value_str)
            parsed_conditions.append(lambda record, f=field, op=op_func, val=value: op(record.get(f), val))
        
        i += 1
        if i < len(conditions_str):
            logical_ops.append(conditions_str[i].upper())
            i += 1

    if not parsed_conditions:
        return lambda record: True

    def combined_condition(record: Dict[str, Any]) -> bool:
        result = parsed_conditions[0](record)
        for j, log_op_str in enumerate(logical_ops):
            next_condition_result = parsed_conditions[j+1](record)
            if log_op_str == "AND":
                result = result and next_condition_result
            elif log_op_str == "OR":
                result = result or next_condition_result
            else:
                raise ValueError(f"Unsupported logical operator: {log_op_str}")
        return result
    return combined_condition


def run_custom_query(dataset: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Processes SQL-like queries (SELECT, WHERE, ORDER BY) on a dataset.

    Args:
        dataset: A list of dictionaries representing the data.
        query: A string with the SQL-like statement.
               Example: "SELECT name, age WHERE age > 30 ORDER BY name ASC"

    Returns:
        A list of dictionaries with the query results.

    Raises:
        ValueError: If the query is malformed or fails.
    """
    query = query.strip()
    processed_data = list(dataset) # Work on a copy

    # Parse ORDER BY clause first if present, as it might be at the end
    order_by_match = re.search(r'\s+ORDER BY\s+(.+)', query, re.IGNORECASE)
    order_by_clause = None
    if order_by_match:
        order_by_clause = order_by_match.group(1).strip()
        query = query[:order_by_match.start()] # Remove ORDER BY from query for further parsing

    # Parse WHERE clause
    where_match = re.search(r'\s+WHERE\s+(.+)', query, re.IGNORECASE)
    where_clause_str = None
    if where_match:
        where_clause_str = where_match.group(1).strip()
        query = query[:where_match.start()] # Remove WHERE from query

    # Parse SELECT clause (must be present)
    select_match = re.match(r'SELECT\s+(.+)', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Malformed query: SELECT clause missing or incorrect.")
    
    select_fields_str = select_match.group(1).strip()
    if not select_fields_str:
        raise ValueError("Malformed query: No fields specified in SELECT clause.")

    # --- Step 1: Apply WHERE clause ---
    if where_clause_str:
        try:
            filter_condition = _parse_where_clause(where_clause_str)
            processed_data = filter_data(processed_data, filter_condition)
        except ValueError as e:
            raise ValueError(f"Error in WHERE clause: {e}")


    # --- Step 2: Apply SELECT clause ---
    fields_to_select = [f.strip() for f in select_fields_str.split(',')]
    if "*" in fields_to_select:
        if len(fields_to_select) > 1:
            raise ValueError("Cannot select '*' with other fields.")
        # If '*' is selected, and there are records, use keys from the first record.
        # If no records after filtering, or original dataset empty, result will be empty list of dicts.
        if processed_data:
            all_fields = list(processed_data[0].keys())
            processed_data = select_fields(processed_data, all_fields)
        # else: processed_data remains empty or list of empty dicts if original was like that
    else:
        processed_data = select_fields(processed_data, fields_to_select)

    # --- Step 3: Apply ORDER BY clause ---
    if order_by_clause:
        order_instructions = []
        parts = [part.strip() for part in order_by_clause.split(',')]
        for part in parts:
            field_name = part
            descending = False
            if part.lower().endswith(" desc"):
                field_name = part[:-5].strip()
                descending = True
            elif part.lower().endswith(" asc"):
                field_name = part[:-4].strip()
            
            if not field_name:
                raise ValueError("Malformed ORDER BY clause: empty field name.")
            order_instructions.append({'field': field_name, 'desc': descending})

        if not order_instructions:
            raise ValueError("Malformed ORDER BY clause: no fields specified.")

        # Multi-level sort: functools.cmp_to_key can be used for complex sorting
        # For simpler multi-key sort, Python's sort is stable, so we can sort by each key in reverse order of significance.
        # However, a lambda with a tuple of sort keys is more direct.
        try:
            processed_data.sort(key=lambda record: tuple(
                (record.get(instr['field']) is None, # Sort None values last or first consistently
                 record.get(instr['field'])) # Actual value
                for instr in order_instructions
            ), reverse=any(instr['desc'] for instr in order_instructions if len(order_instructions) == 1 and instr['desc']) ) # Simplistic reverse for single key, complex for multi

            # For proper multi-key sorting with mixed ASC/DESC, a custom cmp function or multiple sorts are needed.
            # Let's refine the sort for multi-key with mixed directions.
            # We sort from the last key to the first key (stability).
            for instr in reversed(order_instructions):
                processed_data.sort(key=lambda record: (record.get(instr['field']) is None, record.get(instr['field'])), reverse=instr['desc'])

        except TypeError: # Handles sorting incomparable types e.g. int and str
            raise ValueError("Error in ORDER BY clause: Cannot sort due to incompatible data types for a specified field across records, or field does not exist.")
        except KeyError as e:
            raise ValueError(f"Error in ORDER BY clause: Field '{e}' not found in all records.")


    return processed_data
