import re
import operator
from functools import partial

# Operator mapping for WHERE clause
_SQL_OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
}

def _parse_query_value(value_str):
    """
    Parses a value string from a query into a Python literal (int, float, or str).
    Handles quoted strings and unquoted numbers/strings.
    """
    stripped_value = value_str.strip()
    # Check for quoted strings
    if (stripped_value.startswith("'") and stripped_value.endswith("'")) or \
       (stripped_value.startswith('"') and stripped_value.endswith('"')):
        return stripped_value[1:-1]
    
    # Try to parse as integer
    try:
        return int(stripped_value)
    except ValueError:
        pass  # Not an integer

    # Try to parse as float
    try:
        return float(stripped_value)
    except ValueError:
        pass  # Not a float

    # Default to string if not a recognized number and not quoted
    return stripped_value


def run_sql_query(records, command):
    """
    Executes a basic SQL-like statement on a list of dictionaries.

    Args:
        records: A list of dictionaries representing data records.
        command: A string containing the SQL-like statement.
                 Example: "SELECT name, age FROM data WHERE age > 25 ORDER BY age"

    Returns:
        A list of dictionaries, results of the query operation.

    Raises:
        ValueError: If there's an issue with the query format or processing.
    """

    # Main regex to parse the SQL command: SELECT columns FROM data [WHERE condition] [ORDER BY sort_key]
    main_query_pattern = re.compile(
        r"SELECT\s+(.+?)\s+FROM\s+data"
        r"(?:\s+WHERE\s+(.+?))?"
        r"(?:\s+ORDER BY\s+(.+?))?$",
        re.IGNORECASE
    )
    match = main_query_pattern.match(command.strip())

    if not match:
        raise ValueError("Invalid SQL query format. Must start with SELECT ... FROM data ...")

    select_clause_str, where_clause_str, order_by_clause_str = match.groups()

    # Start with a copy of the original records
    processed_records = list(records)

    # 1. Process WHERE clause (filter records)
    if where_clause_str:
        # Simple WHERE condition: field op value (e.g., "age > 25")
        where_pattern = re.compile(r"(\w+)\s*([><!=]=?)\s*(.+)", re.IGNORECASE)
        where_match = where_pattern.match(where_clause_str.strip())
        
        if not where_match:
            raise ValueError(f"Invalid WHERE clause format: '{where_clause_str}'")

        field, op_symbol, value_str = where_match.groups()
        op_symbol = op_symbol.strip() 

        if op_symbol not in _SQL_OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: '{op_symbol}'")

        op_func = _SQL_OPERATORS[op_symbol]
        query_value = _parse_query_value(value_str)
        
        filtered_records = []
        for record in processed_records:
            if field not in record:
                raise ValueError(f"Field '{field}' specified in WHERE clause not found in record: {record}")
            
            record_value = record[field]
            
            try:
                if op_func(record_value, query_value):
                    filtered_records.append(record)
            except TypeError:
                raise ValueError(
                    f"Type mismatch in WHERE clause for field '{field}'. "
                    f"Cannot compare record value '{record_value}' (type {type(record_value).__name__}) "
                    f"with query value '{query_value}' (type {type(query_value).__name__})."
                )
        processed_records = filtered_records

    # 2. Process ORDER BY clause (sort records)
    if order_by_clause_str:
        order_by_pattern = re.compile(r"(\w+)(?:\s+(ASC|DESC))?", re.IGNORECASE)
        order_by_match = order_by_pattern.match(order_by_clause_str.strip())

        if not order_by_match:
            raise ValueError(f"Invalid ORDER BY clause format: '{order_by_clause_str}'")

        sort_key_name = order_by_match.group(1)
        sort_direction = order_by_match.group(2)

        is_reverse = False
        if sort_direction and sort_direction.upper() == "DESC":
            is_reverse = True

        if processed_records and sort_key_name not in processed_records[0]:
            raise ValueError(f"Sort key '{sort_key_name}' not found in records.")
        
        try:
            processed_records.sort(key=lambda r: r[sort_key_name], reverse=is_reverse)
        except KeyError: 
             raise ValueError(f"Sort key '{sort_key_name}' missing in some records.")
        except TypeError:
            raise ValueError(f"Cannot sort by '{sort_key_name}' due to incompatible data types in that column.")

    # 3. Process SELECT clause (project columns)
    if select_clause_str.strip() == "*":
        if not processed_records: 
            return []
        selected_column_names = list(processed_records[0].keys())
    else:
        selected_column_names = [col.strip() for col in select_clause_str.split(',')]
        if not all(isinstance(col, str) and col for col in selected_column_names):
             raise ValueError("Invalid column names in SELECT clause. Ensure names are non-empty strings.")

    final_result = []
    for record in processed_records:
        selected_record = {}
        for col_name in selected_column_names:
            if col_name not in record:
                raise ValueError(f"Selected column '{col_name}' not found in record: {record}")
            selected_record[col_name] = record[col_name]
        final_result.append(selected_record)

    return final_result
