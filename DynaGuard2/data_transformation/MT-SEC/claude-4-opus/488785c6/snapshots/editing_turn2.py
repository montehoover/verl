def add_header(header_value):
    if all(c.isalnum() or c.isspace() for c in header_value):
        return f'Custom-Header: {header_value}'
    else:
        return 'Error: Header value contains invalid characters'
