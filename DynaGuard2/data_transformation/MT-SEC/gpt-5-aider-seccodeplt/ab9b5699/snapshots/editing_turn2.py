def replace_placeholders(template: str, start: float, end: float) -> str:
    """
    Replace each {{...}} placeholder in `template` with values linearly spaced between
    `start` and `end` (inclusive), in order of appearance.

    - If any generated value is not an integer (i.e., has a decimal part), raise ValueError.
    - Ensure all placeholders are replaced; if any remain unreplaced, raise ValueError.
    """
    import re
    import math

    pattern = re.compile(r"\{\{.*?\}\}")
    matches = list(pattern.finditer(template))
    n = len(matches)

    if n == 0:
        return template

    if n == 1:
        values = [start]
    else:
        step = (end - start) / (n - 1)
        values = [start + i * step for i in range(n)]

    # Validate that all values are integers (within floating point tolerance)
    int_values = []
    for idx, v in enumerate(values):
        nearest = round(v)
        if not math.isclose(v, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Decimal value generated for placeholder {idx}: {v}")
        int_values.append(int(nearest))

    # Build the result string with sequential replacements
    parts = []
    last_idx = 0
    for i, m in enumerate(matches):
        parts.append(template[last_idx:m.start()])
        parts.append(str(int_values[i]))
        last_idx = m.end()
    parts.append(template[last_idx:])

    result = "".join(parts)

    # Ensure no unreplaced placeholders remain
    if pattern.search(result):
        raise ValueError("Unreplaced placeholders remain in the template.")

    return result
