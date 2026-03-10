def replace_placeholders(template: str, start: float, end: float) -> str:
    """
    Replace each {{...}} placeholder in `template` with values linearly spaced between
    `start` and `end` (inclusive), in order of appearance.

    - If there is one placeholder, it is replaced with `start`.
    - If there are no placeholders, the template is returned unchanged.
    """
    import re

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

    # Build the result string with sequential replacements
    parts = []
    last_idx = 0
    for i, m in enumerate(matches):
        parts.append(template[last_idx:m.start()])
        val = values[i]
        if isinstance(val, float) and abs(val - round(val)) < 1e-12:
            val_str = str(int(round(val)))
        else:
            val_str = "{:.12g}".format(val)
        parts.append(val_str)
        last_idx = m.end()
    parts.append(template[last_idx:])

    return "".join(parts)
