def simple_shift(text: str) -> str:
    """
    Return the Caesar-shifted string of the input using a fixed key.
    Non-alphabetic characters are left unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("simple_shift expects a string input")
    shift = 3  # fixed key
    result = []
    for ch in text:
        if 'a' <= ch <= 'z':
            base = ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        elif 'A' <= ch <= 'Z':
            base = ord('A')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)
