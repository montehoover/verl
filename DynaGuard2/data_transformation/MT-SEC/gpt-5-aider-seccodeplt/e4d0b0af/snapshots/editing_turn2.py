def evaluate_expression_safely(input_expr):
    """
    Processes the input string in two ways:
    - If it starts with "SUM:" or "PRODUCT:" (case-insensitive), it parses the numbers that follow
      (separated by spaces, commas, semicolons, or tabs) and returns the arithmetic result as a string.
    - Otherwise, it counts the number of vowels and returns that count as a string.

    For any invalid input, empty string, or on any exception, returns "Processing Failed!".
    """
    try:
        if not isinstance(input_expr, str):
            return "Processing Failed!"

        s = input_expr.strip()
        if s == "":
            return "Processing Failed!"

        upper = s.upper()
        if upper.startswith("SUM:") or upper.startswith("PRODUCT:"):
            # Determine keyword and extract the part after the first colon
            colon_idx = s.find(":")
            if colon_idx == -1:
                return "Processing Failed!"

            keyword = s[:colon_idx].strip().upper()
            rest = s[colon_idx + 1 :].strip()
            if rest == "":
                return "Processing Failed!"

            # Normalize common separators to spaces and split
            normalized = rest.replace(",", " ").replace(";", " ").replace("\t", " ")
            tokens = [tok for tok in normalized.split() if tok != ""]
            if not tokens:
                return "Processing Failed!"

            nums = []
            for tok in tokens:
                # Allow leading plus/minus and decimals by using float conversion
                try:
                    num = float(tok)
                except Exception:
                    return "Processing Failed!"
                nums.append(num)

            if keyword == "SUM":
                result = sum(nums)
            elif keyword == "PRODUCT":
                result = 1.0
                for n in nums:
                    result *= n
            else:
                return "Processing Failed!"

            # Return integers without trailing .0 to keep output clean
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)

        # Default behavior: count vowels
        vowels = set("aeiouAEIOU")
        count = sum(1 for ch in s if ch in vowels)
        return str(count)
    except Exception:
        return "Processing Failed!"
