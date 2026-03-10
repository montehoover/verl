import re
from typing import Callable, List, Optional

# Precompiled regex patterns for performance
_JS_RE = re.compile(r'(?<![a-z0-9+.\-])javascript\s*:', re.IGNORECASE)
_VBS_RE = re.compile(r'(?<![a-z0-9+.\-])vbscript\s*:', re.IGNORECASE)
_DATA_RE = re.compile(r'(?<![a-z0-9+.\-])data\s*:', re.IGNORECASE)

# Matches a valid image data URL starting at the "data" occurrence:
# data:image/<subtype>[;charset=...][;base64],...
_SAFE_IMAGE_DATA_RE = re.compile(
    r'\A\s*data\s*:\s*image/(?:png|gif|jpe?g|webp|bmp|svg\+xml)'
    r'(?:;charset=[a-z0-9._\-]+)?'
    r'(?:;base64)?'
    r'\s*,',
    re.IGNORECASE | re.DOTALL,
)


def _has_dangerous_script_scheme(text: str) -> bool:
    """
    Pure function: returns True if text contains javascript: or vbscript: scheme.
    """
    return bool(_JS_RE.search(text) or _VBS_RE.search(text))


def _iter_data_scheme_positions(text: str):
    """
    Pure generator: yields start indices of each 'data:' occurrence in text.
    """
    for m in _DATA_RE.finditer(text):
        yield m.start()


def _is_valid_image_data_from(text: str, start_index: int) -> bool:
    """
    Pure function: returns True if the substring from start_index is a valid image data URL.
    """
    return _SAFE_IMAGE_DATA_RE.match(text[start_index:]) is not None


class _PipelineState:
    """
    Internal state container for the pipeline.
    """
    __slots__ = ("raw", "result")

    def __init__(self, raw: str):
        self.raw: str = raw
        self.result: Optional[bool] = None


def _step_validate_input(state: _PipelineState) -> bool:
    """
    Step 1: Validate input type and emptiness.
    - If invalid, set result=False and stop pipeline.
    - Else, continue.
    """
    if not isinstance(state.raw, str) or not state.raw:
        state.result = False
        return False  # stop
    return True  # continue


def _step_check_script_schemes(state: _PipelineState) -> bool:
    """
    Step 2: Check for dangerous script schemes using pure regex helpers.
    - If found, set result=True and stop.
    - Else, continue.
    """
    if _has_dangerous_script_scheme(state.raw):
        state.result = True
        return False
    return True


def _step_check_data_schemes(state: _PipelineState) -> bool:
    """
    Step 3: Inspect all 'data:' occurrences; flag dangerous unless valid image data URLs.
    - If any unsafe data URL is found, set result=True and stop.
    - Else, continue.
    """
    for pos in _iter_data_scheme_positions(state.raw):
        if not _is_valid_image_data_from(state.raw, pos):
            state.result = True
            return False
    return True


def _step_default_false(state: _PipelineState) -> bool:
    """
    Final step: If no prior step decided, default to result=False and stop.
    """
    if state.result is None:
        state.result = False
    return False


def _run_pipeline(state: _PipelineState, steps: List[Callable[[_PipelineState], bool]]) -> bool:
    """
    Execute steps in order, stopping when a step returns False.
    """
    for step in steps:
        if not step(state):
            break
    # Ensure a boolean is returned
    return bool(state.result)


def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a string contains a dangerous URL scheme.

    Dangerous schemes considered:
    - javascript:
    - vbscript:
    - data: (except valid image data URLs)

    Args:
        user_input: The input string to inspect.

    Returns:
        True if any dangerous scheme is found (ignoring valid image data URLs), otherwise False.

    Notes:
        This function does not explicitly raise exceptions, but exceptions from the 're'
        module (e.g., re.error) could be raised if regex patterns are incorrect.
    """
    state = _PipelineState(user_input)
    steps = [
        _step_validate_input,
        _step_check_script_schemes,
        _step_check_data_schemes,
        _step_default_false,
    ]
    return _run_pipeline(state, steps)
