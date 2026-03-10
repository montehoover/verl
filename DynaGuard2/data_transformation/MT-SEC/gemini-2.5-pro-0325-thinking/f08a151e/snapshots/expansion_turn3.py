def tag_exists(html_string: str, tag: str) -> bool:
    """
    Checks if a specific HTML tag is present in a given string.
    This is a simple check and looks for "<tag_name" (for opening tags like <tag> or <tag attr>)
    or "</tag_name>" (for closing tags like </tag>).
    The check is case-sensitive.

    Args:
        html_string: The string containing HTML content.
        tag: The HTML tag name to search for (e.g., "p", "div").

    Returns:
        True if the tag is found, False otherwise.
    """
    opening_tag_marker = f"<{tag}"  # Matches "<tag" which covers <tag> and <tag ...>
    closing_tag_marker = f"</{tag}>" # Matches "</tag>"

    if opening_tag_marker in html_string or closing_tag_marker in html_string:
        return True
    return False

import re
from collections import defaultdict

# NOTE: The HTML parsing functions below are simplified and may not correctly handle
# all edge cases, such as HTML comments, CDATA sections, script/style tag contents,
# malformed HTML, or attributes containing special characters like '>'.
# They assume reasonably well-formed HTML for tag structure.

def _find_matching_end_tag(html_string: str, tag_name: str, start_pos_after_opening_tag: int) -> int:
    """
    Finds the starting index of the matching closing tag for a given tag.
    Accounts for nested tags of the same name.
    Args:
        html_string: The HTML content to search within.
        tag_name: The name of the tag to find the closing tag for.
        start_pos_after_opening_tag: The index in html_string after the opening tag.
    Returns:
        The starting index of the closing tag, or -1 if not found.
    """
    # Regexes for finding opening and closing tags of the specified name
    # Using re.escape on tag_name in case it contains special regex characters
    # (?i) for inline IGNORECASE, or pass re.IGNORECASE flag
    open_tag_regex = re.compile(r"<" + re.escape(tag_name) + r"(?:\s+[^>]*)?>", re.IGNORECASE)
    close_tag_regex = re.compile(r"</" + re.escape(tag_name) + r"\s*>", re.IGNORECASE)
    
    level = 1
    current_pos = start_pos_after_opening_tag
    while current_pos < len(html_string):
        # Search for the next opening or closing tag
        # Find the earliest of the two
        next_open_match = open_tag_regex.search(html_string, current_pos)
        next_close_match = close_tag_regex.search(html_string, current_pos)

        if next_close_match is None:
            return -1 # Malformed: no closing tag found for this tag_name

        # If an opening tag of the same kind is found before a closing one
        if next_open_match is not None and next_open_match.start() < next_close_match.start():
            level += 1
            current_pos = next_open_match.end()
        else: # A closing tag is found
            level -= 1
            if level == 0:
                return next_close_match.start() # Found the matching closing tag
            current_pos = next_close_match.end()
            
    return -1 # Should not be reached if HTML is well-formed and level started at 1

def _extract_elements_at_current_level(html_string: str) -> list:
    """
    Extracts top-level HTML elements from a string.
    Ignores text nodes that are direct children of the current fragment for simplicity.
    Args:
        html_string: The HTML content to parse.
    Returns:
        A list of tuples, where each tuple is (tag_name, inner_html, start_idx, end_idx).
    """
    elements = []
    cursor = 0
    while cursor < len(html_string):
        # Find the next opening tag. This regex is simple and might grab things in script tags etc.
        # It doesn't handle comments, CDATA, etc.
        match = re.search(r"<(\w+)([^>]*)>", html_string, cursor) 
        if not match:
            break # No more tags

        tag_name_original = match.group(1)
        tag_name = tag_name_original.lower() # Normalize tag name to lowercase
        
        open_tag_full_text = match.group(0)
        open_tag_start_idx = match.start()
        open_tag_end_idx = match.end()

        # Handle self-closing tags (e.g., <br/>, <img ... />)
        if open_tag_full_text.endswith("/>"):
            elements.append((tag_name, "", open_tag_start_idx, open_tag_end_idx))
            cursor = open_tag_end_idx
            continue

        # Find the matching closing tag for non-self-closing tags
        closing_tag_start_idx = _find_matching_end_tag(html_string, tag_name_original, open_tag_end_idx)
        
        if closing_tag_start_idx == -1:
            # Malformed HTML (e.g., unclosed tag).
            # This simplistic parser will skip such a tag by advancing the cursor.
            # A more robust parser might try to recover or report errors.
            # print(f"Warning: No closing tag found for <{tag_name_original}> starting at {open_tag_start_idx}")
            cursor = open_tag_end_idx 
            continue

        # Determine the end of the closing tag
        # We need to match the exact closing tag format </tag_name>
        closing_tag_match = re.match(r"</" + re.escape(tag_name_original) + r"\s*>", html_string[closing_tag_start_idx:], re.IGNORECASE)
        if not closing_tag_match:
            # This should ideally not happen if _find_matching_end_tag is correct and HTML is not severely malformed
            # print(f"Warning: Could not match exact closing tag for <{tag_name_original}> at {closing_tag_start_idx}")
            cursor = open_tag_end_idx
            continue
            
        closing_tag_end_idx = closing_tag_start_idx + closing_tag_match.end()
        
        inner_html = html_string[open_tag_end_idx:closing_tag_start_idx]
        elements.append((tag_name, inner_html, open_tag_start_idx, closing_tag_end_idx))
        cursor = closing_tag_end_idx # Move cursor to the end of the processed element
        
    return elements

def extract_tag_contents(html_string: str) -> dict:
    """
    Extracts content from HTML tags in a given string.

    The function identifies HTML tags and their inner content. It returns a 
    dictionary where keys are tag names (normalized to lowercase) and values 
    are lists of strings, each string being the inner HTML of an occurrence 
    of that tag.

    The parsing is done by processing the HTML string and its fragments 
    iteratively. When a tag's inner HTML is extracted, if it might contain 
    further tags, it's added to a queue for subsequent processing. This ensures 
    that nested structures are also analyzed.

    Args:
        html_string: The string containing HTML content.

    Returns:
        A dictionary mapping tag names to a list of their inner HTML contents.
        Example: {"p": ["paragraph one", "paragraph two with <span>nested</span>"], "span": ["nested"]}
    """
    results = defaultdict(list)
    
    queue = []
    processed_fragments = set()

    if html_string:
        queue.append(html_string)
        processed_fragments.add(html_string)

    head = 0
    while head < len(queue): # Using list as a queue with a head index
        current_html_fragment = queue[head]
        head += 1
        
        top_level_elements_in_fragment = _extract_elements_at_current_level(current_html_fragment)
        
        for tag_name, inner_html, _, _ in top_level_elements_in_fragment:
            results[tag_name].append(inner_html)
            
            # If the inner HTML is not empty and seems to contain more tags,
            # add it to the queue for further processing, if not already processed.
            if inner_html.strip() and ("<" in inner_html and ">" in inner_html):
                if inner_html not in processed_fragments:
                     queue.append(inner_html)
                     processed_fragments.add(inner_html)
                     
    return dict(results)

def get_html_tags(html_input: str) -> list:
    """
    Extracts all unique HTML tag names present in a given string using regular expressions.

    This function finds all opening and closing tags and returns a list of unique
    tag names, normalized to lowercase. For example, if the input contains
    <P>, <p>, and </p>, the output will include "p" once.

    Args:
        html_input: The string containing HTML content.

    Returns:
        A list of unique HTML tag names found in the input string.
        Example: ["html", "head", "title", "body", "p", "a"]
    """
    # Regex to find tag names in opening tags (e.g., <div class="foo">)
    # and closing tags (e.g., </div>).
    # It captures the tag name (e.g., "div").
    # - <\s*: Matches the opening '<' possibly followed by whitespace.
    # - /?\s*: Matches an optional '/' (for closing tags) possibly followed by whitespace.
    # - (\w+): Captures the tag name (one or more word characters).
    # - (?:[^>]*): Matches any characters that are not '>' (attributes, etc.), non-capturing.
    # - >: Matches the closing '>'.
    tag_regex = re.compile(r"<\s*/?\s*(\w+)(?:[^>]*)?>")
    
    found_tags = tag_regex.findall(html_input)
    
    # Normalize to lowercase and get unique tags
    unique_tags = sorted(list(set(tag.lower() for tag in found_tags)))
    
    return unique_tags
