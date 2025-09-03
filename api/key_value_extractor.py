import re
from typing import List, Dict, Any

# --- Configuration (No changes needed here) ---
ALL_KEYWORDS = [
    "address line 1", "address line 2", "address line", "address",
    "first name", "middle name", "last name", "name", "full name",
    "age", "gender", "date of birth", "dob", "birth date",
    "phone", "phone number", "mobile", "contact no",
    "email", "email id", "e-mail address",
    "city", "state", "pin code", "location",
]

REGEX_PATTERNS = {
    "email": r'\b[A-Za-z0.9_%+-]+@[A-Za-z0.9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "date": r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
}
# --- End Configuration ---

# --- NEW: Cleaning Configuration ---

# 1. Define which detected keys map to which cleaning "rule type".
# This allows "First name" and "Last name" to share the same "name" cleaning rule.
KEY_TO_RULE_MAP = {
    "first name": "name", "middle name": "name", "last name": "name", "name": "name", "full name": "name",
    "phone": "phone", "phone number": "phone", "mobile": "phone", "contact no": "phone",
    "email": "email", "email id": "email",
    "address line 1": "address", "address line 2": "address", "address": "address",
    "city": "name", "state": "name", # Cities and states are like names, no special chars
    "pin code": "pincode",
    "date of birth": "date", "dob": "date", "date": "date",
    "gender": "name",
}

# 2. Define the cleaning rules for each value type.
# We specify exactly which characters to remove for each category.
CLEANING_RULES_FOR_VALUES = {
    "name": "!#$*[]{}",                # Remove most symbols from names
    "phone": "()- ",                   # Remove formatting from phone numbers
    "email": " ",                      # Only remove spaces from emails
    "pincode": " .-",                  # Remove spaces, dots, or hyphens from pin codes
    "address": "!",                    # Be gentle with addresses, only remove obvious errors like '!'
    "date": " ",                       # Only remove spaces
    "default": "!#$*[]{}",             # A fallback rule for unknown keys
}

# 3. Define characters to always strip from the start/end of any key or value.
ALWAYS_STRIP_CHARS = " .:,-"
# --- End Cleaning Configuration ---


def _clean_and_normalize_data(data: Dict[str, str]) -> Dict[str, str]:
    """Applies context-aware cleaning rules to the final extracted data."""
    cleaned_data = {}
    for key, value in data.items():
        # Clean the key: remove trailing symbols
        cleaned_key = key.strip(ALWAYS_STRIP_CHARS)
        
        # Determine which cleaning rule to use based on the key
        rule_type = KEY_TO_RULE_MAP.get(cleaned_key.lower(), "default")
        
        # Get the characters to strip for this value type
        chars_to_strip = CLEANING_RULES_FOR_VALUES.get(rule_type, "")
        
        # Clean the value
        cleaned_value = value
        # First, remove specific unwanted characters from anywhere inside the value
        for char in chars_to_strip:
            cleaned_value = cleaned_value.replace(char, "")
        # Then, strip common trailing characters from the start and end
        cleaned_value = cleaned_value.strip(ALWAYS_STRIP_CHARS)
        
        # A few special-case normalizations
        if rule_type == 'email':
            cleaned_value = cleaned_value.replace(" O ", "@").replace(" gnail ", " gmail ") # Fix common OCR errors
        
        cleaned_data[cleaned_key] = cleaned_value
        
    return cleaned_data


# --- (The rest of the file remains the same until the very last line of the main function) ---

def _find_closest_value_box(key_box: Dict[str, Any], candidate_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_candidate = None
    min_horizontal_dist = float('inf')
    key_x_max = key_box['box'][2]
    key_y_center = (key_box['box'][1] + key_box['box'][3]) / 2
    key_height = key_box['box'][3] - key_box['box'][1]
    for candidate in candidate_boxes:
        candidate_x_min = candidate['box'][0]
        candidate_y_center = (candidate['box'][1] + candidate['box'][3]) / 2
        if candidate_x_min <= key_x_max: continue
        vertical_dist = abs(key_y_center - candidate_y_center)
        if vertical_dist > key_height / 2: continue
        horizontal_dist = candidate_x_min - key_x_max
        if horizontal_dist < min_horizontal_dist:
            min_horizontal_dist = horizontal_dist
            best_candidate = candidate
    return best_candidate

def _extract_with_known_keywords(line_groups: List[Dict[str, Any]], all_word_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = {}
    used_box_ids = set()
    ALL_KEYWORDS.sort(key=len, reverse=True)
    pattern = re.compile(r'(' + '|'.join(ALL_KEYWORDS) + r')\s*:?\s*(.*?)(?=\s*(?:' + '|'.join(ALL_KEYWORDS) + r')|$)', re.IGNORECASE)
    for group in line_groups:
        matches = pattern.findall(group.get('line_text', ''))
        if matches:
            for key, value in matches:
                if key.strip() not in results:
                    results[key.strip()] = value.strip()
            for box in all_word_boxes:
                if box['line_idx'] == group['line_idx']:
                    used_box_ids.add(box['id'])
    remaining_boxes = [box for box in all_word_boxes if box['id'] not in used_box_ids]
    lines = {}
    for box in remaining_boxes:
        lines.setdefault(box['line_idx'], []).append(box)
    for _, boxes_in_line in lines.items():
        for i, box in enumerate(boxes_in_line):
            clean_text = box['text'].lower().replace(':', '').strip()
            if clean_text in ALL_KEYWORDS:
                candidates = [b for j, b in enumerate(boxes_in_line) if i != j]
                value_box = _find_closest_value_box(box, candidates)
                if value_box:
                    key = box['text'].replace(':', '').strip()
                    if key not in results:
                        results[key] = value_box['text']
    return results

def _extract_with_generic_colons(line_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = {}
    pattern = re.compile(r'^(.*?)\s*:\s*(.*)$')
    for group in line_groups:
        match = pattern.match(group.get('line_text', ''))
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key and key not in results:
                results[key] = value
    return results

def extract_key_value_pairs(line_groups: List[Dict[str, Any]], all_word_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    for i, box in enumerate(all_word_boxes):
        box['id'] = i
    keyword_results = _extract_with_known_keywords(line_groups, all_word_boxes)
    colon_results = _extract_with_generic_colons(line_groups)
    final_results = keyword_results.copy()
    for key, value in colon_results.items():
        if key not in final_results:
            final_results[key] = value
    for group in line_groups:
        line_text = group.get('line_text', '').strip()
        if not line_text: continue
        is_date_found = any(k in final_results for k in ["date", "dob", "date of birth"])
        if not is_date_found and re.fullmatch(REGEX_PATTERNS['date'], line_text):
            final_results['date'] = line_text
            continue
    
    # --- MODIFICATION: Call the cleaning function at the end ---
    cleaned_and_normalized_results = _clean_and_normalize_data(final_results)

    return cleaned_and_normalized_results