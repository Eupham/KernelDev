import random
import re
from typing import List, Tuple # Ensure List and Tuple are imported

def word_levenshtein_distance(s1_words: List[str], s2_words: List[str]) -> int:
    """
    Calculates the Levenshtein distance between two lists of words.
    This is the version used by text_utils.py in other parts of the codebase.
    """
    if len(s1_words) < len(s2_words):
        return word_levenshtein_distance(s2_words, s1_words)

    if len(s2_words) == 0:
        return len(s1_words)

    previous_row = range(len(s2_words) + 1)
    for i, word1 in enumerate(s1_words):
        current_row = [i + 1]
        for j, word2 in enumerate(s2_words):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (word1 != word2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def shuffle_words_in_sentence(sentence_text: str, shuffle_probability: float) -> Tuple[List[int], List[float]]:
    """
    Shuffles words in a sentence to create a shuffled byte sequence and target rank scores.
    The rank scores indicate the normalized original 1-indexed position of each byte.

    Args:
        sentence_text: The original sentence string.
        shuffle_probability: The probability (0.0 to 1.0) for a word to be included in the shuffle pool.

    Returns:
        A tuple containing:
        - shuffled_byte_sequence (List[int]): List of byte values for the shuffled sentence.
        - target_rank_scores (List[float]): List of normalized original ranks, corresponding
                                             to each byte in shuffled_byte_sequence.
                                             Returns empty lists if sentence_text is empty/whitespace.
    """
    stripped_sentence = sentence_text.strip()
    if not stripped_sentence:
        return [], []

    # 1. Get original bytes with their global 1-indexed positions
    # Ensure we use the stripped version to define words, but original for byte positions if needed.
    # However, the example "cat dog" implies ranks are based on "c a t ' ' d o g".
    # So, use sentence_text for original byte positions, but stripped_sentence for word splitting.

    raw_original_bytes = list(sentence_text.encode('utf-8')) # Bytes from original, potentially with leading/trailing spaces
    L_orig_bytes_raw = len(raw_original_bytes)
    if L_orig_bytes_raw == 0: return [],[] # Should be caught by strip, but defensive.

    original_items_with_pos_raw = [] # list of (byte_val, orig_1_idx_pos) from raw sentence
    for i, byte_val in enumerate(raw_original_bytes):
        original_items_with_pos_raw.append((byte_val, i + 1))

    # 2. Word Segmentation based on stripped_sentence
    #    Identify byte spans for each word and for spaces within the stripped sentence.
    #    Map these spans back to original_items_with_pos_raw using character offsets.

    word_units = [] # list of lists of tuples: [ [(b,p),...], [(b,p),...], ... ]
                    # Each tuple is (byte_value, original_1_indexed_position_in_raw_sentence)

    # Find character offset of stripped_sentence within sentence_text
    char_offset_start = 0
    if sentence_text and stripped_sentence: # Check if either is empty
        try:
            char_offset_start = sentence_text.index(stripped_sentence)
        except ValueError: # Should not happen if stripped_sentence comes from sentence_text
            char_offset_start = 0
            # If it does happen, means stripped_sentence is not a substring, implies sentence_text was only spaces
            # This case is handled by "if not stripped_sentence" already.

    current_char_idx_in_stripped = 0
    for match in re.finditer(r'\s+', stripped_sentence):
        # Word before space
        word_str_segment = stripped_sentence[current_char_idx_in_stripped:match.start()]
        if word_str_segment:
            byte_start_idx = len(sentence_text[:char_offset_start + current_char_idx_in_stripped].encode('utf-8'))
            byte_end_idx = len(sentence_text[:char_offset_start + match.start()].encode('utf-8'))
            word_units.append(original_items_with_pos_raw[byte_start_idx:byte_end_idx])

        # Space segment
        space_str_segment = stripped_sentence[match.start():match.end()]
        byte_start_idx_space = len(sentence_text[:char_offset_start + match.start()].encode('utf-8'))
        byte_end_idx_space = len(sentence_text[:char_offset_start + match.end()].encode('utf-8'))
        word_units.append(original_items_with_pos_raw[byte_start_idx_space:byte_end_idx_space])

        current_char_idx_in_stripped = match.end()

    # Last word
    last_word_str_segment = stripped_sentence[current_char_idx_in_stripped:]
    if last_word_str_segment:
        byte_start_idx = len(sentence_text[:char_offset_start + current_char_idx_in_stripped].encode('utf-8'))
        byte_end_idx = len(sentence_text[:char_offset_start + len(stripped_sentence)].encode('utf-8'))
        word_units.append(original_items_with_pos_raw[byte_start_idx:byte_end_idx])

    if not word_units: # e.g. if stripped_sentence was empty (already handled) or contained no parsable units
        return [], []

    # 3. Select indices of "actual word" units to shuffle (non-space units)
    actual_word_unit_indices = [i for i, unit in enumerate(word_units) if unit and not (len(unit)==1 and unit[0][0]==32 and sentence_text[sentence_text.find(stripped_sentence)+stripped_sentence.find(chr(unit[0][0])) : sentence_text.find(stripped_sentence)+stripped_sentence.find(chr(unit[0][0]))+len(bytes([unit[0][0]]).decode('utf-8',errors='ignore'))].isspace())]


    indices_of_word_units_to_shuffle = [idx for idx in actual_word_unit_indices if random.random() < shuffle_probability]

    final_ordered_word_units = list(word_units) # Make a mutable copy

    if len(indices_of_word_units_to_shuffle) > 1:
        # Extract the units to be shuffled
        subset_to_shuffle = [final_ordered_word_units[i] for i in indices_of_word_units_to_shuffle]
        random.shuffle(subset_to_shuffle) # Shuffle this subset

        # Place them back into their original positions within the list of units
        for i, original_pos_idx in enumerate(indices_of_word_units_to_shuffle):
            final_ordered_word_units[original_pos_idx] = subset_to_shuffle[i]

    # 4. Flatten final_ordered_word_units into shuffled_items_with_orig_pos
    shuffled_items_with_orig_pos = []
    for unit in final_ordered_word_units:
        shuffled_items_with_orig_pos.extend(unit)

    # 5. Extract shuffled_byte_sequence and target_rank_scores
    # L_orig_bytes_raw is the length of the original full sentence bytes (incl leading/trailing spaces)
    # The ranks should be normalized by this raw length.
    shuffled_byte_sequence = [item[0] for item in shuffled_items_with_orig_pos]
    target_rank_scores = [float(item[1]) / L_orig_bytes_raw for item in shuffled_items_with_orig_pos]

    return shuffled_byte_sequence, target_rank_scores


if __name__ == '__main__':
    test_sentences = [
        "cat dog mouse",
        "  leading spaces cat dog",
        "cat dog trailing spaces  ",
        "  cat  multiple   spaces dog  ",
        "singleword",
        "  ", # Only spaces
        "",   # Empty
        "sentence with punctuation ! ? ."
    ]

    probs = [0.0, 0.5, 1.0]

    for sentence in test_sentences:
        print(f"\n--- Original sentence: '{sentence}' (len: {len(sentence)}) ---")
        original_bytes_for_test = list(sentence.encode('utf-8'))
        # print(f"Original bytes: {original_bytes_for_test}")
        # print(f"Original byte ranks: {[i+1 for i in range(len(original_bytes_for_test))]}")


        for p in probs:
            shuffled_bytes, ranks = shuffle_words_in_sentence(sentence, p)

            shuffled_text_for_print = ""
            if shuffled_bytes:
                try:
                    shuffled_text_for_print = bytes(shuffled_bytes).decode('utf-8', errors='replace')
                except Exception as e:
                    shuffled_text_for_print = f"[Decode Error: {e}] {str(shuffled_bytes)}"
            else:
                shuffled_text_for_print = "[Empty Result]"


            print(f"  p={p:.2f}:")
            # print(f"    Shuffled bytes: {shuffled_bytes}")
            print(f"    Shuffled text:  '{shuffled_text_for_print}' (len_bytes: {len(shuffled_bytes)})")
            # print(f"    Target ranks:   {[float(f'{r:.3f}') for r in ranks]}")
            if ranks:
                # Denormalize for easier verification: rank * L_orig_bytes_raw
                L_raw = len(list(sentence.encode('utf-8')))
                denormalized_ranks = [int(round(r * L_raw)) for r in ranks]
                print(f"    Original Pos (denormalized): {denormalized_ranks}")
            else:
                print(f"    Original Pos (denormalized): []")


    # Example from problem: "cat dog" -> "dog cat"
    # Original: "cat dog". Bytes: c a t ' ' d o g. Positions: 1 2 3 4 5 6 7. L_raw = 7
    # Shuffled words: "dog cat".
    # Bytes of shuffled: d o g ' ' c a t.
    # Target ranks for "dog cat": [5/7, 6/7, 7/7, 4/7, 1/7, 2/7, 3/7]
    # Denormalized: [5, 6, 7, 4, 1, 2, 3]
    print("\n--- Specific Example: 'cat dog' ---")
    sentence_specific = "cat dog"
    # Force a shuffle for testing (monkey patch random if needed, or just run multiple times)
    # For now, we'll rely on a high probability and check output.
    # To guarantee shuffle for this specific test, one might temporarily fix random.random or random.shuffle.

    # Test with shuffle_probability = 1.0 to maximize chance of shuffling
    shuffled_bytes, ranks = shuffle_words_in_sentence(sentence_specific, 1.0)
    shuffled_text_for_print = bytes(shuffled_bytes).decode('utf-8', errors='replace')
    L_raw_specific = len(list(sentence_specific.encode('utf-8')))
    denormalized_ranks = [int(round(r * L_raw_specific)) for r in ranks]

    print(f"Original: '{sentence_specific}'")
    print(f"Shuffled: '{shuffled_text_for_print}'")
    print(f"Denormalized Target Ranks: {denormalized_ranks}")
    # Expected if "dog cat": [5, 6, 7, 4, 1, 2, 3]
    # Expected if "cat dog" (no shuffle): [1, 2, 3, 4, 5, 6, 7]

    # Test with a sentence that includes numbers and mixed case
    sentence_complex = "Word1 2word and Word3"
    print(f"\n--- Complex Example: '{sentence_complex}' ---")
    shuffled_bytes_complex, ranks_complex = shuffle_words_in_sentence(sentence_complex, 1.0)
    shuffled_text_complex = bytes(shuffled_bytes_complex).decode('utf-8', errors='replace')
    L_raw_complex = len(list(sentence_complex.encode('utf-8')))
    denormalized_ranks_complex = [int(round(r * L_raw_complex)) for r in ranks_complex]
    print(f"Original: '{sentence_complex}'")
    print(f"Shuffled: '{shuffled_text_complex}'")
    print(f"Denormalized Target Ranks: {denormalized_ranks_complex}")

```
