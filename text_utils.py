import random
import re

def shuffle_words_in_sentence(sentence_text: str, shuffle_probability: float) -> tuple[str, list[str], list[str]]:
    """
    Splits a sentence into words. Based on shuffle_probability, selects a subset
    of words to shuffle among their original positions.
    Returns the shuffled sentence, original word list, and the new word list.
    """
    if not sentence_text.strip():
        return "", [], []

    original_words = [word for word in re.split(r'\s+', sentence_text.strip()) if word]
    if not original_words:
        return "", [], []

    if shuffle_probability <= 0: # No shuffling if probability is zero or less
        return sentence_text, original_words, original_words[:]
    if shuffle_probability >= 1: # Full shuffle if probability is one or more
        # This can revert to the simpler full shuffle logic for p=1.0
        # For consistency with the subset logic, let's ensure it still works.
        # All indices will be selected.
        pass


    indices_to_shuffle = []
    for i in range(len(original_words)):
        if random.random() < shuffle_probability:
            indices_to_shuffle.append(i)

    if len(indices_to_shuffle) <= 1:
        # If 0 or 1 word is selected, no shuffle occurs / is meaningful for the subset
        final_shuffled_word_list = original_words[:]
    else:
        subset_to_shuffle = [original_words[i] for i in indices_to_shuffle]

        # Ensure the shuffled subset is actually different if possible (optional, can be complex)
        # For now, a simple shuffle of the subset:
        shuffled_subset = subset_to_shuffle[:] # Copy
        random.shuffle(shuffled_subset)
        # A loop to ensure difference might be too slow if subset is hard to change
        # while shuffled_subset == subset_to_shuffle and len(set(subset_to_shuffle)) > 1:
        #    random.shuffle(shuffled_subset)

        final_shuffled_word_list = original_words[:]
        for original_idx_in_sentence, new_word_for_pos in zip(indices_to_shuffle, shuffled_subset):
            final_shuffled_word_list[original_idx_in_sentence] = new_word_for_pos

    shuffled_sentence_text = " ".join(final_shuffled_word_list)
    return shuffled_sentence_text, original_words, final_shuffled_word_list

def word_levenshtein_distance(s1_words: list[str], s2_words: list[str]) -> int:
    """
    Calculates the Levenshtein distance between two lists of words.
    This is a basic implementation.
    s1_words: list of words of sentence 1
    s2_words: list of words of sentence 2
    Returns: The Levenshtein distance (int).
    """
    if len(s1_words) < len(s2_words):
        return word_levenshtein_distance(s2_words, s1_words)

    if len(s2_words) == 0:
        return len(s1_words)

    previous_row = range(len(s2_words) + 1)
    for i, c1 in enumerate(s1_words):
        current_row = [i + 1]
        for j, c2 in enumerate(s2_words):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

if __name__ == '__main__':
    # Test shuffle_words_in_sentence
    sentence1 = "This is a sample sentence for testing the new shuffle logic."
    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"Original: '{sentence1}'")
    for p in probs:
        shuffled_s1, orig_w1, shuf_w1 = shuffle_words_in_sentence(sentence1, p)
        dist = word_levenshtein_distance(orig_w1, shuf_w1)
        print(f"  p={p:.2f} -> Shuffled: '{shuffled_s1}' (Dist: {dist})")

    sentence2 = "  Another   example with   multiple spaces  "
    shuffled_s2, _, _ = shuffle_words_in_sentence(sentence2, 0.5)
    print(f"Original: '{sentence2.strip()}' -> Shuffled (p=0.5): '{shuffled_s2}'")

    sentence3 = "Test" # Single word
    shuffled_s3, _, _ = shuffle_words_in_sentence(sentence3, 1.0)
    print(f"Original: '{sentence3}' -> Shuffled (p=1.0): '{shuffled_s3}'")

    sentence4 = ""
    shuffled_s4, _, _ = shuffle_words_in_sentence(sentence4, 1.0)
    print(f"Original: '{sentence4}' -> Shuffled (p=1.0): '{shuffled_s4}'")

    # Test word_levenshtein_distance
    words1 = ["the", "quick", "brown", "fox"]
    words2 = ["the", "fast", "brown", "foxes"]
    dist1 = word_levenshtein_distance(words1, words2)
    print(f"Distance between {words1} and {words2}: {dist1}")

    words3 = ["apple", "banana", "cherry"]
    words4 = ["apple", "cherry", "banana"]
    dist2 = word_levenshtein_distance(words3, words4)
    print(f"Distance between {words3} and {words4}: {dist2}")

    # Example with one of the shuffled sentences from above
    # Note: orig_w1 and shuf_w1 are from the last iteration of the loop (p=1.0)
    if orig_w1 and shuf_w1 : # Check if they are not empty
      dist_shuffled_example = word_levenshtein_distance(orig_w1, shuf_w1)
      print(f"Distance for p=1.0 shuffled sentence 1 ('{sentence1}'): {dist_shuffled_example}")

    words5 = ["a", "b", "c"]
    words6 = ["a", "d", "c", "e"]
    dist5 = word_levenshtein_distance(words5, words6)
    print(f"Distance between {words5} and {words6}: {dist5}")
