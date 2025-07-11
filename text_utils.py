import random
import re

def shuffle_words_in_sentence(sentence_text: str, shuffle_probability: float) -> tuple[str, list[str], list[str], list[int]]:
    """
    Splits a sentence into words. Based on shuffle_probability, selects a subset
    of words to shuffle among their original positions.
    Returns the shuffled sentence, original word list, the new word list (final_shuffled_word_list),
    and permuted_word_source_indices.
    permuted_word_source_indices[i] is the original index in original_words of the word
    that is now at final_shuffled_word_list[i].
    """
    stripped_sentence = sentence_text.strip()
    if not stripped_sentence:
        return "", [], [], []

    original_words = [word for word in re.split(r'\s+', stripped_sentence) if word]
    if not original_words: # Handles cases like sentence_text being just spaces
        return "", [], [], []

    # Initialize permuted_word_source_indices as identity mapping
    # This list tracks the original index of the word at each position in the (eventually) shuffled list.
    permuted_word_source_indices = list(range(len(original_words)))

    if shuffle_probability <= 0: # No shuffling if probability is zero or less
        return stripped_sentence, original_words, original_words[:], permuted_word_source_indices

    # For shuffle_probability >= 1, all words are candidates for shuffling.
    # The logic below naturally handles this by selecting all indices if random.random() < 1.0 always true.

    indices_to_shuffle = [] # These are indices *within original_words* that are chosen to be part of the shuffle
    for i in range(len(original_words)):
        if random.random() < shuffle_probability:
            indices_to_shuffle.append(i)

    if len(indices_to_shuffle) <= 1:
        # If 0 or 1 word is selected, no shuffle occurs / is meaningful for the subset
        final_shuffled_word_list = original_words[:]
        # permuted_word_source_indices remains identity, which is correct
    else:
        # subset_to_shuffle contains the actual word strings
        subset_to_shuffle = [original_words[i] for i in indices_to_shuffle]

        # original_indices_of_subset_words contains the original indices of these words
        # These are the values from permuted_word_source_indices at the chosen positions
        original_indices_of_subset_words = [permuted_word_source_indices[i] for i in indices_to_shuffle]

        # Shuffle the subset of words and their corresponding original indices together
        # to maintain their association after shuffling.
        zipped_subset_and_indices = list(zip(subset_to_shuffle, original_indices_of_subset_words))
        random.shuffle(zipped_subset_and_indices)

        # Unzip back into separate lists
        shuffled_subset_words_only, shuffled_original_indices_for_subset = [], []
        if zipped_subset_and_indices: # Ensure not empty before trying to unzip
             shuffled_subset_words_only, shuffled_original_indices_for_subset = zip(*zipped_subset_and_indices)

        final_shuffled_word_list = original_words[:] # Start with a copy of original words

        # Place the shuffled words and their tracked original indices back into the full list context
        for i, target_position_in_sentence in enumerate(indices_to_shuffle):
            final_shuffled_word_list[target_position_in_sentence] = shuffled_subset_words_only[i]
            permuted_word_source_indices[target_position_in_sentence] = shuffled_original_indices_for_subset[i]

    shuffled_sentence_text = " ".join(final_shuffled_word_list)
    return shuffled_sentence_text, original_words, final_shuffled_word_list, permuted_word_source_indices

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
