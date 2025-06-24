import random
import re

def shuffle_words_in_sentence(sentence_text: str) -> tuple[str, list[str], list[str]]:
    """
    Splits a sentence into words, shuffles them, and rejoins.
    Returns the shuffled sentence, original word list, and shuffled word list.
    Uses regex \s+ to split by any whitespace and handles multiple spaces.
    Filters out empty strings that might result from multiple spaces.
    """
    if not sentence_text.strip():
        return "", [], []

    original_words = [word for word in re.split(r'\s+', sentence_text.strip()) if word]
    if not original_words:
        return "", [], []

    shuffled_words = original_words[:] # Create a copy

    # Shuffle only if there's more than one word to avoid infinite loop with single word
    if len(shuffled_words) > 1:
        # Keep shuffling until it's different from original or max attempts
        # This is to ensure the shuffled version is actually different for meaningful training examples.
        # However, for very short sentences or sentences with repeated words, it might always be the same.
        # A simpler approach for now: just shuffle once. If it's the same, the Levenshtein distance will be 0.
        random.shuffle(shuffled_words)

    shuffled_sentence = " ".join(shuffled_words)
    return shuffled_sentence, original_words, shuffled_words

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
    sentence1 = "This is a sample sentence."
    shuffled_s1, orig_w1, shuf_w1 = shuffle_words_in_sentence(sentence1)
    print(f"Original: '{sentence1}' -> Shuffled: '{shuffled_s1}'")
    print(f"Original words: {orig_w1}")
    print(f"Shuffled words: {shuf_w1}")

    sentence2 = "  Another   example with   multiple spaces  "
    shuffled_s2, orig_w2, shuf_w2 = shuffle_words_in_sentence(sentence2)
    print(f"Original: '{sentence2.strip()}' -> Shuffled: '{shuffled_s2}'")

    sentence3 = "Test"
    shuffled_s3, orig_w3, shuf_w3 = shuffle_words_in_sentence(sentence3)
    print(f"Original: '{sentence3}' -> Shuffled: '{shuffled_s3}'")

    sentence4 = ""
    shuffled_s4, orig_w4, shuf_w4 = shuffle_words_in_sentence(sentence4)
    print(f"Original: '{sentence4}' -> Shuffled: '{shuffled_s4}'")

    # Test word_levenshtein_distance
    words1 = ["the", "quick", "brown", "fox"]
    words2 = ["the", "fast", "brown", "foxes"]
    dist1 = word_levenshtein_distance(words1, words2)
    print(f"Distance between {words1} and {words2}: {dist1}") # Expected: 2 (fast!=quick, foxes!=fox)

    words3 = ["apple", "banana", "cherry"]
    words4 = ["apple", "cherry", "banana"]
    dist2 = word_levenshtein_distance(words3, words4) # Should be 2 (banana/cherry swap requires 2 ops: del banana, ins banana or 2 subs)
    print(f"Distance between {words3} and {words4}: {dist2}")

    dist3 = word_levenshtein_distance(orig_w1, shuf_w1)
    print(f"Distance for shuffled sentence 1 ('{sentence1}'): {dist3}")

    # Test with identical shuffled (e.g. single word)
    if orig_w3 and shuf_w3: # only if not empty
        dist_s3 = word_levenshtein_distance(orig_w3, shuf_w3)
        print(f"Distance for shuffled sentence 3 ('{sentence3}'): {dist_s3}")


    words5 = ["a", "b", "c"]
    words6 = ["a", "d", "c", "e"]
    dist5 = word_levenshtein_distance(words5, words6)
    print(f"Distance between {words5} and {words6}: {dist5}") # Expected: 2 (sub b->d, ins e)
```
