"""DNA sequence scoring utilities."""


MATCH_SCORE = 1
MISMATCH_SCORE = -1


def score_bases(base1, base2):
    """Return a score for comparing two DNA bases."""
    if base1 == base2:
        return MATCH_SCORE

    return MISMATCH_SCORE


def score_sequences(sequence1, sequence2):
    """Return the total score for two DNA sequences."""
    if len(sequence1) != len(sequence2):
        raise ValueError("Sequences must have the same length")

    total_score = 0

    for base1, base2 in zip(sequence1, sequence2):
        total_score += score_bases(base1, base2)

    return total_score