"""DNA sequence scoring utilities."""


MATCH_SCORE = 1
MISMATCH_SCORE = -1


def score_bases(base1, base2):
    """Return a score for comparing two DNA bases."""
    if base1 == base2:
        return MATCH_SCORE

    return MISMATCH_SCORE