
"""DNA sequence scoring utilities."""

MATCH_SCORE = 1
MISMATCH_SCORE = -1
VALID_BASES = set("ACGT")


def validate_dna(sequence):
    """Validate that a sequence contains only A, C, G, and T."""
    if not isinstance(sequence, str):
        raise ValueError("DNA sequence must be a string")

    if any(base.upper() not in VALID_BASES for base in sequence):
        raise ValueError("DNA sequence can contain only A, C, G, and T")


def score_bases(base1, base2):
    """Return a score for comparing two DNA bases."""
    validate_dna(base1)
    validate_dna(base2)

    if len(base1) != 1 or len(base2) != 1:
        raise ValueError("Each base must contain exactly one character")

    if base1.upper() == base2.upper():
        return MATCH_SCORE

    return MISMATCH_SCORE


def score_sequences(sequence1, sequence2):
    """Return the total score for two DNA sequences."""
    validate_dna(sequence1)
    validate_dna(sequence2)

    if len(sequence1) != len(sequence2):
        raise ValueError("Sequences must have the same length")

    total_score = 0

    for base1, base2 in zip(sequence1, sequence2):
        total_score += score_bases(base1, base2)

    return total_score