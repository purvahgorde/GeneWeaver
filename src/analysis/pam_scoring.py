
"""Integration of PAM detection and DNA sequence scoring."""

from src.pam.pam_detector import find_pam_sites
from src.scoring.dna_scoring import score_sequences


def score_pam_sites(sequence, target_sequence):
    """Find NGG PAM sites and score each site against a target.

    The three-base PAM motif is compared with the target sequence.
    The target must therefore contain exactly three bases.
    """
    if len(target_sequence) != 3:
        raise ValueError("Target sequence must have exactly 3 bases")

    pam_sites = find_pam_sites(sequence)
    results = []

    for site in pam_sites:
        score = score_sequences(site["pam"], target_sequence)

        results.append({
            "position": site["position"],
            "pam": site["pam"],
            "score": score,
        })

    return results