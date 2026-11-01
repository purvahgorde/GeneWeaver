
"""Basic PAM detection for SpCas9 NGG motifs."""


def find_pam_sites(sequence):
    """Find NGG PAM motifs in a DNA sequence.

    Returns a list of dictionaries containing each PAM and its
    zero-based start position.
    """
    sequence = sequence.upper()
    pam_sites = []

    for i in range(len(sequence) - 2):
        motif = sequence[i:i + 3]

        if motif[1:] == "GG" and motif[0] in "ACGT":
            pam_sites.append({
                "position": i,
                "pam": motif,
            })

    return pam_sites