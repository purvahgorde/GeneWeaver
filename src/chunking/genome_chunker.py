def chunk_sequence(sequence: str, chunk_size: int) -> list[str]:
    """Split a DNA sequence into chunks of specified length.

    Parameters
    ----------
    sequence : str
        The DNA sequence to chunk.
    chunk_size : int
        Size of each chunk (must be positive).

    Returns
    -------
    list[str]
        List of sequence chunks.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if not isinstance(sequence, str):
        raise TypeError(f"sequence must be a string, got {type(sequence).__name__}")

    chunks = []
    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i : i + chunk_size])

    return chunks
