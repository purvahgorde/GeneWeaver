def chunk_sequence(sequence, chunk_size):
    chunks = []

    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i:i + chunk_size])

    return chunks
