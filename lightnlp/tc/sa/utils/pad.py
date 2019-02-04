def pad_sequnce(sequence, seq_length, pad_token='<pad>'):
    padded_seq = sequence[:]
    if len(padded_seq) < seq_length:
        padded_seq.extend([pad_token for i in range(len(padded_seq), seq_length)])
    return padded_seq