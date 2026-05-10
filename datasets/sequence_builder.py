import numpy as np

def build_sequences(X, Y, seq_len=5, step=2):
    """
    Build sequences from independent samples using a sliding window.
    Input:
        X: (N, F) features
        Y: (N, 2) labels (x, y)
    Output:
        X_seq: (num_seq, seq_len, F)
        Y_seq: (num_seq, 2) - usually the label of the last frame in the sequence
    """
    num_samples = X.shape[0]
    X_seq = []
    Y_seq = []
    
    for i in range(0, num_samples - seq_len + 1, step):
        seq_x = X[i : i + seq_len]
        # For regression, we typically predict the position of the last sample in the sequence
        seq_y = Y[i + seq_len - 1] 
        
        X_seq.append(seq_x)
        Y_seq.append(seq_y)
        
    return np.array(X_seq), np.array(Y_seq)
