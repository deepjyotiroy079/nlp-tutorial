import h5py
with h5py.File('hamlet_lstm_model.h5', 'r') as f:
    print(f.keys())