import pickle

a = 1

with open('a', 'wb') as f:
    pickle.dump(a, f)