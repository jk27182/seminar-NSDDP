import pickle

with open("data.pckl", "rb") as f:
    data = pickle.load(f)

print(type(data))