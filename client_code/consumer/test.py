from dolores.data import load_dataset

dataset = load_dataset("../../data/Iris.csv", target="Species")

train, test = dataset.split()

print(dataset.y)
