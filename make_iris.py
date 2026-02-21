import csv
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
headers = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]

with open("data/iris.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(X)

print("iris.csv written correctly")