from dolores.data import load_dataset
from dolores.experiment import train
from dolores.inference import load_model, save_bundle
from dolores.tracking import log_run
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

EXPERIMENT = "test"
dataset = load_dataset("../../data/Iris.csv", target="Species", drop=["Id"])


print(dataset.features)
print(dataset.y)

train_df, test_df = dataset.split()

candidates = {
    "logreg": LogisticRegression(max_iter=200),
    "rf": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    "gb": GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42),
}

results = [train(est, train_df, test_df, name=n) for n, est in candidates.items()]

for i, result in enumerate(results):
    log_run(
        result,
        experiment=EXPERIMENT,
    )
    print(result.metrics.scalars())
    result.metrics.write_json(f"../../data/metrics{i}.json")


results[0].save_bundle("../../data/iris-classifier.zip")
print(results[0].metrics.confusion_matrix)

loaded_model = load_model("../../data/iris-classifier.zip")
print(type(loaded_model))

# loaded_model.predict()
