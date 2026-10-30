from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
    stratify=iris.target
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

artifact_dir = Path("artifacts")
artifact_dir.mkdir(exist_ok=True)

model_path = artifact_dir / "iris_model.joblib"
joblib.dump(model, model_path)

print(f"Model training completed. Accuracy: {accuracy:.4f}")
print(f"Model saved to: {model_path}")