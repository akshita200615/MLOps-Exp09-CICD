
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_iris_dataset_shape():
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150

def test_model_prediction_count():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2,
        random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
