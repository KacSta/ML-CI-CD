import numpy as np
import pytest
from zadanie_4 import app, model, X, y
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predictions_not_none():
    prediction = model.predict([[10]])
    assert prediction is not None, "Predykcja nie powinna być None."

def test_predictions_length():
    test_input = np.array([[60],[70],[80]])
    preds = model.predict(test_input)

    assert len(preds) > 0, "Lista predykcji nie może być pusta."
    assert len(preds) == len(test_input), f"Oczekiwano {len(test_input)} wyników, otrzymano {len(preds)}"

def test_predictions_value_range():
    test_input = np.array([1],[100])
    preds = model.predict(test_input)

    for p in preds:
        assert p > 0, f"Predykcja {p} jest poza oczekiwanym zakresem."

def test_model_accuracy():
    score = model.score(X,y)
    min_acceptable_score = 0.95
    assert score >= min_acceptable_score, f"Dokładność modelu ({score}) jest poniżej progu {min_acceptable_score}"