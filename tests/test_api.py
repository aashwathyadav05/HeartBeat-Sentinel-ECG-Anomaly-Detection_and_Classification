import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from web.app import app


def test_health():
    client = app.test_client()
    r = client.get('/api/health')
    assert r.status_code == 200
    data = r.get_json()
    assert data['status'] == 'ok'
    print("test_health passed")


def test_metrics():
    client = app.test_client()
    r = client.get('/api/metrics')
    assert r.status_code == 200
    print("test_metrics passed")


def test_predict_invalid():
    client = app.test_client()
    r = client.post('/api/predict', json={'signal': [1,2,3]})
    assert r.status_code == 400
    print("test_predict_invalid passed")


if __name__ == "__main__":
    test_health()
    test_metrics()
    test_predict_invalid()
