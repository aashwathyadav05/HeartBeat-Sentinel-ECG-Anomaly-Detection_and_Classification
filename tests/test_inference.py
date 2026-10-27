import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from web.inference import inference_engine


def test_inference():
    sig = np.random.randn(140).astype(np.float32)
    res = inference_engine.predict(sig)
    assert 'segment_mse' in res
    assert 'predicted_label' in res
    assert 'reconstructed' in res
    assert len(res['reconstructed']) == 140
    print("test_inference passed")


if __name__ == "__main__":
    test_inference()
