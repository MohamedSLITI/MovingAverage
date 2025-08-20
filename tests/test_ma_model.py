import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ma_model import fit_ma_model, walk_forward_validation
import pandas as pd
import numpy as np


def test_fit_ma_model():
    # Generate synthetic MA(2) data
    np.random.seed(42)
    e = np.random.normal(size=100)
    series = pd.Series(e[:-2] + 0.5 * e[1:-1] + 0.3 * e[2:])
    model = fit_ma_model(series, q=2)
    assert model is not None
    assert len(model.params) == 3  # const + 2 MA terms


def test_walk_forward_validation():
    np.random.seed(42)
    series = pd.Series(np.random.normal(size=50))
    pred = walk_forward_validation(series, q=2)
    assert len(pred) == int(0.2 * 50)  # predictions length = test set size
