
import numpy as np


def test(x):
    x += np.ones(3)

x = np.zeros(3)

test(x)
print(x)
