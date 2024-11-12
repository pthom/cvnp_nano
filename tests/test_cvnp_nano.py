import numpy as np
import cvnp_nano_example  # noqa


def t() -> None:
    a = np.eye(3, 4, dtype=np.float32)
    a[0, 1] = 2
    b = cvnp_nano_example.inspect(a)
    print(b)

t()
