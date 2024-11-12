import numpy as np
import cvnp_nano_example  # noqa


def test_CvNp_TestHelper() -> None:
    o = cvnp_nano_example.CvNp_TestHelper()
    print(o.m)


def t() -> None:
    a = np.eye(3, 4, dtype=np.float32)
    a[0, 1] = 2
    b = cvnp_nano_example.inspect(a)
    print(b)



# test_CvNp_TestHelper()
t()
