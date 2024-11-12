import numpy as np
import cvnp_nano_example  # noqa
from cvnp_nano_example import CvNp_TestHelper  # noqa
import math


def are_float_close(x: float, y: float):
    return math.fabs(x - y) < 1e-5


def test_mat_shared():
    """
    Test cv::Mat
    We are playing with these elements
        cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
        void SetM(int row, int col, uchar v) { m.at<uchar>(row, col) = v; }
    """
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()
    # o.m is a *shared* matrix i.e `cvnp::Mat_shared` in the object
    assert o.m.shape == (3, 4)

    # play with its internal cv::Mat

    # From python, change value in the C++ Mat (o.m) and assert that the changes are applied, and visible from python
    o.m[0, 0] = 2
    assert o.m[0, 0] == 2

    # Make a python linked copy of the C++ Mat, named m_linked.
    # Values of m_mlinked and the C++ mat should change together
    m_linked = o.m
    m_linked[1, 1] = 3
    assert o.m[1, 1] == 3

    # Ask C++ to change a value in the matrix, at (0,0)
    # and verify that m_linked as well as o.m are impacted
    o.SetM(0, 0, 10)
    o.SetM(2, 3, 15)
    assert m_linked[0, 0] == 10
    assert m_linked[2, 3] == 15
    assert o.m[0, 0] == 10
    assert o.m[2, 3] == 15

    # Make a clone of the C++ mat and change a value in it
    # => Make sure that the C++ mat is not impacted
    m_clone = np.copy(o.m)
    m_clone[1, 1] = 18
    assert o.m[1, 1] != 18

    # Change the whole C++ mat, by assigning to it a new matrix of different type and dimension
    # check that the shape has changed, and that values are ok
    new_shape = (3, 4, 2)
    new_type = np.float32
    new_mat = np.zeros(new_shape, new_type)
    new_mat[0, 0, 0] = 42.1
    new_mat[1, 0, 1] = 43.1
    new_mat[0, 1, 1] = 44.1
    o.m = new_mat
    assert o.m.shape == new_shape
    assert o.m.dtype == new_type
    assert are_float_close(o.m[0, 0, 0], 42.1)
    assert are_float_close(o.m[1, 0, 1], 43.1)
    assert are_float_close(o.m[0, 1, 1], 44.1)


def test_mat__shared():
    """
    Test cv::Mat_<Tp>
    We are playing with these elements
        cv::Mat_<uint8_t> m_uint8 = cv::Mat_<uint8_t>::eye(cv::Size(4, 3));
        cv::Mat_<int8_t> m_int8 = cv::Mat_<int8_t>::eye(cv::Size(4, 3));
        cv::Mat_<uint16_t> m_uint16 = cv::Mat_<uint16_t>::eye(cv::Size(4, 3));
        cv::Mat_<int16_t> m_int16 = cv::Mat_<int16_t>::eye(cv::Size(4, 3));
        cv::Mat_<int32_t> m_int32 = cv::Mat_<int32_t>::eye(cv::Size(4, 3));
        cv::Mat_<float> m_float = cv::Mat_<float>::eye(cv::Size(4, 3));
        cv::Mat_<double> m_double = cv::Mat_<double>::eye(cv::Size(4, 3));
        void set_m_double(int row, int col, double v) { m_double(row, col) = v; }
    """
    # CvNp_TestHelper is a test helper object
    o = CvNp_TestHelper()

    # Test 1: shapes and types
    assert o.m_uint8.shape == (3, 4)
    assert o.m_uint8.dtype.name == "uint8"
    assert o.m_int8.shape == (3, 4)
    assert o.m_int8.dtype.name == "int8"

    assert o.m_uint16.shape == (3, 4)
    assert o.m_uint16.dtype.name == "uint16"
    assert o.m_int16.shape == (3, 4)
    assert o.m_int16.dtype.name == "int16"

    assert o.m_int32.shape == (3, 4)
    assert o.m_int32.dtype.name == "int32"

    assert o.m_float.shape == (3, 4)
    assert o.m_float.dtype.name == "float32"

    assert o.m_double.shape == (3, 4)
    assert o.m_double.dtype.name == "float64"

    # Test 2: make a linked python copy, modify it, and assert that changes are visible in C++
    m_uint8 = o.m_uint8 # m_uint8 is a python linked numpy matrix
    m_uint8[1, 2] = 3   # modify the python matrix
    assert(o.m_uint8[1, 2] == 3)  # assert that changes are propagated to C++

    m_int32 = o.m_int32 # m_uint8 is a python linked numpy matrix
    m_int32[1, 2] = 3   # modify the python matrix
    assert(o.m_int32[1, 2] == 3)  # assert that changes are propagated to C++

    m_double = o.m_double
    m_double[1, 2] = 3.5
    assert(o.m_double[1, 2] == 3.5)

    # Test 3: make changes from c++
    o.set_m_double(2, 1, 4.5)
    assert(m_double[2, 1] == 4.5)
    assert(o.m_double[2, 1] == 4.5)



def main():
    test_mat_shared()
    test_mat__shared()


if __name__ == "__main__":
     main()
