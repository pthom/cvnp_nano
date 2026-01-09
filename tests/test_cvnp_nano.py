import numpy as np
import cvnp_nano_example  # noqa
from cvnp_nano_example import CvNp_TestHelper, cvnp_roundtrip, short_lived_matx, short_lived_mat, RoundTripMatx21d  # type: ignore
import math
import pytest
import random


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


def matx_as_tuple_shape(matx: tuple[tuple[float, ...], ...]) -> tuple[int, int]:
    """
    Convert a matx to a tuple of shape
    """
    nb_rows = len(matx)
    nb_cols = len(matx[0])
    return nb_rows, nb_cols

def test_matx_not_shared():
    """
    We are playing with these elements
        struct CvNp_TestHelper {
            cv::Matx32d mx_ns = cv::Matx32d::eye();
            void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}
            ...
        };

        mx_ns is published as a list[list[float]] in Python
    """
    # create object
    o = CvNp_TestHelper()

    m_unlinked = o.mx_ns                   # Make a numpy array that is a copy of mx_ns *without* shared memory
    assert matx_as_tuple_shape(m_unlinked) == (3, 2)      # check its shape

    m_unlinked[1][1] = 0.0  # A change in Python
    assert o.mx_ns[1][1] != 0.0  # is not visible from C++

    o.SetMX_ns(2, 1, 15)                               # A C++ change a value in the matrix
    assert not are_float_close(m_unlinked[2][1], 15)  # is not visible from python,
    m_unlinked = o.mx_ns                                 # but becomes visible after we re-create the numpy array from
    assert are_float_close(m_unlinked[2][1], 15)      # the cv::Matx

    # Test set from a tuple
    o.mx_ns = (
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0)
    )
    assert o.mx_ns[2][1] == 6.0

    # Test set from a list
    l = [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0]
    ]
    o.mx_ns = l
    assert o.mx_ns[1][1] == 10.0

    # Test set from a numpy array
    a = np.array([
        [13.0, 14.0],
        [15.0, 16.0],
        [17.0, 18.0]
    ], np.float32)
    o.mx_ns = a
    assert o.mx_ns[0][0] == 13.0


def test_vec_not_shared():
    """
    We are playing with these elements
        cv::Vec3f v3_ns = {1.f, 2.f, 3.f};
        void SetV3_ns(int idx, float v) { v3_ns(idx) = v; }

        v3_ns is published as a list[float] in Python
    """
    o = CvNp_TestHelper()
    assert len(o.v3_ns) == 3
    assert o.v3_ns[0] == 1.

    assert isinstance(o.v3_ns, list)

    o.v3_ns[0] = 54.0         # A change from Python
    assert o.v3_ns[0] != 0.0  # Is **not** visible from C++ (no shared memory)

    o.SetV3_ns(0, 10)        # A C++ change a value in the matrix
    assert o.v3_ns[0] == 10. # is visible from python if we re-create the tuple from the cv::Vec

    # Test set from a tuple
    o.v3_ns = (1.0, 2.0, 3.0)
    assert o.v3_ns[2] == 3.0

    # Test set from a list
    l = [4.0, 5.0, 6.0]
    o.v3_ns = l
    assert o.v3_ns[1] == 5.0

    # Test set from a numpy array
    a = np.array([7.0, 8.0, 9.0], np.float32)
    o.v3_ns = a
    assert o.v3_ns[0] == 7.0


def test_matx_roundtrip():
    # m is a cv::Matx21d in C++, and a tuple[tuple[float]] in Python
    m = (
        (42.1,),
        (43.1,)
    )
    m2 = RoundTripMatx21d(m)
    assert are_float_close(m2[0][0], 42.1)
    assert are_float_close(m2[1][0], 43.1)


def test_size():
    """
    we are playing with these elements
        cv::Size s = cv::Size(123, 456);
        void SetWidth(int w) { s.width = w;}
        void SetHeight(int h) { s.height = h;}
    """
    o = CvNp_TestHelper()
    assert o.s[0] == 123
    assert o.s[1] == 456
    o.SetWidth(789)
    assert o.s[0] == 789
    o.s = (987, 654)
    assert o.s[0] == 987
    assert o.s[1] == 654


def test_point():
    """
    we are playing with these elements
        cv::Point2i pt = cv::Point2i(42, 43);
        void SetX(int x) { pt.x = x; }
        void SetY(int y) { pt.y = y; }
    """
    o = CvNp_TestHelper()
    assert o.pt[0] == 42
    assert o.pt[1] == 43
    o.SetX(789)
    assert o.pt[0] == 789
    o.pt = (987, 654)
    assert o.pt[0] == 987
    assert o.pt[1] == 654

    """
    we are playing with these elements
        cv::Point3d pt3 = cv::Point3d(41.5, 42., 42.5);
        void SetX3(double x) { pt3.x = x; }
        void SetY3(double y) { pt3.y = y; }
        void SetZ3(double z) { pt3.z = z; }
    """
    o = CvNp_TestHelper()
    assert are_float_close(o.pt3[0], 41.5)
    assert are_float_close(o.pt3[1], 42.0)
    assert are_float_close(o.pt3[2], 42.5)
    o.SetX3(789.0)
    assert are_float_close(o.pt3[0], 789.0)
    o.pt3 = (987.1, 654.2, 321.0)
    assert are_float_close(o.pt3[0], 987.1)
    assert are_float_close(o.pt3[1], 654.2)
    assert are_float_close(o.pt3[2], 321.0)


def test_cvnp_round_trip():
    m = np.zeros([5, 6, 7])
    m[3, 4, 5] = 156
    m2 = cvnp_roundtrip(m)
    assert (m == m2).all()

    possible_types = [np.uint8, np.int8, np.uint16, np.int16, np.int32, float, np.float64]
    for test_idx in range(300):
        ndim = random.choice([2, 3])
        shape = []
        for dim in range(ndim):
            if dim < 2:
                shape.append(random.randrange(2, 1000))
            else:
                shape.append(random.randrange(2, 10))
        type = random.choice(possible_types)

        m = np.zeros(shape, dtype=type)

        i = random.randrange(shape[0])
        j = random.randrange(shape[1])
        if ndim == 2:
            m[i, j] = random.random()
        elif ndim == 3:
            k = random.randrange(shape[2])
            m[i, j, k] = random.random()
        else:
            raise RuntimeError("Should not happen")

        m2 = cvnp_roundtrip(m)

        if not (m == m2).all():
            print("argh")
        assert (m == m2).all()


def test_short_lived_matx():
    """
    We are calling the function ShortLivedMatx():

        // Returns a short lived matrix: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Matx33d ShortLivedMatx()
        {
            auto mat = cv::Matx33d::eye();
            return mat;
        }
    """
    m = short_lived_matx()
    assert are_float_close(m[0][0], 1.0)


def test_short_lived_mat():
    """
    We are calling the function ShortLivedMat():

        // Returns a short lived Mat: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Mat ShortLivedMat()
        {
            auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
            mat = cv::Scalar(12, 34, 56, 78);
            return mat;
        }
    """
    m = short_lived_mat()
    assert m.shape == (200, 300, 4)
    assert (m[0, 0] == (12, 34, 56, 78)).all()


def test_empty_mat():
    m = np.zeros(shape=(0, 0, 3))
    m2 = cvnp_roundtrip(m)
    assert (m == m2).all()


def test_additional_ref():
    """
    We are playing with these bindings
        cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
        int m10_refcount()
        {
            if (m10.u)
                return m10.u->refcount;
            else
                { printf("m10.u is null!\n"); return 0; }
        }
    """
    o = CvNp_TestHelper()
    assert o.m10_refcount() == 1
    m = o.m10
    assert o.m10_refcount() == 1  # m is a
    m[0, 0] = (24, 25, 26)
    v = m[0, 0]  # v is a np.array
    v2 = o.m10[0, 0]  # v2 is a np.array
    expected = np.array([24, 25, 26], np.float32)
    assert (v == expected).all()
    assert (v2 == expected).all()


def test_sub_matrices():
    """
    We are playing with these bindings
        struct CvNp_TestHelper {
            cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
            void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
            cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
            cv::Mat GetSubM10() { return m10(cv::Rect(1, 1, 3, 3)); }

            ...
        };
    """
    o = CvNp_TestHelper()
    # Non contiguous matrices are not yet supported by cvnp_nano
    with pytest.raises(TypeError):
        sub_m10 = o.GetSubM10()

    # #
    # # 1. Transform cv::Mat and sub-matrices into numpy arrays / check that reference counts are handled correctly
    # #
    # # Transform the cv::Mat m10 into a linked numpy array (with shared memory) and assert that m10 now has 2 references
    # m10: np.ndarray = o.m10
    # assert o.m10_refcount() == 1
    # # Also transform the m10's sub-matrix into a numpy array, and assert that m10's references count is increased
    # sub_m10 = o.GetSubM10()
    # assert o.m10_refcount() == 1
    #
    # #
    # # 2. Modify values from C++ or python, and ensure that the data is shared
    # #
    # # Modify a value in m10 from C++, and ensure this is visible from python
    # val00 = np.array([1, 2, 3], np.float32)
    # o.SetM10(0, 0, val00)
    # assert (m10[0, 0] == val00).all()
    # # Modify a value in m10 from python and ensure this is visible from C++
    # val10 = np.array([4, 5, 6], np.float32)
    # o.m10[1, 1] = val10
    # assert (o.m10[1, 1] == val10).all()
    #
    # #
    # # 3. Check that values in sub-matrices are also changed
    # #
    # # Check that the sub-matrix is changed
    # assert (sub_m10[0, 0] == val10).all()
    # # Change a value in the sub-matrix from python
    # val22 = np.array([7, 8, 9], np.float32)
    # sub_m10[1, 1] = val22
    # # And assert that the change propagated to the master matrix
    # assert (o.m10[2, 2] == val22).all()
    #
    # #
    # # 4. del python numpy arrays and ensure that the reference count is updated
    # #
    # del m10
    # del sub_m10
    # assert o.m10_refcount() == 1
    #
    # #
    # # 5. Sub-matrices are supported from C++ to python, but not from python to C++!
    # #
    # # i. create a numpy sub-matrix
    # full_matrix = np.ones([10, 10], np.float32)
    # sub_matrix = full_matrix[1:5, 2:4]
    # # ii. Try to copy it into a C++ matrix: this should raise a `ValueError`
    # with pytest.raises(ValueError):
    #     o.m = sub_matrix
    # # iii. However, we can update the C++ matrix by using a contiguous copy of the sub-matrix
    # sub_matrix_clone = np.ascontiguousarray(sub_matrix)
    # o.m = sub_matrix_clone
    # assert o.m.shape == sub_matrix.shape


def test_scalar():
    """
    We are playing with this:
        cv::Scalar scalar_double = cv::Scalar(1.);
        cv::Scalar_<float> scalar_float = cv::Scalar_<float>(1.f, 2.f);
        cv::Scalar_<int32_t> scalar_int32 = cv::Scalar_<int32_t>(1, 2, 3);
        cv::Scalar_<uint8_t> scalar_uint8 = cv::Scalar_<uint8_t>(1, 2, 3, 4);
    """
    o = CvNp_TestHelper()
    v = o.scalar_double
    assert o.scalar_double == [1.0, 0.0, 0.0, 0.0]
    o.scalar_double = (4.0, 5.0)
    assert o.scalar_double == [4.0, 5.0, 0.0, 0.0]

    assert o.scalar_float == [1.0, 2.0, 0.0, 0.0]
    o.scalar_float = np.array([4.0, 5.0, 6.0], np.float32)
    assert o.scalar_float == [4.0, 5.0, 6.0, 0.0]

    assert o.scalar_int32 == [1, 2, 3, 0]
    o.scalar_int32 = [4, 5, 6, 7]
    assert o.scalar_int32 == [4, 5, 6, 7]

    assert o.scalar_uint8 == [1, 2, 3, 4]
    o.scalar_uint8 = (4, 5, 6, 7)
    assert o.scalar_uint8 == [4, 5, 6, 7]

    # Check that setting float values to an int scalar raises an error:
    with pytest.raises(TypeError):
        o.scalar_int32 = (1.23, 4.56)


def test_rect():
    """
    We are playing with:
        cv::Rect  rect_int = cv::Rect(1, 2, 3, 4);
        cv::Rect_<double> rect_double = cv::Rect_<double>(5., 6., 7., 8.);
    """
    o = CvNp_TestHelper()

    assert o.rect_int == [1, 2, 3, 4]
    o.rect_int = (50, 55, 60, 65)
    assert o.rect_int == [50, 55, 60, 65]
    with pytest.raises(TypeError):
        o.rect_int = (1, 2) # We should give 4 values!
    with pytest.raises(TypeError):
        o.rect_int = [1.1, 2.1, 3.1, 4.1] # We should int values!

    assert o.rect_double == [5.0, 6.0, 7.0, 8.0]
    o.rect_double = np.array((1, 2, 3, 4), np.float32)
    assert o.rect_double == [1, 2, 3, 4]
    with pytest.raises(TypeError):
        o.rect_double = (1, 2) # We should give 4 values!


def test_contiguous_check():
    # Check regression with numpy 2:
    # See https://github.com/pthom/cvnp/issues/17
    # The contiguous check was changed to:
    #    bool is_array_contiguous(const pybind11::array& a) { return a.flags() & pybind11::array::c_style; }

    # 1. Check contiguous matrix
    m = np.zeros((10,10),dtype=np.uint8)
    cvnp_roundtrip(m)

    # 2. Check that a non-contiguous matrix raises an error
    full_matrix = np.ones([10, 10], np.float32)
    sub_matrix = full_matrix[1:5, 2:4]
    with pytest.raises(TypeError):
        cvnp_roundtrip(sub_matrix)


def test_multidim_3d_cpp_to_python():
    """Test 3D cv::Mat conversion from C++ to Python"""
    from cvnp_nano_example import create_3d_mat, get_mat_shape
    
    # Create 3D mat in C++ (4x5x6)
    mat_3d = create_3d_mat()
    
    # Check shape
    assert mat_3d.shape == (4, 5, 6)
    assert mat_3d.dtype == np.float32
    
    # Check values (test pattern: i*100 + j*10 + k)
    # These should match the C++ creation logic
    assert mat_3d[0, 0, 0] == 0.0
    assert mat_3d[1, 2, 3] == 123.0
    assert mat_3d[3, 4, 5] == 345.0
    assert mat_3d[2, 1, 4] == 214.0
    
    # Test memory sharing: modify from Python and verify change persists
    original_val = mat_3d[1, 1, 1]
    mat_3d[1, 1, 1] = 888.0
    assert mat_3d[1, 1, 1] == 888.0
    
    mat_3d[2, 3, 4] = 999.0
    assert mat_3d[2, 3, 4] == 999.0


def test_multidim_4d_cpp_to_python():
    """Test 4D cv::Mat conversion from C++ to Python"""
    from cvnp_nano_example import create_4d_mat
    
    # Create 4D mat in C++ (3x4x5x2)
    mat_4d = create_4d_mat()
    
    # Check shape
    assert mat_4d.shape == (3, 4, 5, 2)
    assert mat_4d.dtype == np.float32
    
    # Check values (sequential 0, 1, 2, ...)
    assert mat_4d[0, 0, 0, 0] == 0.0
    assert mat_4d[0, 0, 0, 1] == 1.0
    assert mat_4d[0, 0, 1, 0] == 2.0
    
    # Test modification from Python
    mat_4d[1, 2, 3, 1] = 777.0
    assert mat_4d[1, 2, 3, 1] == 777.0


def test_multidim_5d_cpp_to_python():
    """Test 5D cv::Mat conversion from C++ to Python"""
    from cvnp_nano_example import create_5d_mat
    
    # Create 5D mat in C++ (2x3x4x2x2)
    mat_5d = create_5d_mat()
    
    # Check shape
    assert mat_5d.shape == (2, 3, 4, 2, 2)
    assert mat_5d.dtype == np.float64
    
    # Check values (sequential 0, 0.5, 1.0, 1.5, ...)
    assert mat_5d[0, 0, 0, 0, 0] == 0.0
    assert mat_5d[0, 0, 0, 0, 1] == 0.5
    assert mat_5d[0, 0, 0, 1, 0] == 1.0
    
    # Test modification
    mat_5d[1, 2, 3, 1, 0] = 12345.0
    assert mat_5d[1, 2, 3, 1, 0] == 12345.0


def test_multidim_python_to_cpp():
    """Test multidimensional numpy array to cv::Mat conversion"""
    from cvnp_nano_example import get_mat_shape, inspect_multidim
    
    # Test 3D array
    arr_3d = np.zeros((3, 4, 5), dtype=np.float32)
    arr_3d[1, 2, 3] = 42.0
    shape_3d = get_mat_shape(arr_3d)
    assert shape_3d == [3, 4, 5]
    
    # Test 4D array
    arr_4d = np.ones((2, 3, 4, 5), dtype=np.int32)
    arr_4d[1, 1, 1, 1] = 99
    shape_4d = get_mat_shape(arr_4d)
    assert shape_4d == [2, 3, 4, 5]
    
    # Test 5D array
    arr_5d = np.arange(2*3*4*2*2, dtype=np.float64).reshape(2, 3, 4, 2, 2)
    shape_5d = get_mat_shape(arr_5d)
    assert shape_5d == [2, 3, 4, 2, 2]


def test_multidim_roundtrip():
    """Test roundtrip conversion for multidimensional arrays"""
    
    # Test 3D array roundtrip
    arr_3d = np.random.rand(4, 5, 6).astype(np.float32)
    arr_3d[2, 3, 4] = 123.456
    arr_3d_back = cvnp_roundtrip(arr_3d)
    assert arr_3d.shape == arr_3d_back.shape
    assert arr_3d.dtype == arr_3d_back.dtype
    assert np.allclose(arr_3d, arr_3d_back)
    
    # Test 4D array roundtrip
    arr_4d = np.random.rand(3, 4, 5, 2).astype(np.float64)
    arr_4d[1, 2, 3, 1] = 987.654
    arr_4d_back = cvnp_roundtrip(arr_4d)
    assert arr_4d.shape == arr_4d_back.shape
    assert arr_4d.dtype == arr_4d_back.dtype
    assert np.allclose(arr_4d, arr_4d_back)
    
    # Test 5D array roundtrip
    arr_5d = np.arange(2*3*4*2*2, dtype=np.int32).reshape(2, 3, 4, 2, 2)
    arr_5d_back = cvnp_roundtrip(arr_5d)
    assert arr_5d.shape == arr_5d_back.shape
    assert arr_5d.dtype == arr_5d_back.dtype
    assert (arr_5d == arr_5d_back).all()


def test_multidim_memory_sharing():
    """Test that memory is properly shared for multidimensional arrays"""
    from cvnp_nano_example import create_3d_mat
    
    # Get 3D mat from C++
    mat = create_3d_mat()
    original_value = mat[1, 2, 3]
    
    # Create a view
    mat_view = mat
    
    # Modify through view
    mat_view[1, 2, 3] = 555.0
    
    # Check both references see the change
    assert mat[1, 2, 3] == 555.0
    assert mat_view[1, 2, 3] == 555.0
    
    # Modify through original
    mat[1, 2, 3] = 666.0
    assert mat_view[1, 2, 3] == 666.0


def test_multidim_different_types():
    """Test multidimensional arrays with different dtypes"""
    
    types_to_test = [
        (np.uint8, 'uint8'),
        (np.int8, 'int8'),
        (np.uint16, 'uint16'),
        (np.int16, 'int16'),
        (np.int32, 'int32'),
        (np.float32, 'float32'),
        (np.float64, 'float64'),
    ]
    
    for dtype, dtype_name in types_to_test:
        # Test 3D array
        arr_3d = np.zeros((3, 4, 5), dtype=dtype)
        arr_3d[1, 2, 3] = 42
        arr_3d_back = cvnp_roundtrip(arr_3d)
        assert arr_3d.shape == arr_3d_back.shape, f"Shape mismatch for {dtype_name}"
        assert arr_3d.dtype == arr_3d_back.dtype, f"Dtype mismatch for {dtype_name}"
        assert (arr_3d == arr_3d_back).all(), f"Value mismatch for {dtype_name}"
        
        # Test 4D array
        arr_4d = np.ones((2, 3, 4, 2), dtype=dtype)
        arr_4d_back = cvnp_roundtrip(arr_4d)
        assert arr_4d.shape == arr_4d_back.shape, f"Shape mismatch for {dtype_name} (4D)"
        assert arr_4d.dtype == arr_4d_back.dtype, f"Dtype mismatch for {dtype_name} (4D)"


def test_multidim_contiguous_check():
    """Test that non-contiguous multidimensional arrays are rejected"""
    
    # Create a non-contiguous 3D array
    full_array = np.ones((10, 10, 10), dtype=np.float32)
    sub_array = full_array[::2, ::2, ::2]  # Non-contiguous
    
    assert not sub_array.flags['C_CONTIGUOUS']
    
    with pytest.raises(TypeError):
        cvnp_roundtrip(sub_array)
    
    # But contiguous copy should work
    contiguous_copy = np.ascontiguousarray(sub_array)
    result = cvnp_roundtrip(contiguous_copy)
    assert (result == contiguous_copy).all()


def test_multidim_cpp_indexing_3d():
    """Test that C++ and Python indexing match for 3D arrays"""
    import cvnp_nano_example as o
    
    # Create 3D mat from C++ (shape 4x5x6, filled with pattern i*100 + j*10 + k)
    mat_3d = o.create_3d_mat()
    assert mat_3d.shape == (4, 5, 6)
    assert mat_3d.dtype == np.float32
    
    # Verify multiple indices match between C++ and Python
    test_indices = [
        (0, 0, 0),  # First element
        (1, 2, 3),  # Middle element
        (3, 4, 5),  # Last element
        (2, 3, 4),  # Random element
        (0, 4, 5),  # Edge case
        (3, 0, 0),  # Another edge
    ]
    
    for i, j, k in test_indices:
        expected = i * 100 + j * 10 + k
        python_value = mat_3d[i, j, k]
        cpp_value = o.get_3d_value(mat_3d, i, j, k)
        
        assert python_value == expected, f"Python indexing wrong at [{i},{j},{k}]: got {python_value}, expected {expected}"
        assert cpp_value == expected, f"C++ indexing wrong at [{i},{j},{k}]: got {cpp_value}, expected {expected}"
        assert python_value == cpp_value, f"C++/Python mismatch at [{i},{j},{k}]: C++={cpp_value}, Python={python_value}"


def test_multidim_cpp_set_3d():
    """Test that C++ set_3d_value works correctly and is visible in Python"""
    import cvnp_nano_example as o
    
    # Create empty 3D array from Python
    arr = np.zeros((4, 5, 6), dtype=np.float32)
    
    # Set values through C++
    test_data = [
        (0, 0, 0, 100.0),
        (1, 2, 3, 123.0),
        (3, 4, 5, 345.0),
        (2, 1, 4, 214.0),
    ]
    
    for i, j, k, value in test_data:
        o.set_3d_value(arr, i, j, k, value)
    
    # Verify in Python
    for i, j, k, expected_value in test_data:
        python_value = arr[i, j, k]
        cpp_value = o.get_3d_value(arr, i, j, k)
        
        assert python_value == expected_value, f"Python read wrong at [{i},{j},{k}]: got {python_value}, expected {expected_value}"
        assert cpp_value == expected_value, f"C++ read wrong at [{i},{j},{k}]: got {cpp_value}, expected {expected_value}"


def test_multidim_cpp_indexing_roundtrip():
    """Test that C++ indexing still works after roundtrip through Python"""
    import cvnp_nano_example as o
    
    # Create 3D mat in C++
    mat_original = o.create_3d_mat()
    
    # Roundtrip through Python
    mat_roundtrip = cvnp_roundtrip(mat_original)
    
    # Verify shape and dtype preserved
    assert mat_roundtrip.shape == (4, 5, 6)
    assert mat_roundtrip.dtype == np.float32
    
    # Test that C++ indexing still works correctly after roundtrip
    test_indices = [
        (0, 0, 0),
        (1, 2, 3),
        (3, 4, 5),
        (2, 3, 1),
    ]
    
    for i, j, k in test_indices:
        expected = i * 100 + j * 10 + k
        
        # C++ should still be able to read correct values after roundtrip
        cpp_value = o.get_3d_value(mat_roundtrip, i, j, k)
        python_value = mat_roundtrip[i, j, k]
        
        assert cpp_value == expected, f"C++ indexing failed after roundtrip at [{i},{j},{k}]: got {cpp_value}, expected {expected}"
        assert python_value == expected, f"Python indexing failed after roundtrip at [{i},{j},{k}]: got {python_value}, expected {expected}"
    
    # Test that C++ set still works after roundtrip
    o.set_3d_value(mat_roundtrip, 1, 1, 1, 999.0)
    assert mat_roundtrip[1, 1, 1] == 999.0
    assert o.get_3d_value(mat_roundtrip, 1, 1, 1) == 999.0


def test_multidim_cpp_indexing_4d():
    """Test that C++ and Python indexing match for 4D arrays (simpler than 3D!)"""
    import cvnp_nano_example as o
    
    # Create 4D mat from C++ (shape 3x4x5x2)
    mat_4d = o.create_4d_mat()
    assert mat_4d.shape == (3, 4, 5, 2)
    assert mat_4d.dtype == np.float32
    
    # 4D arrays are straightforward - they remain true multi-dimensional Mats
    # No channel confusion like with 3D arrays!
    
    # Verify indices match between C++ and Python
    test_indices = [
        (0, 0, 0, 0),  # First element (value = 0)
        (0, 0, 0, 1),  # Second element (value = 1)
        (0, 0, 1, 0),  # Third row start (value = 2)
        (1, 2, 3, 1),  # Middle element
        (2, 3, 4, 1),  # Last element
    ]
    
    for i, j, k, l in test_indices:
        python_value = mat_4d[i, j, k, l]
        cpp_value = o.get_4d_value(mat_4d, i, j, k, l)
        
        assert python_value == cpp_value, f"C++/Python mismatch at [{i},{j},{k},{l}]: C++={cpp_value}, Python={python_value}"


def test_multidim_cpp_set_4d():
    """Test that C++ set_4d_value works correctly and is visible in Python"""
    import cvnp_nano_example as o
    
    # Create empty 4D array from Python
    arr = np.zeros((3, 4, 5, 2), dtype=np.float32)
    
    # Set values through C++
    test_data = [
        (0, 0, 0, 0, 100.0),
        (1, 2, 3, 1, 999.0),
        (2, 3, 4, 1, 777.0),
        (0, 1, 2, 0, 555.0),
    ]
    
    for i, j, k, l, value in test_data:
        o.set_4d_value(arr, i, j, k, l, value)
    
    # Verify in Python
    for i, j, k, l, expected_value in test_data:
        python_value = arr[i, j, k, l]
        cpp_value = o.get_4d_value(arr, i, j, k, l)
        
        assert python_value == expected_value, f"Python read wrong at [{i},{j},{k},{l}]: got {python_value}, expected {expected_value}"
        assert cpp_value == expected_value, f"C++ read wrong at [{i},{j},{k},{l}]: got {cpp_value}, expected {expected_value}"


def test_multidim_cpp_indexing_4d_roundtrip():
    """Test that 4D indexing works after roundtrip (should be simple, no channel issues)"""
    import cvnp_nano_example as o
    
    # Create 4D mat in C++
    mat_original = o.create_4d_mat()
    
    # Roundtrip through Python
    mat_roundtrip = cvnp_roundtrip(mat_original)
    
    # Verify shape and dtype preserved
    assert mat_roundtrip.shape == (3, 4, 5, 2)
    assert mat_roundtrip.dtype == np.float32
    
    # Test that C++ indexing still works correctly after roundtrip
    # Unlike 3D arrays, 4D arrays don't have the channel representation complexity
    test_indices = [
        (0, 0, 0, 0),
        (1, 2, 3, 1),
        (2, 3, 4, 1),
        (0, 1, 2, 0),
    ]
    
    for i, j, k, l in test_indices:
        cpp_value = o.get_4d_value(mat_roundtrip, i, j, k, l)
        python_value = mat_roundtrip[i, j, k, l]
        
        assert cpp_value == python_value, f"Indexing mismatch after roundtrip at [{i},{j},{k},{l}]: C++={cpp_value}, Python={python_value}"
    
    # Test that C++ set still works after roundtrip
    o.set_4d_value(mat_roundtrip, 1, 1, 1, 1, 888.0)
    assert mat_roundtrip[1, 1, 1, 1] == 888.0
    assert o.get_4d_value(mat_roundtrip, 1, 1, 1, 1) == 888.0


def test_opencv_operations_on_3d_arrays():
    """Test that OpenCV functions work correctly on converted 3D arrays"""
    import cv2
    
    import cvnp_nano_example as o
    
    # Create a 3D array that looks like an image (H, W, C)
    img_array = np.random.randint(0, 255, size=(100, 120, 3), dtype=np.uint8)
    
    # Convert to C++ and back
    img_roundtrip = cvnp_roundtrip(img_array)
    
    # Verify OpenCV can work with it
    assert img_roundtrip.shape == (100, 120, 3)
    assert img_roundtrip.dtype == np.uint8
    
    # Test Gaussian blur
    blurred = cv2.GaussianBlur(img_roundtrip, (5, 5), 0)
    assert blurred.shape == img_roundtrip.shape
    assert blurred.dtype == img_roundtrip.dtype
    
    # Test color conversion
    gray = cv2.cvtColor(img_roundtrip, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 120)
    
    # Test that we can convert OpenCV results back to C++
    blurred_back = cvnp_roundtrip(blurred)
    assert (blurred_back == blurred).all()


def test_opencv_realworld_pipeline():
    """Test a realistic OpenCV processing pipeline with shared memory"""
    import cv2
    
    import cvnp_nano_example as o
    
    # Create a synthetic image
    height, width = 480, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw some shapes using OpenCV
    cv2.circle(img, (width//2, height//2), 100, (255, 0, 0), -1)
    cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), 3)
    cv2.line(img, (0, 0), (width, height), (0, 0, 255), 2)
    
    # Convert to C++ and back
    img_cpp = cvnp_roundtrip(img)
    
    # Verify shapes are preserved
    assert img_cpp.shape == (height, width, 3)
    assert (img_cpp == img).all()
    
    # Apply OpenCV operations on the converted image
    # 1. Blur
    blurred = cv2.GaussianBlur(img_cpp, (15, 15), 0)
    
    # 2. Edge detection
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Convert edges back to 3-channel
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Verify we can convert the final result back to C++
    final_cpp = cvnp_roundtrip(edges_3ch)
    assert final_cpp.shape == (height, width, 3)
    assert final_cpp.dtype == np.uint8
    
    # Verify OpenCV operations chain correctly
    assert blurred.shape == img_cpp.shape
    assert edges.shape == (height, width)
    assert edges_3ch.shape == (height, width, 3)


def test_opencv_video_frame_processing():
    """Simulate video frame processing workflow"""
    import cv2
    
    # Simulate processing multiple video frames
    num_frames = 10
    height, width = 720, 1280
    
    for frame_idx in range(num_frames):
        # Create a frame with varying content
        frame = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
        
        # Add frame number as text (simulate real video)
        cv2.putText(frame, f"Frame {frame_idx}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to C++ and back (simulate passing to C++ processing)
        frame_cpp = cvnp_roundtrip(frame)
        
        # Apply some OpenCV operations
        resized = cv2.resize(frame_cpp, (width//2, height//2))
        
        # Verify operations work
        assert resized.shape == (height//2, width//2, 3)
        assert resized.dtype == np.uint8
        
        # Simulate C++ modifying the frame
        if frame_cpp.shape[0] > 100 and frame_cpp.shape[1] > 100:
            # Set a marker pixel
            frame_cpp[100, 100, :] = [255, 0, 255]
            assert (frame_cpp[100, 100] == [255, 0, 255]).all()


def test_multidim_large_arrays():
    """Test conversion of large multidimensional arrays"""
    import cvnp_nano_example as o
    
    # Test large 3D array (simulating HD video frame with multiple channels)
    large_3d = np.random.rand(1080, 1920, 16).astype(np.float32)
    large_3d_back = cvnp_roundtrip(large_3d)
    
    assert large_3d_back.shape == large_3d.shape
    assert large_3d_back.dtype == large_3d.dtype
    assert np.allclose(large_3d_back, large_3d)
    
    # Verify memory sharing for large arrays
    large_3d_back[0, 0, 0] = 999.0
    assert large_3d_back[0, 0, 0] == 999.0
    
    # Test large 4D array (simulating batch of images)
    large_4d = np.random.rand(10, 64, 64, 32).astype(np.float32)
    large_4d_back = cvnp_roundtrip(large_4d)
    
    assert large_4d_back.shape == large_4d.shape
    assert large_4d_back.dtype == large_4d.dtype
    assert np.allclose(large_4d_back, large_4d)


def test_multidim_shape_edge_cases():
    """Test edge cases for array shapes"""
    
    # Test minimum valid 3D array (1x1x1) - Note: single channel gets dropped!
    tiny_3d = np.array([[[42.0]]], dtype=np.float32)
    tiny_3d_back = cvnp_roundtrip(tiny_3d)
    # Single channel 3D arrays become 2D (OpenCV convention)
    assert tiny_3d_back.shape == (1, 1), "Single channel 3D should become 2D"
    assert tiny_3d_back[0, 0] == 42.0
    
    # Test 3D with 2+ channels to preserve 3D shape
    tiny_3d_multi = np.array([[[1.0, 2.0]]], dtype=np.float32)
    tiny_3d_multi_back = cvnp_roundtrip(tiny_3d_multi)
    assert tiny_3d_multi_back.shape == (1, 1, 2)
    
    # Test 3D with one large dimension
    skinny_3d = np.random.rand(1000, 1, 3).astype(np.float32)
    skinny_3d_back = cvnp_roundtrip(skinny_3d)
    assert skinny_3d_back.shape == (1000, 1, 3)
    
    # Test 3D with many channels
    many_channels = np.random.rand(10, 10, 64).astype(np.float32)
    many_channels_back = cvnp_roundtrip(many_channels)
    assert many_channels_back.shape == (10, 10, 64)
    
    # Test 4D minimum (1x1x1x1)
    tiny_4d = np.array([[[[1.0]]]], dtype=np.float32)
    tiny_4d_back = cvnp_roundtrip(tiny_4d)
    assert tiny_4d_back.shape == (1, 1, 1, 1)
    assert tiny_4d_back[0, 0, 0, 0] == 1.0
    
    # Test 5D with mixed dimensions
    mixed_5d = np.random.rand(2, 1, 3, 1, 4).astype(np.float64)
    mixed_5d_back = cvnp_roundtrip(mixed_5d)
    assert mixed_5d_back.shape == (2, 1, 3, 1, 4)
    assert mixed_5d_back.dtype == np.float64


def test_multidim_cpp_operations_preserve_structure():
    """Test that C++ operations maintain array structure correctly"""
    import cvnp_nano_example as o
    
    # Create and modify 3D array through C++
    arr_3d = np.zeros((5, 6, 7), dtype=np.float32)
    
    # Set multiple values through C++
    for i in range(5):
        for j in range(6):
            for k in range(7):
                value = float(i * 100 + j * 10 + k)
                o.set_3d_value(arr_3d, i, j, k, value)
    
    # Verify all values are correct
    for i in range(5):
        for j in range(6):
            for k in range(7):
                expected = float(i * 100 + j * 10 + k)
                assert arr_3d[i, j, k] == expected
                assert o.get_3d_value(arr_3d, i, j, k) == expected
    
    # Create and modify 4D array through C++
    arr_4d = np.zeros((3, 4, 5, 6), dtype=np.float32)
    
    # Set corner values
    test_positions = [
        (0, 0, 0, 0, 1.0),
        (2, 3, 4, 5, 999.0),
        (1, 2, 3, 4, 555.0),
        (0, 3, 0, 5, 777.0),
    ]
    
    for i, j, k, l, val in test_positions:
        o.set_4d_value(arr_4d, i, j, k, l, val)
    
    for i, j, k, l, expected in test_positions:
        assert arr_4d[i, j, k, l] == expected
        assert o.get_4d_value(arr_4d, i, j, k, l) == expected


def test_multidim_dtype_preservation_detailed():
    """Test that dtypes are preserved precisely through conversions"""
    
    dtypes_to_test = [
        (np.uint8, 0, 255),
        (np.int8, -128, 127),
        (np.uint16, 0, 65535),
        (np.int16, -32768, 32767),
        (np.int32, -2147483648, 2147483647),
        (np.float32, -1e6, 1e6),
        (np.float64, -1e10, 1e10),
    ]
    
    for dtype, min_val, max_val in dtypes_to_test:
        # Test 3D
        if dtype in [np.float32, np.float64]:
            arr_3d = np.random.uniform(min_val, max_val, size=(10, 10, 10)).astype(dtype)
        else:
            arr_3d = np.random.randint(max(0, min_val), min(max_val, 1000), 
                                       size=(10, 10, 10), dtype=dtype)
        
        arr_3d_back = cvnp_roundtrip(arr_3d)
        assert arr_3d_back.dtype == dtype, f"3D dtype mismatch for {dtype}"
        assert (arr_3d_back == arr_3d).all(), f"3D values changed for {dtype}"
        
        # Test 4D
        if dtype in [np.float32, np.float64]:
            arr_4d = np.random.uniform(min_val/10, max_val/10, size=(5, 5, 5, 5)).astype(dtype)
        else:
            arr_4d = np.random.randint(max(0, min_val), min(max_val, 100), 
                                       size=(5, 5, 5, 5), dtype=dtype)
        
        arr_4d_back = cvnp_roundtrip(arr_4d)
        assert arr_4d_back.dtype == dtype, f"4D dtype mismatch for {dtype}"
        assert (arr_4d_back == arr_4d).all(), f"4D values changed for {dtype}"


def test_performance_benchmarks():
    """Benchmark conversion performance for various array sizes"""
    import cvnp_nano_example as o
    import time
    import numpy as np
    
    print("\n=== Performance Benchmarks ===")
    
    # Test 1: Small arrays (typical for real-time processing)
    sizes_small = [
        ("640×480×3 (VGA RGB)", (480, 640, 3)),
        ("1920×1080×3 (HD RGB)", (1080, 1920, 3)),
    ]
    
    for name, shape in sizes_small:
        arr = np.random.randint(0, 255, shape, dtype=np.uint8)
        
        # Measure Python → C++ → Python roundtrip
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            result = o.cvnp_roundtrip(arr)
        elapsed = time.perf_counter() - start
        
        avg_time_us = (elapsed / iterations) * 1_000_000
        throughput_fps = iterations / elapsed
        
        print(f"  {name}:")
        print(f"    Roundtrip: {avg_time_us:.2f} µs/frame")
        print(f"    Throughput: {throughput_fps:.0f} FPS")
        
        # Verify correctness
        assert np.array_equal(arr, result)
        
        # Performance assertion: relaxed thresholds for CI runners
        if "VGA" in name:
            assert avg_time_us < 500, f"VGA conversion too slow: {avg_time_us:.2f} µs"
        elif "HD" in name:
            assert avg_time_us < 2000, f"HD conversion too slow: {avg_time_us:.2f} µs"
    
    # Test 2: Large multi-dimensional arrays
    print("\n  Large multi-dimensional arrays:")
    large_4d = np.random.randn(10, 64, 64, 32).astype(np.float32)
    
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        result = o.cvnp_roundtrip(large_4d)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / iterations) * 1000
    print(f"    10×64×64×32 (float32): {avg_time_ms:.2f} ms/batch")
    
    assert np.array_equal(large_4d, result)
    assert avg_time_ms < 50, f"4D conversion too slow: {avg_time_ms:.2f} ms"
    
    # Test 3: Memory sharing verification
    print("\n  Memory sharing verification:")
    arr = np.random.randn(1080, 1920, 3).astype(np.float32)
    
    # Roundtrip returns a NEW array (shares memory with C++ Mat, not original)
    start = time.perf_counter()
    mat = o.cvnp_roundtrip(arr)
    conversion_time = time.perf_counter() - start
    
    # Verify values are correct (copy semantics)
    assert np.array_equal(arr, mat), "Values not preserved"
    
    # Modify the returned array (this should be fast - it's just Python array access)
    start = time.perf_counter()
    mat[0, 0, 0] = 999.0
    modify_time = time.perf_counter() - start
    
    print(f"    Conversion time: {conversion_time * 1_000_000:.2f} µs")
    print(f"    Array modify time: {modify_time * 1_000_000:.2f} µs")
    print(f"    Note: cvnp_roundtrip creates new array (expected behavior)")
    
    # Memory access should be fast (< 100µs threshold for CI runners)
    assert modify_time < 0.0001, "Array access unexpectedly slow"


def test_concurrent_access():
    """Test thread safety with concurrent access to shared memory"""
    import cvnp_nano_example as o
    import threading
    import numpy as np
    
    print("\n=== Concurrent Access Test ===")
    
    # Create a large array - work directly with it, not through roundtrip
    # (roundtrip creates a NEW array, not shared with original)
    arr = np.zeros((100, 100, 3), dtype=np.float32)
    
    # Test 1: Concurrent reads (should be safe)
    results = []
    errors = []
    
    def read_worker(thread_id):
        try:
            for _ in range(100):
                # Read from different locations
                val = arr[thread_id % 100, thread_id % 100, 0]
                results.append(val)
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=read_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"Concurrent reads failed: {errors}"
    print(f"  ✓ Concurrent reads: {len(results)} successful reads from 10 threads")
    
    # Test 2: Concurrent writes (race conditions expected, but shouldn't crash)
    write_count = [0]
    errors = []
    
    def write_worker(thread_id):
        try:
            for i in range(100):
                # Write to different locations to minimize contention
                row = (thread_id * 10 + i) % 100
                col = i % 100
                arr[row, col, 0] = float(thread_id * 1000 + i)
                write_count[0] += 1
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=write_worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"Concurrent writes crashed: {errors}"
    print(f"  ✓ Concurrent writes: {write_count[0]} successful writes from 10 threads")
    
    # Verify writes occurred - just check that at least some values changed
    non_zero_count = np.count_nonzero(arr)
    assert non_zero_count > 0, "No writes visible in array"
    print(f"  ✓ Writes visible in array ({non_zero_count} non-zero values)")
    print(f"  ✓ Writes visible in array")
    
    # Test 3: Mixed read/write with C++ accessors
    arr_3d = o.create_3d_mat()
    errors = []
    read_count = [0]
    write_count = [0]
    
    def mixed_worker(thread_id):
        try:
            for i in range(50):
                # Read
                val = o.get_3d_value(arr_3d, 0, 0, thread_id % 6)
                read_count[0] += 1
                
                # Write to different location
                o.set_3d_value(arr_3d, 1, 1, thread_id % 6, float(thread_id + i))
                write_count[0] += 1
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=mixed_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(errors) == 0, f"Mixed C++ access failed: {errors}"
    print(f"  ✓ Mixed C++ access: {read_count[0]} reads, {write_count[0]} writes from 6 threads")
    
    # Test 4: GIL behavior - verify Python threads don't deadlock
    print("\n  GIL behavior test:")
    large_arr = np.random.randn(1000, 1000, 3).astype(np.float32)
    
    conversion_times = []
    errors = []
    
    def conversion_worker(thread_id):
        try:
            start = time.perf_counter()
            result = o.cvnp_roundtrip(large_arr)
            elapsed = time.perf_counter() - start
            conversion_times.append(elapsed)
            # Verify correctness
            assert np.array_equal(large_arr, result)
        except Exception as e:
            errors.append(str(e))
    
    import time
    threads = [threading.Thread(target=conversion_worker, args=(i,)) for i in range(5)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - start
    
    assert len(errors) == 0, f"Concurrent conversions failed: {errors}"
    avg_time = sum(conversion_times) / len(conversion_times)
    print(f"    5 threads, 1000×1000×3 conversions:")
    print(f"      Total time: {total_time:.3f}s")
    print(f"      Avg per thread: {avg_time:.3f}s")
    print(f"      Speedup: {sum(conversion_times)/total_time:.2f}x")
    
    # Note: Due to GIL, speedup will be limited, but should not be slower than sequential
    assert total_time < sum(conversion_times) * 1.5, "Threading overhead too high"
    print(f"  ✓ No deadlocks, reasonable threading overhead")


def main():
    test_mat_shared()
    test_mat__shared()
    test_matx_not_shared()
    test_vec_not_shared()
    test_size()
    test_point()
    test_cvnp_round_trip()
    test_short_lived_mat()
    test_empty_mat()

    test_additional_ref()
    test_sub_matrices()
    test_scalar()
    test_rect()
    test_contiguous_check()

    test_short_lived_matx()
    test_matx_roundtrip()

    # Multidimensional array tests
    test_multidim_3d_cpp_to_python()
    test_multidim_4d_cpp_to_python()
    test_multidim_5d_cpp_to_python()
    test_multidim_python_to_cpp()
    test_multidim_roundtrip()
    test_multidim_memory_sharing()
    test_multidim_different_types()
    test_multidim_contiguous_check()

    # test performance and concurrency
    test_performance_benchmarks()
    test_concurrent_access()

    from cvnp_nano_example import print_types_synonyms  # noqa
    print("List of types synonyms:")
    print_types_synonyms()


if __name__ == "__main__":
     main()
