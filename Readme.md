## cvnp_nano: nanobind casts and transformers between numpy and OpenCV, with shared memory

cvnp_nano provides automatic casts between OpenCV matrices and numpy arrays when using nanobind:

* `cv::Mat`, `cv::Mat_<Tp>`, `cv::Matx<ScalarType, M, N>`, `cv::Vec<ScalarType, N>`: are transformed to numpy array *with shared memory* (i.e. modification to matrices elements made from python are immediately visible to C++, and vice-versa).
* Sub-matrices / non contiguous matrices are not supported: 
  * for numpy arrays, you will need to transform them to a contiguous array before being shared to C++
  * for cv::Mat, you can transform them using `cv::Mat::clone()` before sharing them to python
* Casts *without* shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::Point3`, `cv::Scalar_<Tp>`, `cv::Rect_<Tp>` and python `tuple`


## How to use it in your project

1. Add cvnp_nano to your project. For example:

```bash
cd external
git submodule add https://github.com/pthom/cvnp_nano.git
```

2. Link it to your python module:

Either using the provided CMakeLists:

```cmake
add_subdirectory(path/to/cvnp_nano)
target_link_libraries(your_target PRIVATE cvnp_nano)
```

Or, just add `cvnp_nano/cvnp_nano.cpp` and `cvnp_nano/cvnp_nano.h` to your sources.


3. In your module, include cvnp:

```cpp
#include "cvnp_nano/cvnp_nano.h"
```



### Demo with cv::Mat : shared memory and sub-matrices

Below is on extract from the test [test/test_cvnp_nano.py](tests/test_cvnp_nano.py):

```python
def test_cpp_sub_matrices():
  """
  We are playing with these bindings:
      struct CvNp_TestHelper {
          // m10 is a cv::Mat with 3 float channels
          cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
          // GetSubM10 returns a sub-matrix of m10
          cv::Mat GetSubM10() { return m10(cv::Rect(1, 1, 3, 3)); }
          // Utilities to trigger value changes made by C++ from python 
          void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
          cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
          ...
      };
  """
  o = CvNp_TestHelper()

  #
  # 1. Transform cv::Mat and sub-matrices into numpy arrays / check that reference counts are handled correctly
  #
  # Transform the cv::Mat m10 into a linked numpy array (with shared memory) and assert that m10 now has 2 references
  m10: np.ndarray = o.m10
  assert o.m10_refcount() == 2
  # Also transform the m10's sub-matrix into a numpy array, and assert that m10's references count is increased
  sub_m10 = o.GetSubM10()
  assert o.m10_refcount() == 3

  #
  # 2. Modify values from C++ or python, and ensure that the data is shared
  #
  # Modify a value in m10 from C++, and ensure this is visible from python
  val00 = np.array([1, 2, 3], np.float32)
  o.SetM10(0, 0, val00)
  assert (m10[0, 0] == val00).all()
  # Modify a value in m10 from python and ensure this is visible from C++
  val10 = np.array([4, 5, 6], np.float32)
  o.m10[1, 1] = val10
  assert (o.m10[1, 1] == val10).all()

  #
  # 3. Check that values in sub-matrices are also changed
  #
  # Check that the sub-matrix is changed
  assert (sub_m10[0, 0] == val10).all()
  # Change a value in the sub-matrix from python
  val22 = np.array([7, 8, 9], np.float32)
  sub_m10[1, 1] = val22
  # And assert that the change propagated to the master matrix
  assert (o.m10[2, 2] == val22).all()

  #
  # 4. del python numpy arrays and ensure that the reference count is updated
  #
  del m10
  del sub_m10
  assert o.m10_refcount() == 1

  #
  # 5. Sub-matrices are supported from C++ to python, but not from python to C++!
  #
  # i. create a numpy sub-matrix
  full_matrix = np.ones([10, 10], np.float32)
  sub_matrix = full_matrix[1:5, 2:4]
  # ii. Try to copy it into a C++ matrix: this should raise a `ValueError`
  with pytest.raises(ValueError):
    o.m = sub_matrix
  # iii. However, we can update the C++ matrix by using a contiguous copy of the sub-matrix
  sub_matrix_clone = np.ascontiguousarray(sub_matrix)
  o.m = sub_matrix_clone
  assert o.m.shape == sub_matrix.shape
```

## Build and test

_These steps are only for development and testing of this package, they are not required in order to use it in a different project._

### install python dependencies (opencv-python, pytest, numpy)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install C++ dependencies (pybind11, OpenCV)

You will need to have `OpenCV` installed on your system (you can use `vcpkg` or your package manager).

### Build

You need to specify the path to the python executable:

```bash
mkdir build && cd build
cmake .. -DPython_EXECUTABLE=../venv/bin/python
make
```

### Test

In the build dir, run:

```
cmake --build . --target test
```

(this will run native C++ tests and python tests)

## Notes

Thanks to Dan Mašek who gave me some inspiration here:
https://stackoverflow.com/questions/60949451/how-to-send-a-cvmat-to-python-over-shared-memory

This code is intended to be integrated into your own pip package. As such, no pip tooling is provided.
