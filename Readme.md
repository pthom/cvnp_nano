## cvnp_nano: nanobind casts and transformers between numpy and OpenCV, with shared memory

cvnp_nano provides automatic casts between OpenCV matrices and numpy arrays when using nanobind:

* `cv::Mat` and `cv::Mat_<Tp>`: are transformed to numpy array *with shared memory* (i.e. modification to matrices elements made from python are immediately visible to C++, and vice-versa).
* Sub-matrices / non contiguous matrices are not supported: 
  * for numpy arrays, you will need to transform them to a contiguous array before being shared to C++
  * for cv::Mat, you can transform them using `cv::Mat::clone()` before sharing them to python
* Casts *without* shared memory for simple types, between `cv::Size`, `cv::Point`, `cv::Point3`, `cv::Scalar_<Tp>`, `cv::Rect_<Tp>`, `cv::Vec_<Tp>` and python `tuple`
* Casts *without* shared memory between `cv::Matx<Tp, M, N>` and python `tuple[tuple[float]]`.


> Note: for pybind11, see [cvnp](https://github.com/pthom/cvnp)


## How to use it in your project

1. Add cvnp_nano to your project. For example:

```bash
cd external
git submodule add https://github.com/pthom/cvnp_nano.git
```

2. In your module, include cvnp:

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

          // Utilities to trigger value changes made by C++ from python 
          void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
          ...
      };
  """
  o = CvNp_TestHelper()

  #
  # 1. Transform cv::Mat and sub-matrices into numpy arrays / check that reference counts are handled correctly
  #
  # Transform the cv::Mat m10 into a linked numpy array (with shared memory) and assert that m10 now has 2 references
  m10: np.ndarray = o.m10

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

Run:

```
python tests/test_cvnp_nano.py
```


## Notes

This code is intended to be integrated into your own pip package. As such, no pip tooling is provided.
