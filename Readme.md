## cvnp_nano: nanobind casts and transformers between numpy and OpenCV, with shared memory

cvnp_nano provides automatic casts between OpenCV matrices and numpy arrays when using nanobind:

* `cv::Mat` and `cv::Mat_<Tp>`: are transformed to numpy array *with shared memory* (i.e. modification to matrices elements made from python are immediately visible to C++, and vice-versa).
* **Multidimensional arrays** (dims > 2) are supported with shared memory. See the [Multidimensional Arrays](#multidimensional-arrays-dims--2) section for important details about 3D vs 4D+ arrays.
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


## Multidimensional Arrays (dims > 2)

cvnp_nano supports multidimensional arrays with shared memory, but there's an important distinction between 3D and 4D+ arrays due to OpenCV's channel representation:

### 3D Arrays: Channel Representation

**3D NumPy arrays are converted to 2D OpenCV Mats with channels** to maintain compatibility with traditional image processing workflows (e.g., RGB images):

- A `(H, W, C)` NumPy array becomes a 2D `cv::Mat` with:
  - `mat.rows = H`
  - `mat.cols = W`
  - `mat.channels() = C`
  - `mat.dims = 2` (not 3!)

**Accessing elements:** Use `mat.ptr<T>(i, j)[k]` to access element at position `(i, j, k)`. The `ptr<T>(i, j)` returns a pointer to row `i`, column `j`, and `[k]` indexes into the `k`-th channel. See [example/cvnp_nano_example.cpp](example/cvnp_nano_example.cpp) functions `get_3d_value()` and `set_3d_value()` for reference.

### 4D+ Arrays: True Multidimensional Mats

**4D and higher dimensional NumPy arrays become true multidimensional OpenCV Mats:**

- A `(D1, D2, D3, D4)` NumPy array becomes a 4D `cv::Mat` with:
  - `mat.dims = 4`
  - `mat.size[0] = D1`, `mat.size[1] = D2`, `mat.size[2] = D3`, `mat.size[3] = D4`
  - `mat.channels() = 1`

**Simpler:** Indexing is straightforward - you can use OpenCV's `mat.at<T>(int* idx)` directly! See [example/cvnp_nano_example.cpp](example/cvnp_nano_example.cpp) functions `get_4d_value()` and `set_4d_value()` for reference.

### Example: Accessing Multidimensional Array Elements

Both 3D and 4D+ arrays have clean, simple access patterns:

```cpp
// 3D array indexing (2D Mat with channels - use mat.ptr())
float get_3d_value(const cv::Mat& mat, int i, int j, int k) {
    return mat.ptr<float>(i, j)[k];  // Get pointer to (i,j), index [k] for channel
}

// 4D array indexing (true multi-dimensional Mat - use mat.at())
float get_4d_value(const cv::Mat& mat, int i, int j, int k, int l) {
    int idx[] = {i, j, k, l};
    return mat.at<float>(idx);  // Direct multi-dimensional indexing
}
```

**Key insight:** While 3D arrays become 2D Mats with channels, `mat.ptr<T>(i, j)` gives you a pointer to position (i, j), making channel access simple. 4D+ arrays remain true multi-dimensional Mats, so `mat.at<T>(int* idx)` works directly.



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
