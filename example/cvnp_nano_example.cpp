#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "cvnp_nano/cvnp_nano.h"
#include <opencv2/opencv.hpp>


cv::Mat inspect(const cv::Mat& mat)
{
    std::cout << "[C++] Inspect cv::Mat_<_Tp>" << std::endl;
    std::cout << "        rows: " << mat.rows << std::endl;
    std::cout << "        cols: " << mat.cols << std::endl;
    std::cout << "        channels: " << mat.channels() << std::endl;
    std::cout << "        type: " << cv::typeToString(mat.type()) << std::endl;
    return mat;
}


// This simple function will call
// * the cast Python->C++ (for the input parameter)
// * the cast C++->Python (for the returned value)
// The unit tests check that the values and types are unmodified
cv::Mat cvnp_roundtrip(const cv::Mat& m)
{
    return m;
}

struct CvNp_TestHelper
{
    //
    // cv::Mat (shared)
    //
    // Create a Mat with 3 rows, 4 columns and 1 channel. Its shape for numpy should be (3, 4)
    cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM(int row, int col, uchar v) { m.at<uchar>(row, col) = v; }

    // cv::Mat_<Tp> (shared)
    cv::Mat_<uint8_t> m_uint8 = cv::Mat_<uint8_t>::eye(cv::Size(4, 3));
    cv::Mat_<int8_t> m_int8 = cv::Mat_<int8_t>::eye(cv::Size(4, 3));
    cv::Mat_<uint16_t> m_uint16 = cv::Mat_<uint16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int16_t> m_int16 = cv::Mat_<int16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int32_t> m_int32 = cv::Mat_<int32_t>::eye(cv::Size(4, 3));
    cv::Mat_<float> m_float = cv::Mat_<float>::eye(cv::Size(4, 3));
    cv::Mat_<double> m_double = cv::Mat_<double>::eye(cv::Size(4, 3));
    void set_m_double(int row, int col, double v) { m_double(row, col) = v; }

    //
    // cv::Matx (not shared)
    //
    cv::Matx32d mx_ns = cv::Matx32d::eye();
    void SetMX_ns(int row, int col, double v) { mx_ns(row, col) = v;}

    //
    // cv::Vec not shared
    //
    cv::Vec3f v3_ns = {1.f, 2.f, 3.f};
    void SetV3_ns(int idx, float v) { v3_ns(idx) = v; }

    //
    // *Not* shared simple structs (Size, Point2 and Point3)
    //
    cv::Size_<int> s = cv::Size(123, 456);
    void SetWidth(int w) { s.width = w;}
    void SetHeight(int h) { s.height = h;}

    cv::Point2i pt = cv::Point2i(42, 43);
    void SetX(int x) { pt.x = x; }
    void SetY(int y) { pt.y = y; }

    cv::Point3d pt3 = cv::Point3d(41.5, 42., 42.5);
    void SetX3(double x) { pt3.x = x; }
    void SetY3(double y) { pt3.y = y; }
    void SetZ3(double z) { pt3.z = z; }

    //
    // cv::Mat and sub matrices
    //
    cv::Mat m10 = cv::Mat(cv::Size(100, 100), CV_32FC3, cv::Scalar(0.f, 0.f, 0.f));
    void SetM10(int row, int col, cv::Vec3f v) { m10.at<cv::Vec3f>(row, col) = v; }
    cv::Vec3f GetM10(int row, int col) { return m10.at<cv::Vec3f>(row, col); }
    cv::Mat GetSubM10() {
        cv::Mat sub = m10(cv::Rect(1, 1, 3, 3));
        return sub;
    }
    int m10_refcount()
    {
        if (m10.u)
            return m10.u->refcount;
        else
        {
            printf("m10.u is null!\n");
            return 0;
        }
    }

    //
    // cv::Scalar_
    //
    cv::Scalar scalar_double = cv::Scalar(1.);
    cv::Scalar_<float> scalar_float = cv::Scalar_<float>(1.f, 2.f);
    cv::Scalar_<int32_t> scalar_int32 = cv::Scalar_<int32_t>(1, 2, 3);
    cv::Scalar_<uint8_t> scalar_uint8 = cv::Scalar_<uint8_t>(1, 2, 3, 4);

    //
    // cv::Rect
    //
    cv::Rect  rect_int = cv::Rect(1, 2, 3, 4);
    cv::Rect_<double> rect_double = cv::Rect_<double>(5., 6., 7., 8.);
};


// Returns a short lived Matx: sharing memory for this matrix makes *no sense at all*,
// since its pointer lives on the stack and is deleted as soon as we exit the function!
cv::Matx33d ShortLivedMatx()
{
    auto mat = cv::Matx33d::eye();
    return mat;
}


// Returns a short lived Mat: sharing memory for this matrix *makes sense*
// since the capsule will add to its reference count
cv::Mat ShortLivedMat()
{
    auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
    mat = cv::Scalar(12, 34, 56, 78);
    return mat;
}


cv::Matx33d make_eye()
{
    return cv::Matx33d::eye();
}


void display_eye(cv::Matx33d m = cv::Matx33d::eye())
{
    printf("display_eye\n");
    for(int row=0; row < 3; ++row)
        printf("%lf, %lf, %lf\n", m(row, 0), m(row, 1), m(row, 2));
}


cv::Matx21d RoundTripMatx21d(cv::Matx21d & m)
{
    //std::cout << "RoundTripMatx21d received "  << m(0, 0) << "    " << m(1, 0) << "\n";
    return m;
}


// Multidimensional array test functions
// ======================================

// Create a 3D single-channel Mat (e.g., 4x5x6)
cv::Mat create_3d_mat()
{
    int sizes[] = {4, 5, 6};
    cv::Mat mat(3, sizes, CV_32F);
    
    // Fill with test pattern: value = i*100 + j*10 + k
    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                int idx[] = {i, j, k};
                mat.at<float>(idx) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }
    return mat;
}

// Create a 4D single-channel Mat (e.g., 3x4x5x2)
cv::Mat create_4d_mat()
{
    int sizes[] = {3, 4, 5, 2};
    cv::Mat mat(4, sizes, CV_32F);
    
    // Fill with test pattern
    float value = 0.0f;
    float* data = (float*)mat.data;
    for (int i = 0; i < mat.total(); ++i) {
        data[i] = value;
        value += 1.0f;
    }
    return mat;
}

// Create a 5D single-channel Mat (e.g., 2x3x4x2x2)
cv::Mat create_5d_mat()
{
    int sizes[] = {2, 3, 4, 2, 2};
    cv::Mat mat(5, sizes, CV_64F);
    
    // Fill with test pattern
    double* data = (double*)mat.data;
    for (int i = 0; i < mat.total(); ++i) {
        data[i] = static_cast<double>(i * 0.5);
    }
    return mat;
}

// Get value from 3D mat
float get_3d_value(const cv::Mat& mat, int i, int j, int k)
{
    // IMPORTANT: 3D arrays are a special case due to OpenCV's channel representation!
    // After roundtrip through Python, a 3D NumPy array becomes a 2D OpenCV Mat with channels:
    //   - mat.dims = 2 (not 3!)
    //   - mat.rows = first dimension
    //   - mat.cols = second dimension  
    //   - mat.channels() = third dimension
    //
    // Fortunately, we can use mat.ptr<T>(i, j) which returns a pointer to row i, column j.
    // Then [k] indexes into the k-th channel. Simple and clean!
    
    return mat.ptr<float>(i, j)[k];
}

// Set value in 3D mat
void set_3d_value(cv::Mat& mat, int i, int j, int k, float value)
{
    // Same simple approach: ptr to (i,j), then index [k] for the channel
    mat.ptr<float>(i, j)[k] = value;
}

// Get value from 4D mat
float get_4d_value(const cv::Mat& mat, int i, int j, int k, int l)
{
    // 4D+ arrays are straightforward - they remain true multi-dimensional Mats!
    // No channel confusion: mat.dims = 4, mat.channels() = 1
    // We can use OpenCV's built-in mat.at<T>(int* idx) directly!
    // This is MUCH simpler than 3D arrays (which require manual stride calculation)
    
    int idx[] = {i, j, k, l};
    return mat.at<float>(idx);
}

// Set value in 4D mat
void set_4d_value(cv::Mat& mat, int i, int j, int k, int l, float value)
{
    // Simple and straightforward for 4D+ arrays - just use mat.at()!
    // No channel complications like with 3D arrays
    
    int idx[] = {i, j, k, l};
    mat.at<float>(idx) = value;
}

// Get shape of multidimensional mat
std::vector<int> get_mat_shape(const cv::Mat& mat)
{
    std::vector<int> shape;
    if (mat.dims <= 2) {
        shape.push_back(mat.rows);
        shape.push_back(mat.cols);
        if (mat.channels() > 1) {
            shape.push_back(mat.channels());
        }
    } else {
        for (int i = 0; i < mat.dims; ++i) {
            shape.push_back(mat.size[i]);
        }
        if (mat.channels() > 1) {
            shape.push_back(mat.channels());
        }
    }
    return shape;
}

// Inspect multidimensional mat
void inspect_multidim(const cv::Mat& mat)
{
    std::cout << "[C++] Multidimensional Mat:" << std::endl;
    std::cout << "        dims: " << mat.dims << std::endl;
    std::cout << "        channels: " << mat.channels() << std::endl;
    std::cout << "        total elements: " << mat.total() << std::endl;
    std::cout << "        type: " << cv::typeToString(mat.type()) << std::endl;
    std::cout << "        shape: [";
    for (int i = 0; i < mat.dims; ++i) {
        std::cout << mat.size[i];
        if (i < mat.dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "        step (bytes): [";
    for (int i = 0; i < mat.dims; ++i) {
        std::cout << mat.step[i];
        if (i < mat.dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "        elemSize: " << mat.elemSize() << std::endl;
    std::cout << "        elemSize1: " << mat.elemSize1() << std::endl;
}


// Debug function to inspect mat after conversion from Python
void debug_3d_mat(const cv::Mat& mat)
{
    std::cout << "\n[C++] Debug 3D Mat after conversion from Python:" << std::endl;
    std::cout << "  dims: " << mat.dims << std::endl;
    std::cout << "  rows: " << mat.rows << std::endl;
    std::cout << "  cols: " << mat.cols << std::endl;
    
    if (mat.dims > 0 && mat.size.p != nullptr) {
        std::cout << "  size.p: [";
        for (int i = 0; i < mat.dims; ++i) {
            std::cout << mat.size.p[i];
            if (i < mat.dims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  step.p: [";
    if (mat.step.p != nullptr) {
        for (int i = 0; i < mat.dims; ++i) {
            std::cout << mat.step.p[i];
            if (i < mat.dims - 1) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  elemSize: " << mat.elemSize() << std::endl;
    std::cout << "  elemSize1: " << mat.elemSize1() << std::endl;
}


NB_MODULE(cvnp_nano_example, m)
{
    m.def("inspect", &inspect);

    nanobind::class_<CvNp_TestHelper>(m, "CvNp_TestHelper")
        .def(nanobind::init<>())
        .def_rw("m", &CvNp_TestHelper::m)
        .def("SetM", &CvNp_TestHelper::SetM)

        .def_rw("m_uint8", &CvNp_TestHelper::m_uint8)
        .def_rw("m_int8", &CvNp_TestHelper::m_int8)
        .def_rw("m_uint16", &CvNp_TestHelper::m_uint16)
        .def_rw("m_int16", &CvNp_TestHelper::m_int16)
        .def_rw("m_int16", &CvNp_TestHelper::m_int16)
        .def_rw("m_int32", &CvNp_TestHelper::m_int32)
        .def_rw("m_float", &CvNp_TestHelper::m_float)
        .def_rw("m_double", &CvNp_TestHelper::m_double)
        .def("set_m_double", &CvNp_TestHelper::set_m_double)

        .def_rw("mx_ns", &CvNp_TestHelper::mx_ns)
        .def("SetMX_ns", &CvNp_TestHelper::SetMX_ns)

        .def_rw("v3_ns", &CvNp_TestHelper::v3_ns)
        .def("SetV3_ns", &CvNp_TestHelper::SetV3_ns)

        .def_rw("s", &CvNp_TestHelper::s)
        .def("SetWidth", &CvNp_TestHelper::SetWidth)
        .def("SetHeight", &CvNp_TestHelper::SetHeight)

        .def_rw("pt", &CvNp_TestHelper::pt)
        .def("SetX", &CvNp_TestHelper::SetX)
        .def("SetY", &CvNp_TestHelper::SetY)

        .def_rw("pt3", &CvNp_TestHelper::pt3)
        .def("SetX3", &CvNp_TestHelper::SetX3)
        .def("SetY3", &CvNp_TestHelper::SetY3)
        .def("SetZ3", &CvNp_TestHelper::SetZ3)

        .def_rw("m10", &CvNp_TestHelper::m10)
        .def("SetM10", &CvNp_TestHelper::SetM10)
        .def("GetM10", &CvNp_TestHelper::GetM10)
        .def("GetSubM10", &CvNp_TestHelper::GetSubM10)
        .def("m10_refcount", &CvNp_TestHelper::m10_refcount)

        .def_rw("scalar_double", &CvNp_TestHelper::scalar_double)
        .def_rw("scalar_float", &CvNp_TestHelper::scalar_float)
        .def_rw("scalar_int32", &CvNp_TestHelper::scalar_int32)
        .def_rw("scalar_uint8", &CvNp_TestHelper::scalar_uint8)

        .def_rw("rect_int", &CvNp_TestHelper::rect_int)
        .def_rw("rect_double", &CvNp_TestHelper::rect_double)
        ;

    m.def("cvnp_roundtrip", cvnp_roundtrip);

    m.def("short_lived_matx", ShortLivedMatx);
    m.def("short_lived_mat", ShortLivedMat);
    m.def("RoundTripMatx21d", RoundTripMatx21d);

    // Multidimensional array functions
    m.def("create_3d_mat", &create_3d_mat);
    m.def("create_4d_mat", &create_4d_mat);
    m.def("create_5d_mat", &create_5d_mat);
    m.def("get_3d_value", &get_3d_value);
    m.def("set_3d_value", &set_3d_value);
    m.def("get_4d_value", &get_4d_value);
    m.def("set_4d_value", &set_4d_value);
    m.def("get_mat_shape", &get_mat_shape);
    m.def("inspect_multidim", &inspect_multidim);
    m.def("debug_3d_mat", &debug_3d_mat);

    m.def("print_types_synonyms", cvnp_nano::print_types_synonyms);
}
