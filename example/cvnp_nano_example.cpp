#include <nanobind/nanobind.h>
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


struct CvNp_TestHelper
{
    // Create a Mat with 3 rows, 4 columns and 1 channel. Its shape for numpy should be (3, 4)
    cv::Mat m = cv::Mat::eye(cv::Size(4, 3), CV_8UC1);
    void SetM(int row, int col, uchar v) { m.at<uchar>(row, col) = v; }

    cv::Mat_<uint8_t> m_uint8 = cv::Mat_<uint8_t>::eye(cv::Size(4, 3));
    cv::Mat_<int8_t> m_int8 = cv::Mat_<int8_t>::eye(cv::Size(4, 3));
    cv::Mat_<uint16_t> m_uint16 = cv::Mat_<uint16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int16_t> m_int16 = cv::Mat_<int16_t>::eye(cv::Size(4, 3));
    cv::Mat_<int32_t> m_int32 = cv::Mat_<int32_t>::eye(cv::Size(4, 3));
    cv::Mat_<float> m_float = cv::Mat_<float>::eye(cv::Size(4, 3));
    cv::Mat_<double> m_double = cv::Mat_<double>::eye(cv::Size(4, 3));

    void set_m_double(int row, int col, double v) { m_double(row, col) = v; }
};


NB_MODULE(cvnp_nano_example, m)
{
    m.def("inspect", &inspect);

    nanobind::class_<CvNp_TestHelper>(m, "CvNp_TestHelper")
        .def(nanobind::init<>())
        .def_rw("m", &CvNp_TestHelper::m)
        .def("SetM", &CvNp_TestHelper::SetM)
    ;
}
