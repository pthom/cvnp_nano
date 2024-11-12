#include <nanobind/nanobind.h>
#include "cvnp_nano/cvnp_nano.h"
//#include "cvnp/cvnp.h"

//void pydef_cvnp(pybind11::module& m);
//void pydef_cvnp_test(pybind11::module& m);


int add(int i, int j) {
    return i + j;
}

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


NB_MODULE(cvnp_nano_example, m)
{
    m.def("add", &add);
    m.def("inspect", &inspect);
//    pydef_cvnp_test(m);
//    pydef_cvnp(m);
}
