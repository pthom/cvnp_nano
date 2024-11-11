#include <nanobind/nanobind.h>
//#include "cvnp/cvnp.h"

//void pydef_cvnp(pybind11::module& m);
//void pydef_cvnp_test(pybind11::module& m);


int add(int i, int j) {
    return i + j;
}

#include <opencv2/opencv.hpp>
template <typename _Tp>
void inspect(const cv::Mat_<_Tp> mat)
{
    std::cout << "[C++] Inspect cv::Mat_<_Tp>" << std::endl;
    std::cout << "        rows: " << mat.rows << std::endl;
    std::cout << "        cols: " << mat.cols << std::endl;
    std::cout << "        channels: " << mat.channels() << std::endl;
    std::cout << "        type: " << cv::typeToString(mat.type()) << std::endl;
}


NB_MODULE(cvnp_nano, m)
{
    m.def("add", &add);
    m.def("inspect", &inspect<uint8_t>);
//    pydef_cvnp_test(m);
//    pydef_cvnp(m);
}
