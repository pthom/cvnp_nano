#pragma once
//#include "cvnp/cvnp_synonyms.h"

#include <opencv2/core/core.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace cvnp_nano
{
    nanobind::ndarray<> mat_to_nparray(const cv::Mat &m);
    cv::Mat nparray_to_mat(nanobind::ndarray<> &a);
}
