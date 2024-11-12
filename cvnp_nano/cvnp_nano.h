#pragma once
#include <opencv2/core/core.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace cvnp_nano
{
    nanobind::ndarray<> mat_to_nparray(const cv::Mat &m);
    cv::Mat nparray_to_mat(nanobind::ndarray<> &a);
}


NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Type caster for cv::Mat
template <>
struct type_caster<cv::Mat>
{
    NB_TYPE_CASTER(cv::Mat, const_name("cv::Mat"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        if (!isinstance<ndarray<>>(src))
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat type_caster from_python: expected a numpy.ndarray");
            return false;
        }
        try
        {
            auto a = nanobind::cast<ndarray<>>(src);
            this->value = cvnp_nano::nparray_to_mat(a);
            return true;
        }
        catch (const std::exception& e)
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat type_caster from_python, exception: %s", e.what());
            return false;
        }
    }

    static handle from_cpp(const cv::Mat &mat, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        try
        {
            // policy defaults
            if (policy == rv_policy::automatic)
                    policy = rv_policy::copy;
            else if (policy == rv_policy::automatic_reference)
                    policy = rv_policy::reference;

            // Exported ndarray
            ndarray<> a;
            {
                // The exported array is either constructed from the origin cv::Mat
                // (or from a copy on the heap if moving)
                if (policy != rv_policy::move)
                {
                    a = cvnp_nano::mat_to_nparray(mat);
                }
                else
                {
                    // if `rv_policy::move` is used, we need to create a new cv::Mat on the heap
                    cv::Mat* heap_mat = new cv::Mat(mat);  // Allocate on the heap
                    nanobind::object owner = nanobind::capsule(heap_mat, [](void* p) noexcept { delete (cv::Mat*)p; });
                    a = cvnp_nano::mat_to_nparray(*heap_mat);
                }
            }

            // inspired by ndarray.h caster:
            // We need to call ndarray_export to export a python handle for the ndarray
            auto r = ndarray_export(
                a.handle(), // internal array handle
                nanobind::numpy::value, // framework (i.e numpy, pytorch, etc)
                policy,
                cleanup);
            return r;
        }
        catch (const std::exception& e)
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "nanobind: exception in MatrixFixedSize type_caster from_cpp: %s", e.what());
            return {};
        }
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
