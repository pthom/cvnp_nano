#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace cvnp_nano
{
    nanobind::ndarray<> mat_to_nparray(const cv::Mat &m, nanobind::handle owner);
    cv::Mat nparray_to_mat(nanobind::ndarray<> &a, nanobind::handle owner);

    template <typename _Tp>
    cv::Mat_<_Tp> nparray_to_mat_typed(nanobind::ndarray<> &a, nanobind::handle owner) {
        cv::Mat mat = nparray_to_mat(a, owner);
        return mat;  // Convert to Mat_<_Tp>
    }
}

#define DEBUG_CVNP(x) std::cout << "DEBUG_CVNP: " << x << std::endl;
//#define DEBUG_CVNP(x)


NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

//
// Type caster for cv::Mat
// ========================
template <>
struct type_caster<cv::Mat>
{
    NB_TYPE_CASTER(cv::Mat, const_name("numpy.array"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Mat");
        if (!isinstance<ndarray<>>(src))
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat type_caster from_python: expected a numpy.ndarray");
            return false;
        }
        try
        {
            auto a = nanobind::cast<ndarray<>>(src);

            // Create a capsule that keeps the Python ndarray alive as long as cv::Mat needs it
            nanobind::object capsule_owner = nanobind::capsule(src.ptr(), [](void* p) noexcept {
                Py_XDECREF(reinterpret_cast<PyObject*>(p));  // Decrement reference count of ndarray when capsule is destroyed
            });
            Py_INCREF(src.ptr());  // Increment reference to ensure ndarray is not prematurely garbage collected


            this->value = cvnp_nano::nparray_to_mat(a, capsule_owner);
            DEBUG_CVNP("Leave from_python Type caster for cv::Mat");
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
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Mat");

        try
        {
            // Set default policies if automatic
            if (policy == rv_policy::automatic)
                policy = rv_policy::copy;
            else if (policy == rv_policy::automatic_reference)
                policy = rv_policy::reference;

            // Exported ndarray
            ndarray<> a;
            {
                if (policy == rv_policy::take_ownership)
                {
                    // Create a Python object that wraps the existing C++ instance and takes full ownership of it. No copies are made. Python will call the C++ destructor and delete operator when the Python wrapper is garbage collected at some later point. The C++ side must relinquish ownership and is not allowed to destruct the instance, or undefined behavior will ensue
                    DEBUG_CVNP("    rv_policy::take_ownership => capsule takes ownership");
                    // Warning: the capsule will delete the mat which is passed as a parameter, since "the C++ side must relinquish ownership and is not allowed to destruct the instance, or undefined behavior will ensue"
                    nanobind::object owner = nanobind::capsule(&mat, [](void* p) noexcept { delete (cv::Mat*)p; });
                    a = cvnp_nano::mat_to_nparray(mat, owner);
                }
                else if (policy == rv_policy::copy)
                {
                    // Copy-construct a new Python object from the C++ instance. The new copy will be owned by Python, while C++ retains ownership of the original.
                    DEBUG_CVNP("    rv_policy::copy => copy mat on heap");
                    cv::Mat* heap_mat = new cv::Mat(mat);  // Allocate on the heap
                    // Note: the constructor cv::Mat(mat) will not copy the data, but instead increment the reference counter
                    // of `cv::Mat mat`
                    // => the python numpy array and the C++ cv::Mat will share the same data
                    nanobind::object owner = nanobind::capsule(heap_mat, [](void* p) noexcept { delete (cv::Mat*)p; });
                    a = cvnp_nano::mat_to_nparray(*heap_mat, owner);
                }
                else if (policy == rv_policy::move)
                {
                    // Move-construct a new Python object from the C++ instance. The new object will be owned by Python, while C++ retains ownership of the original (whose contents were likely invalidated by the move operation).

                    // move is implemented as rv_policy::copy (because OpenCV's Mat already has a reference counter)
                    DEBUG_CVNP("    rv_policy::move => copy mat on heap (not a real move)");
                    cv::Mat* heap_mat = new cv::Mat(mat);  // Allocate on the heap (this is not a move per se)
                    // => the python numpy array and the C++ cv::Mat will share the same data
                    nanobind::object owner = nanobind::capsule(heap_mat, [](void* p) noexcept { delete (cv::Mat*)p; });
                    a = cvnp_nano::mat_to_nparray(*heap_mat, owner);
                }
                else if (policy == rv_policy::reference)
                {
                    // Create a Python object that wraps the existing C++ instance without taking ownership of it. No copies are made. Python will never call the destructor or delete operator, even when the Python wrapper is garbage collected.
                    DEBUG_CVNP("    rv_policy::reference => using no_owner");
                    nanobind::handle no_owner = {};
                    a = cvnp_nano::mat_to_nparray(mat, no_owner);
                }
                else if (policy == rv_policy::reference_internal)
                {
                    // A safe extension of the reference policy for methods that implement some form of attribute access. It creates a Python object that wraps the existing C++ instance without taking ownership of it. Additionally, it adjusts reference counts to keeps the method’s implicit self argument alive until the newly created object has been garbage collected.
                    DEBUG_CVNP("    rv_policy::reference_internal => using no_owner");
                    nanobind::handle no_owner = {};
                    a = cvnp_nano::mat_to_nparray(mat, no_owner);
                }
                else if (policy == rv_policy::none)
                {
                    // This is the most conservative policy: it simply refuses the cast unless the C++ instance already has a corresponding Python object, in which case the question of ownership becomes moot.
                    DEBUG_CVNP("    rv_policy::none => unhandled yet");
                    throw std::runtime_error("rv_policy::none not yet supported in cv::Mat caster");
                }
                else
                {
                    printf("policy received : %i\n", policy);
                    throw std::runtime_error("unexpected rv_policy in cv::Mat caster");
                }
            }

            // inspired by ndarray.h caster:
            // We need to call ndarray_export to export a python handle for the ndarray
            auto r = ndarray_export(
                a.handle(), // internal array handle
                nanobind::numpy::value, // framework (i.e numpy, pytorch, etc)
                policy,
                cleanup);

            DEBUG_CVNP("Leave from_cpp Type caster for cv::Mat");
            return r;
        }
        catch (const std::exception& e)
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "nanobind: exception in MatrixFixedSize type_caster from_cpp: %s", e.what());
            return {};
        }
    }
};


//
// Type caster for cv::Mat_<_Tp>  (reuses cv::Mat caster)
// =======================================================
template <typename _Tp>
struct type_caster<cv::Mat_<_Tp>> : public type_caster<cv::Mat> // Inherit from cv::Mat caster
{
    // Adjust type_caster for Mat_<_Tp> to handle _Tp and ensure correct dtype
    NB_TYPE_CASTER(cv::Mat_<_Tp>, const_name("numpy.ndarray"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Mat_<_Tp>");

        if (!isinstance<ndarray<>>(src))
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat_<_Tp> type_caster from_python: expected a numpy.ndarray");
            return false;
        }

        try
        {
            auto a = nanobind::cast<ndarray<>>(src);

            // Check if the dtype of ndarray matches _Tp
            if (a.dtype() != nanobind::dtype<_Tp>()) {
                PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: dtype of ndarray does not match cv::Mat_<_Tp> type");
                return false;
            }

            // Create a capsule that keeps the Python ndarray alive as long as cv::Mat_<_Tp> needs it
            nanobind::object capsule_owner = nanobind::capsule(src.ptr(), [](void* p) noexcept {
                Py_XDECREF(reinterpret_cast<PyObject*>(p));
            });
            Py_INCREF(src.ptr());

            // Use nparray_to_mat_typed to convert ndarray to cv::Mat_<_Tp>
            this->value = cvnp_nano::nparray_to_mat_typed<_Tp>(a, capsule_owner);

            DEBUG_CVNP("Leave from_python Type caster for cv::Mat_<_Tp>");
            return true;
        }
        catch (const std::exception& e)
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat_<_Tp> type_caster from_python, exception: %s", e.what());
            return false;
        }
    }

    static handle from_cpp(const cv::Mat_<_Tp> &mat, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Mat_<_Tp>");

        try
        {
            // Call the base cv::Mat type_caster's from_cpp method
            return type_caster<cv::Mat>::from_cpp(mat, policy, cleanup);
        }
        catch (const std::exception& e)
        {
            PyErr_WarnFormat(PyExc_Warning, 1, "cvnp_nano: cv::Mat_<_Tp> type_caster from_cpp, exception: %s", e.what());
            return {};
        }
    }
};



// Type caster for cv::Vec or cv::Matx
// ====================================
template <typename T>
using is_vec_or_matx = std::disjunction<std::is_same<T, cv::Vec<typename T::value_type, T::channels>>, std::is_same<T, cv::Matx<typename T::value_type, T::rows, T::cols>>>;

template <typename T>
struct type_caster<T, enable_if_t<is_vec_or_matx<T>::value>>
{

    // Info about cv::Vec or cv::Matx
    using _Tp = typename T::value_type;  // scalar type
    static constexpr int rows = T::rows; // number of rows
    static constexpr int cols = T::cols; // number of columns (cn)
    static constexpr bool is_matx = std::is_same_v<T, cv::Matx<typename T::value_type, T::rows, T::cols>>;
    static constexpr size_t ndim = is_matx ? 2 : 1; // number of dimensions

    // Define ndarray
    using ndarray_shape = std::conditional_t<is_matx, shape<rows, cols>, shape<rows>>;
    using NDArray = ndarray<_Tp, numpy, ndarray_shape>;
    using NDArrayCaster = type_caster<NDArray>;

    NB_TYPE_CASTER(T, NDArrayCaster::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        // Check if src is ndarray
        NDArrayCaster caster;
        bool is_valid_ndarray = caster.from_python(src, flags, cleanup);
        if (!is_valid_ndarray)
        {
            return false;
        }

        // Convert ndarray to cv::Vec or cv::Matx
        const NDArray &array = caster.value;
        memcpy(value.val, array.data(), rows * cols * sizeof(_Tp));

        return true;
    }

    static handle from_cpp(const T &matx, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        size_t shape[ndim];

        if constexpr (is_matx)
        {
            shape[0] = (size_t)rows;
            shape[1] = (size_t)cols;
        }
        else
        {
            shape[0] = (size_t)rows;
        }

        void *ptr = (void *)matx.val;

        switch (policy)
        {
            case rv_policy::automatic:
                policy = rv_policy::copy;
                break;

            case rv_policy::automatic_reference:
                policy = rv_policy::reference;
                break;

            default: // leave policy unchanged
                break;
        }

        object owner;
        if (policy == rv_policy::move)
        {
            T *temp = new T(std::move(matx));
            owner = capsule(temp, [](void *p) noexcept
            { delete (T *)p; });
            ptr = temp->val;
        }

        rv_policy array_rv_policy = policy == rv_policy::move ? rv_policy::reference : policy;

        // Convert cv::Vec or cv::Matx to ndarray
        object o = steal(NDArrayCaster::from_cpp(NDArray(ptr, ndim, shape), policy, cleanup));
        return o.release();
    }
};



template<typename _Tp>
struct type_caster<cv::Size_<_Tp>>
{
    using SizeTp = cv::Size_<int>;

    NB_TYPE_CASTER(SizeTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        if (!nanobind::isinstance<nanobind::tuple>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        if (tuple.size() != 2)
        {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size 2 to convert to cv::Size.");
            return false;
        }

        try {
            _Tp width = nanobind::cast<_Tp>(tuple[0]);
            _Tp height = nanobind::cast<_Tp>(tuple[1]);
            value = SizeTp(width, height);
            return true;
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion part 2 (C++ -> Python, i.e., cv::Size_ -> tuple)
    static handle from_cpp(const SizeTp &value, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        return nanobind::make_tuple(value.width, value.height).release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
