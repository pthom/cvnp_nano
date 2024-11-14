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

    void                      print_types_synonyms();
}

//#define DEBUG_CVNP(x) std::cout << "DEBUG_CVNP: " << x << std::endl;
#define DEBUG_CVNP(x)


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
                    DEBUG_CVNP("policy received: " << (int)policy);
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



// Type caster for cv::Vec
// =======================
template <typename _Tp, int cn>
struct type_caster<cv::Vec<_Tp, cn>>
{
    using VecTp = cv::Vec<_Tp, cn>;
    using ScalarTp = _Tp;
    static constexpr size_t size = cn;

    NB_TYPE_CASTER(VecTp, const_name("tuple"));

    // Conversion from Python to C++ (tuple -> cv::Vec)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        DEBUG_CVNP("Enter from_python Type caster for cv::Vec");
        if (!isinstance<sequence>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        if (tuple.size() != size) {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size N to convert to cv::Vec.");
            return false;
        }

        try {
            for (size_t i = 0; i < size; ++i) {
                value[i] = cast<ScalarTp>(tuple[i]);
            }
            DEBUG_CVNP("Leave from_python Type caster for cv::Vec");
            return true;
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion from C++ to Python (cv::Vec -> tuple)
    static handle from_cpp(const VecTp &value, rv_policy policy, cleanup_list *cleanup) noexcept {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Vec");
        nanobind::list tuple_as_list;
        for (size_t i = 0; i < size; ++i) {
            tuple_as_list.append(value[i]);
        }
        nanobind::tuple tuple(tuple_as_list);
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Vec");
        return tuple.release();
    }
};


// Type caster for cv::Matx
// ========================
template <typename _Tp, int m, int n>
struct type_caster<cv::Matx<_Tp, m, n>>
{
    using MatxTp = cv::Matx<_Tp, m, n>;
    using ScalarTp = _Tp;
    static constexpr size_t rows = m;
    static constexpr size_t cols = n;

    NB_TYPE_CASTER(MatxTp, const_name("tuple"));

    // Conversion from Python to C++ (tuple -> cv::Matx)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        DEBUG_CVNP("Enter from_python Type caster for cv::Matx");
        if (!isinstance<sequence>(src))
            return false;

        auto outer_tuple = nanobind::cast<nanobind::tuple>(src);
        if (outer_tuple.size() != rows) {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size 'rows' to convert to cv::Matx.");
            return false;
        }

        try {
            for (size_t i = 0; i < rows; ++i) {
                auto inner_obj = outer_tuple[i];
                if (!isinstance<sequence>(inner_obj)) {
                    PyErr_SetString(PyExc_ValueError, "Expected inner elements to be sequences for cv::Matx.");
                    return false;
                }
                auto inner_tuple = nanobind::cast<nanobind::tuple>(inner_obj);
                if (inner_tuple.size() != cols) {
                    PyErr_SetString(PyExc_ValueError, "Expected inner tuples of size 'cols' to convert to cv::Matx.");
                    return false;
                }
                for (size_t j = 0; j < cols; ++j) {
                    value(i, j) = cast<ScalarTp>(inner_tuple[j]);
                }
            }
            DEBUG_CVNP("Leave from_python Type caster for cv::Matx");
            return true;
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion from C++ to Python (cv::Matx -> tuple)
    static handle from_cpp(const MatxTp &value, rv_policy policy, cleanup_list *cleanup) noexcept {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Matx");
        nanobind::list outer_list;
        for (size_t i = 0; i < rows; ++i) {
            nanobind::list inner_list;
            for (size_t j = 0; j < cols; ++j) {
                inner_list.append(value(i, j));
            }
            nanobind::tuple inner_tuple(inner_list);
            outer_list.append(inner_tuple);
        }
        nanobind::tuple outer_tuple(outer_list);
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Matx");
        return outer_tuple.release();
    }
};



// Type caster for cv::Size
// ========================
template<typename _Tp>
struct type_caster<cv::Size_<_Tp>>
{
    using SizeTp = cv::Size_<_Tp>;

    NB_TYPE_CASTER(SizeTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Size_");
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
            DEBUG_CVNP("Leave from_python Type caster for cv::Size_");
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
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Size_");
        auto r = nanobind::make_tuple(value.width, value.height).release();
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Size_");
        return r;
    }
};


// Type caster for cv::Point
// =========================
template<typename _Tp>
struct type_caster<cv::Point_<_Tp>>
{
    using PointTp = cv::Point_<_Tp>;

    NB_TYPE_CASTER(PointTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Point");
        if (!nanobind::isinstance<nanobind::tuple>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        if (tuple.size() != 2)
        {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size 2 to convert to cv::Point.");
            return false;
        }

        try {
            _Tp x = nanobind::cast<_Tp>(tuple[0]);
            _Tp y = nanobind::cast<_Tp>(tuple[1]);
            value = PointTp (x, y);
            DEBUG_CVNP("Leave from_python Type caster for cv::Point");
            return true;
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion part 2 (C++ -> Python, i.e., cv::Size_ -> tuple)
    static handle from_cpp(const PointTp &value, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Point");
        auto r = nanobind::make_tuple(value.x, value.y).release();
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Point");
        return r;
    }
};


// Type caster for cv::Point3_
// ===========================
template<typename _Tp>
struct type_caster<cv::Point3_<_Tp>>
{
    using PointTp = cv::Point3_<_Tp>;

    NB_TYPE_CASTER(PointTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Point3_");
        if (!nanobind::isinstance<nanobind::tuple>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        if (tuple.size() != 3)
        {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size 3 to convert to cv::Point3_.");
            return false;
        }

        try {
            _Tp x = nanobind::cast<_Tp>(tuple[0]);
            _Tp y = nanobind::cast<_Tp>(tuple[1]);
            _Tp z = nanobind::cast<_Tp>(tuple[2]);
            value = PointTp (x, y, z);
            DEBUG_CVNP("Leave from_python Type caster for cv::Point3_");
            return true;
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion part 2 (C++ -> Python, i.e., cv::Size_ -> tuple)
    static handle from_cpp(const PointTp &value, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Point3_");
        auto r = nanobind::make_tuple(value.x, value.y, value.z).release();
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Point3_");
        return r;
    }
};


// Type caster for cv::Scalar_
// ===========================
template<typename _Tp>
struct type_caster<cv::Scalar_<_Tp>>
{
    using ScalarTp = cv::Scalar_<_Tp>;

    NB_TYPE_CASTER(ScalarTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Scalar_");
        if (!nanobind::isinstance<nanobind::tuple>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        size_t tupleSize = tuple.size();
        if (tupleSize > 4)
        {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size<=4 to convert to cv::Scalar_.");
            return false;
        }

        try {
            ScalarTp r;
            if (tupleSize == 1)
                r = ScalarTp(nanobind::cast<_Tp>(tuple[0]));
            else if (tupleSize == 2)
                r = ScalarTp(nanobind::cast<_Tp>(tuple[0]), nanobind::cast<_Tp>(tuple[1]));
            else if (tupleSize == 3)
                r = ScalarTp(nanobind::cast<_Tp>(tuple[0]), nanobind::cast<_Tp>(tuple[1]), nanobind::cast<_Tp>(tuple[2]));
            else if (tupleSize == 4)
                r = ScalarTp(nanobind::cast<_Tp>(tuple[0]), nanobind::cast<_Tp>(tuple[1]), nanobind::cast<_Tp>(tuple[2]), nanobind::cast<_Tp>(tuple[3]));

            value = r;
            DEBUG_CVNP("Leave from_python Type caster for cv::Scalar_");
            return true;
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion part 2 (C++ -> Python, i.e., cv::Size_ -> tuple)
    static handle from_cpp(const ScalarTp &value, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Scalar_");
        auto r = nanobind::make_tuple(value[0], value[1], value[2], value[3]).release();
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Scalar_");
        return r;
    }
};


// Type caster for cv::Rect
// ========================
template<typename _Tp>
struct type_caster<cv::Rect_<_Tp>>
{
    using RectTp = cv::Rect_<_Tp>;

    NB_TYPE_CASTER(RectTp, const_name("tuple"));

    // Conversion part 1 (Python -> C++, i.e., tuple -> cv::Size_)
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_python Type caster for cv::Rect_");
        if (!nanobind::isinstance<nanobind::tuple>(src))
            return false;

        auto tuple = nanobind::cast<nanobind::tuple>(src);
        if (tuple.size() != 4)
        {
            PyErr_SetString(PyExc_ValueError, "Expected a tuple of size 4 to convert to cv::Rect_.");
            return false;
        }

        try {
            _Tp x = nanobind::cast<_Tp>(tuple[0]);
            _Tp y = nanobind::cast<_Tp>(tuple[1]);
            _Tp width = nanobind::cast<_Tp>(tuple[2]);
            _Tp height = nanobind::cast<_Tp>(tuple[3]);
            value = RectTp(x, y, width, height);
            DEBUG_CVNP("Leave from_python Type caster for cv::Rect_");
            return true;
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
    }

    // Conversion part 2 (C++ -> Python, i.e., cv::Size_ -> tuple)
    static handle from_cpp(const RectTp &value, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        DEBUG_CVNP("Enter from_cpp Type caster for cv::Rect_");
        auto r = nanobind::make_tuple(value.x, value.y, value.width, value.height).release();
        DEBUG_CVNP("Leave from_cpp Type caster for cv::Rect_");
        return r;
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
