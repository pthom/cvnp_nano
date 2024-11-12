#include "cvnp_nano/cvnp_nano.h"
#include "cvnp_nano/cvnp_nano_synonyms.h"
#include "opencv2/core.hpp"

namespace cvnp_nano
{
    namespace detail
    {
        #define DEBUG_ALLOCATOR

#ifdef DEBUG_ALLOCATOR
        int nbAllocations = 0;
#endif

        // Translated from cv2_numpy.cpp in OpenCV source code
        // A custom allocator for cv::Mat that attaches an owner to the cv::Mat
        class CvnpAllocator : public cv::MatAllocator
        {
        public:
            CvnpAllocator() = default;
            ~CvnpAllocator() = default;

            // Attaches an owner to a cv::Mat
            static void attach_nparray(cv::Mat &m, nanobind::handle owner)
            {
                static CvnpAllocator instance;

                // Ensure no existing custom allocator to avoid accidental double attachment
                if (m.u && m.allocator) {
                    throw std::logic_error("attach_nparray: cv::Mat already has a custom allocator attached");
                }

                cv::UMatData* u = new cv::UMatData(&instance);
                u->data = u->origdata = (uchar*)m.data;
                u->size = m.total();

                u->userdata = &owner;
                u->refcount = 1;

                #ifdef DEBUG_ALLOCATOR
                ++nbAllocations;
                printf("CvnpAllocator::attach_nparray(py::array) nbAllocations=%d\n", nbAllocations);
                #endif

                m.u = u;
                m.allocator = &instance;
            }

            cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const override
            {
                throw nanobind::value_error("CvnpAllocator::allocate \"standard\" should never happen");
                // return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
            }

            bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const override
            {
                throw nanobind::value_error("CvnpAllocator::allocate \"copy\" should never happen");
                // return stdAllocator->allocate(u, accessFlags, usageFlags);
            }

            void deallocate(cv::UMatData* u) const override
            {
                if(!u)
                {
#ifdef DEBUG_ALLOCATOR
                    printf("CvnpAllocator::deallocate() with null ptr!!! nbAllocations=%d\n", nbAllocations);
#endif
                    return;
                }

                // This function can be called from anywhere, so need the GIL
                nanobind::gil_scoped_acquire gil;
                assert(u->urefcount >= 0);
                assert(u->refcount >= 0);
                if(u->refcount == 0)
                {
                    PyObject* o = (PyObject*)u->userdata;
                    Py_XDECREF(o);
                    delete u;
#ifdef DEBUG_ALLOCATOR
                    --nbAllocations;
                    printf("CvnpAllocator::deallocate() nbAllocations=%d\n", nbAllocations);
#endif
                }
                else
                {
#ifdef DEBUG_ALLOCATOR
                    printf("CvnpAllocator::deallocate() - not doing anything since urefcount=%d nbAllocations=%d\n",
                            u->urefcount,
                           nbAllocations);
#endif
                }
            }
        };


        nanobind::dlpack::dtype determine_np_dtype(int cv_depth)
        {
            for (auto format_synonym : cvnp_nano::sTypeSynonyms)
                if (format_synonym.cv_depth == cv_depth)
                    return format_synonym.dtype;

            std::string msg = "numpy does not support this OpenCV depth: " + std::to_string(cv_depth) +  " (in determine_np_dtype)";
            throw std::invalid_argument(msg.c_str());
        }

        int determine_cv_depth(nanobind::dlpack::dtype dt)
        {
            for (auto format_synonym : cvnp_nano::sTypeSynonyms)
                if (format_synonym.dtype == dt)
                    return format_synonym.cv_depth;

            std::string msg = std::string("OpenCV does not support this numpy array type (in determine_np_dtype)!");
            throw std::invalid_argument(msg.c_str());
        }

        int determine_cv_type(const nanobind::ndarray<>& a, int depth)
        {
            if (a.ndim() < 2)
                throw std::invalid_argument("determine_cv_type needs at least two dimensions");
            if (a.ndim() > 3)
                throw std::invalid_argument("determine_cv_type needs at most three dimensions");
            if (a.ndim() == 2)
                return CV_MAKETYPE(depth, 1);

            //We now know that shape.size() == 3
            int nb_channels = a.shape(2);
            return CV_MAKETYPE(depth, nb_channels);
        }

        cv::Size determine_cv_size(const nanobind::ndarray<>& a)
        {
            if (a.ndim() < 2)
                throw std::invalid_argument("determine_cv_size needs at least two dimensions");
            return cv::Size(static_cast<int>(a.shape(1)), static_cast<int>(a.shape(0)));
        }

        std::vector<std::size_t> determine_shape(const cv::Mat& m)
        {
            if (m.channels() == 1) {
                return {
                    static_cast<size_t>(m.rows)
                    , static_cast<size_t>(m.cols)
                };
            }
            return {
                static_cast<size_t>(m.rows)
                , static_cast<size_t>(m.cols)
                , static_cast<size_t>(m.channels())
            };
        }

        std::vector<int64_t> determine_strides(const cv::Mat& m) {
            // Return strides in nb element (not bytes)
            if (m.channels() == 1) {
                return {
                    static_cast<int64_t>(m.step[0] / m.elemSize1()), // row stride
                    static_cast<int64_t>(m.step[1] / m.elemSize1())  // column stride
                };
            }
            return {
                static_cast<int64_t>(m.step[0] / m.elemSize1()), // row stride
                static_cast<int64_t>(m.step[1] / m.elemSize1()), // column stride
                static_cast<int64_t>(1) // channel stride
            };
        }

        int determine_ndim(const cv::Mat& m)
        {
            return m.channels() == 1 ? 2 : 3;
        }
    } // namespace detail


    nanobind::ndarray<> mat_to_nparray(const cv::Mat &m, nanobind::handle owner)
    {
        void *data = static_cast<void *>(m.data);
        size_t ndim = detail::determine_ndim(m);
        std::vector<size_t> shape = detail::determine_shape(m);
        std::vector<int64_t> strides = detail::determine_strides(m);
        nanobind::dlpack::dtype dtype = detail::determine_np_dtype(m.depth());

        auto a = nanobind::ndarray<>(
            data,
            ndim,
            shape.data(),
            owner,
            strides.data(),
            dtype
        );
        return a;
    }


    bool is_array_contiguous(const nanobind::ndarray<>& a)
    {
        if (a.ndim() < 2 || a.ndim() > 3)
            throw std::invalid_argument("cvnp_nano only supports 2D or 3D numpy arrays");

        if (a.ndim() == 2)
        {
            return a.stride(0) == a.shape(1) && a.stride(1) == 1;
        }
        else
        {
            return a.stride(0) == a.shape(1) * a.shape(2) && a.stride(1) == a.shape(2) && a.stride(2) == 1;
        }
    }


    cv::Mat nparray_to_mat(nanobind::ndarray<>& a, nanobind::handle owner)
    {
        // note: empty arrays are not contiguous, but that's fine. Just
        //       make sure to not access mutable_data
        bool is_contiguous = is_array_contiguous(a);
        bool is_not_empty = a.size() != 0;
        if (! is_contiguous && is_not_empty) {
            throw std::invalid_argument("cvnp::nparray_to_mat / Only contiguous numpy arrays are supported. / Please use np.ascontiguousarray() to convert your matrix");
        }

        int depth = detail::determine_cv_depth(a.dtype());
        int type = detail::determine_cv_type(a, depth);
        cv::Size size = detail::determine_cv_size(a);
        cv::Mat m(size, type, is_not_empty ? a.data() : nullptr);

        if (is_not_empty)
            detail::CvnpAllocator::attach_nparray(m, owner);

        return m;
    }

}
