#pragma once
#include <opencv2/core.hpp>
#include <nanobind/ndarray.h>
#include <string>
#include <vector>


namespace cvnp_nano
{
    struct TypeSynonyms
    {
        int         cv_depth = -1;
        std::string cv_depth_name;
        std::string scalar_typename_;
        nanobind::dlpack::dtype dtype;

        std::string str() const;
    };

    extern std::vector<TypeSynonyms> sTypeSynonyms;

    std::vector<TypeSynonyms> list_types_synonyms();
    void                      print_types_synonyms();
}