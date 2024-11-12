#include "cvnp_nano_synonyms.h"
#include <string>
#include <cstdint>
#include <iostream>


namespace cvnp_nano
{
    std::vector<TypeSynonyms> sTypeSynonyms
    {
        { CV_8U,  "CV_8U", "uint8", nanobind::dtype<uint8_t>() },
        { CV_8S,  "CV_8S", "int8", nanobind::dtype<int8_t>() },
        { CV_16U, "CV_16U", "uint16", nanobind::dtype<uint16_t>() },
        { CV_16S, "CV_16S", "int16", nanobind::dtype<int16_t>() },
        { CV_32S, "CV_32S", "int32", nanobind::dtype<int32_t>() },
        { CV_32F, "CV_32F", "float32", nanobind::dtype<float>() },
        { CV_64F, "CV_64F", "float64", nanobind::dtype<double>() },

        // Note: this format needs adaptations: float16
    };


    static int sColumnWidth = 12;

    static std::string align_center(const std::string& s)
    {
        int nb_spaces = s.size() < sColumnWidth ? sColumnWidth - s.size() : 0;
        int nb_spaces_left = nb_spaces / 2;
        int nb_spaces_right = sColumnWidth - s.size() - nb_spaces_left;
        if (nb_spaces_right < 0)
            nb_spaces_right = 0;
        return std::string((size_t)nb_spaces_left, ' ') + s + std::string( (size_t)nb_spaces_right, ' ');
    }
    static std::string align_center(const int v)
    {
        return align_center(std::to_string(v));
    }

    std::string TypeSynonyms::str() const
    {
        return    align_center(cv_depth) + align_center(cv_depth_name) 
                + align_center(scalar_typename_);
    }

    
    std::string _print_types_synonyms_str()
    {
        std::string title = 
              align_center("cv_depth") + align_center("cv_depth_name") 
            + align_center("np_format") + align_center("np_format_long");; 

        std::string r;
        r = title + "\n";
        for (const auto& format: sTypeSynonyms)
            r = r + format.str() + "\n";
        return r;
    }

    std::vector<TypeSynonyms> list_types_synonyms()
    {
        return sTypeSynonyms;
    }

    void print_types_synonyms()
    {
        std::cout << _print_types_synonyms_str();
    }
}