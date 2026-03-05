#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp_core.hpp"

namespace py = pybind11;

py::tuple sample_sov_wrap(py::array_t<double, py::array::c_style | py::array::forcecast> a_arr, 
                          py::array_t<double, py::array::c_style | py::array::forcecast> b_arr, 
                          py::array_t<double, py::array::c_style | py::array::forcecast> L_arr, 
                          int n,
                          unsigned int seed = 1,
                          int spherical = 1,
                          int rqmc = 1,
                          std::optional<std::vector<double>> shift_opt = std::nullopt) {
    
    int d = a_arr.size();
    if (b_arr.size() != d) throw std::invalid_argument("b size mismatch");
    auto L_info = L_arr.request();
    if (L_info.ndim != 2 || L_info.shape[0] != d || L_info.shape[1] != d) throw std::invalid_argument("L shape mismatch");

    const double* a = static_cast<const double*>(a_arr.data());
    const double* b = static_cast<const double*>(b_arr.data());
    const double* L = static_cast<const double*>(L_arr.data());

    auto result = sample_sov_cpp(a, b, L, d, n, seed, spherical, rqmc, shift_opt);

    py::array_t<double> samples_py({n, d});
    py::array_t<double> weights_py(n);

    auto samples_m = samples_py.mutable_unchecked<2>();
    auto weights_m = weights_py.mutable_unchecked<1>();

    for(int i=0; i<n; ++i) {
        weights_m(i) = result.second[i];
        for(int j=0; j<d; ++j) {
            samples_m(i, j) = result.first[i * d + j];
        }
    }

    return py::make_tuple(samples_py, weights_py);
}

py::tuple st_cdf_wrap(py::array_t<double, py::array::c_style | py::array::forcecast> mu_arr, 
                      py::array_t<double, py::array::c_style | py::array::forcecast> L_arr, 
                      py::array_t<double, py::array::c_style | py::array::forcecast> c1_arr,
                      double c2,
                      int n,
                      unsigned int seed = 1,
                      std::optional<std::vector<double>> shift_opt = std::nullopt,
                      int rqmc = 1,
                      int debug = 1) {

    int d = mu_arr.size();
    if (c1_arr.size() != d) throw std::invalid_argument("c1 size mismatch");
    auto L_info = L_arr.request();
    if (L_info.ndim != 2 || L_info.shape[0] != d || L_info.shape[1] != d) throw std::invalid_argument("L shape mismatch");

    const double* mu = static_cast<const double*>(mu_arr.data());
    const double* L = static_cast<const double*>(L_arr.data());
    const double* c1 = static_cast<const double*>(c1_arr.data());

    auto result = st_cdf_cpp(mu, L, c1, c2, d, n, seed, shift_opt, rqmc, debug);

    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(cpp_core, m) {
    m.def("sample_sov", &sample_sov_wrap, 
          py::arg("a"), py::arg("b"), py::arg("L"), py::arg("n"), 
          py::arg("seed") = 1, py::arg("spherical") = 1, py::arg("rqmc") = 1, 
          py::arg("shift") = py::cast(std::vector<double>()));
    m.def("joint_cdf_bivnormal", &joint_cdf_bivnormal);
    m.def("st_cdf", &st_cdf_wrap,
          py::arg("mu"), py::arg("L"), py::arg("c1"), py::arg("c2"), py::arg("n"), 
          py::arg("seed") = 1, py::arg("shift") = py::cast(std::vector<double>()), py::arg("rqmc") = 1, py::arg("debug") = 1);
}