#pragma once

#include <vector>
#include <optional>
#include <utility>

double joint_cdf_bivnormal(double h, double k, double rho);

std::pair<std::vector<double>, std::vector<double>> sample_sov_cpp(
    const double* a, 
    const double* b, 
    const double* L, 
    int d,
    int n,
    unsigned int seed = 1,
    int spherical = 1,
    int rqmc = 1,
    std::optional<std::vector<double>> shift_opt = std::nullopt);

std::pair<double, double> st_cdf_cpp(
    const double* mu, 
    const double* L, 
    const double* c1,
    double c2,
    int d,
    int n,
    unsigned int seed = 1,
    std::optional<std::vector<double>> shift_opt = std::nullopt,
    int rqmc = 1,
    int debug = 1);
