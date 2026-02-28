#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/owens_t.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <optional>

namespace py = pybind11;

const double MACHINE_EPS = std::numeric_limits<double>::epsilon();
const double PI = std::acos(-1.0);

inline double ndtr(double x) {
    static boost::math::normal norm;
    return boost::math::cdf(norm, x);
}

inline double ndtri(double p) {
    static boost::math::normal norm;
    return boost::math::quantile(norm, p);
}

inline double owens_t(double h, double a) {
    return boost::math::owens_t(h, a);
}

py::tuple sample_sov(py::array_t<double> a_arr, 
                     py::array_t<double> b_arr, 
                     py::array_t<double> L_arr, 
                     int n,
                     unsigned int seed = 1,
                     int spherical = 1,
                     int rqmc = 1,
                     std::optional<py::array_t<double>> shift_opt = std::nullopt) {
    
    auto a = a_arr.unchecked<1>();
    auto b = b_arr.unchecked<1>();
    auto L = L_arr.unchecked<2>();
    int d = a.shape(0);

    std::vector<double> shift(d, 0.0);
    if (shift_opt) {
        auto s_u = shift_opt->unchecked<1>();
        for(int i=0; i<d; ++i) shift[i] = s_u(i);
    }

    py::array_t<double> samples({n, d});
    py::array_t<double> weights(n);
    auto samples_m = samples.mutable_unchecked<2>();
    auto weights_m = weights.mutable_unchecked<1>();

    boost::random::mt19937 rng(seed);
    boost::random::uniform_01<double> dist;

    std::vector<double> x(d);
    std::vector<double> u(d);

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < d; ++j) {
            u[j] = dist(rng);
            if (rqmc == 1) {
                u[j] = u[j] * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
            }
        }

        double w = 1.0;
        for (int j = 0; j < d; ++j) {
            double tmp = 0.0;
            for (int i = 0; i < j; ++i) {
                tmp += L(j, i) * x[i];
            }
            double aj_ = (a(j) - tmp) / L(j, j) - shift[j];
            double bj_ = (b(j) - tmp) / L(j, j) - shift[j];
            
            double lo = (aj_ > 0) ? (1.0 - ndtr(-aj_)) : ndtr(aj_);
            double hi = (bj_ > 0) ? (1.0 - ndtr(-bj_)) : ndtr(bj_);
            double factor = hi - lo;
            
            double uj_ = lo + factor * u[j];
            uj_ = uj_ * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
            
            if (uj_ > 0.5) {
                x[j] = -ndtri(1.0 - uj_) + shift[j];
            } else {
                x[j] = ndtri(uj_) + shift[j];
            }
            w *= std::exp(0.5 * shift[j] * shift[j] - x[j] * shift[j]) * factor;
        }

        if (spherical == 1) {
            for (int j = 0; j < d; ++j) samples_m(k, j) = x[j];
        } else {
            for (int ii = 0; ii < d; ++ii) {
                double tmp = 0.0;
                for (int jj = 0; jj < d; ++jj) {
                    tmp += L(ii, jj) * x[jj];
                }
                samples_m(k, ii) = tmp;
            }
        }
        weights_m(k) = w;
    }
    return py::make_tuple(samples, weights);
}

double joint_cdf_bivnormal(double h, double k, double rho) {
    double ph = (h < 0) ? ndtr(h) : (1.0 - ndtr(-h));
    double pk = (k < 0) ? ndtr(k) : (1.0 - ndtr(-k));
    double m_hk = std::min(h, k);
    double phk = (m_hk < 0) ? ndtr(m_hk) : (1.0 - ndtr(-m_hk));
    
    if (rho == 0.0) return ph * pk;
    if (rho == 1.0) return phk;
    if (rho == -1.0) return (h + k >= 0) ? (ph + pk - 1.0) : 0.0;
    if (h == 0.0 && k == 0.0) return 0.25 + std::asin(rho) / (2.0 * PI);

    double rho2 = std::sqrt(1.0 - rho * rho);
    double J = ((h * k > 0) || (h * k == 0 && h + k >= 0)) ? 0.0 : 1.0;
    
    return 0.5 * ph + 0.5 * pk - owens_t(h, (k - rho * h) / (h * rho2)) - owens_t(k, (h - rho * k) / (k * rho2)) - 0.5 * J;
}

py::tuple st_cdf(py::array_t<double> mu_arr, 
                 py::array_t<double> L_arr, 
                 py::array_t<double> c1_arr,
                 double c2,
                 int n,
                 unsigned int seed,
                 std::optional<py::array_t<double>> shift_opt = std::nullopt,
                 int rqmc = 1,
                 int debug = 1) {

    auto mu = mu_arr.unchecked<1>();
    auto L = L_arr.unchecked<2>();
    auto c1 = c1_arr.unchecked<1>();
    int d = mu.shape(0);

    std::vector<double> shift(d, 0.0);
    if (shift_opt) {
        auto s_u = shift_opt->unchecked<1>();
        for(int i=0; i<d; ++i) shift[i] = s_u(i);
    }

    boost::random::mt19937 rng(seed);
    boost::random::uniform_01<double> dist;

    std::vector<double> weights(n);
    std::vector<double> numerators(n);

    std::vector<double> x(d);
    std::vector<double> u(d);

    std::vector<double> c1_t(d, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            c1_t[i] += L(j, i) * c1(j); // L^T @ c1
        }
    }

    double c2_t = c2;
    for (int j = 0; j < d; ++j) {
        c2_t += c1(j) * mu(j);
    }
    double c1_ = c1_t[d - 1];

    std::vector<double> a(d);
    for(int i=0; i<d; ++i) a[i] = -mu(i);

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < d; ++j) {
            u[j] = dist(rng);
            if (rqmc == 1) u[j] = u[j] * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
        }

        double w = 1.0;
        double num = 0.0;
        
        for (int j = 0; j < d; ++j) {
            double tmp = 0.0;
            for (int i = 0; i < j; ++i) tmp += L(j, i) * x[i];
            
            double aj_ = (a[j] - tmp) / L(j, j) - shift[j];
            double lo = (aj_ > 0) ? (1.0 - ndtr(-aj_)) : ndtr(aj_);
            double factor = 1.0 - lo;

            if (j < d - 1) {
                double uj_ = lo + factor * u[j];
                uj_ = uj_ * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
                
                if (uj_ > 0.5) {
                    x[j] = -ndtri(1.0 - uj_) + shift[j];
                } else {
                    x[j] = ndtri(uj_) + shift[j];
                }
                w *= std::exp(0.5 * shift[j] * shift[j] - x[j] * shift[j]) * factor;
            } else {
                double c2_ = c2_t;
                for (int i = 0; i < d - 1; ++i) c2_ += c1_t[i] * x[i];
                num = w * joint_cdf_bivnormal(-aj_, c2_ / std::sqrt(1.0 + c1_ * c1_), c1_ / std::sqrt(1.0 + c1_ * c1_));
                w *= factor;
            }
        }
        weights[k] = w;
        numerators[k] = num;
    }

    double mean_num = 0.0, mean_weights = 0.0;
    for(int k=0; k<n; ++k) {
        mean_num += numerators[k];
        mean_weights += weights[k];
    }
    mean_num /= n;
    mean_weights /= n;

    return py::make_tuple(mean_num, mean_weights);
}

py::array_t<double> Gibbs(py::array_t<double> mu_arr, 
                          py::array_t<double> Sigma_arr, 
                          py::array_t<double> initial_arr, 
                          int maxit, 
                          unsigned int seed, 
                          int PC_every = -1, 
                          std::optional<py::array_t<double>> eigval_opt = std::nullopt,
                          std::optional<py::array_t<double>> PCs_opt = std::nullopt,
                          std::optional<py::array_t<double>> eta_opt = std::nullopt, 
                          int eta_every = -1) {

    auto mu = mu_arr.unchecked<1>();
    auto Sigma = Sigma_arr.unchecked<2>();
    auto initial = initial_arr.unchecked<1>();
    int d = mu.shape(0);

    boost::random::mt19937 rng(seed);
    boost::random::uniform_01<double> udist;
    boost::random::uniform_int_distribution<> coord_dist(0, d - 1);

    int nPC = 0;
    std::vector<double> eigval;
    std::vector<double> eigval_probs;
    std::vector<std::vector<double>> PCs_vec;
    if (PC_every > 0 && eigval_opt && PCs_opt) {
        auto ev_u = eigval_opt->unchecked<1>();
        nPC = ev_u.shape(0);
        double sum_eig = 0.0;
        for(int i=0; i<nPC; ++i) sum_eig += ev_u(i);
        for(int i=0; i<nPC; ++i) {
            eigval.push_back(ev_u(i));
            eigval_probs.push_back(ev_u(i) / sum_eig);
        }
        auto pcs_u = PCs_opt->unchecked<2>();
        for(int i=0; i<nPC; ++i) {
            std::vector<double> row;
            for(int j=0; j<d; ++j) row.push_back(pcs_u(i, j));
            PCs_vec.push_back(row);
        }
    }
    
    std::optional<boost::random::discrete_distribution<>> pc_dist;
    if (!eigval_probs.empty()) {
        pc_dist.emplace(eigval_probs.begin(), eigval_probs.end());
    }

    std::vector<double> x_t(d);
    for(int i=0; i<d; ++i) x_t[i] = initial(i);

    std::vector<std::vector<double>> cond_mean_factors(d, std::vector<double>(d - 1, 0.0));
    std::vector<double> cond_sigmas(d, 0.0);

    py::module_ np = py::module_::import("numpy");
    py::module_ nplinalg = py::module_::import("numpy.linalg");
    
    for (int i = 0; i < d; ++i) {
        std::vector<int> i_indices;
        for(int j=0; j<d; ++j) if(j != i) i_indices.push_back(j);
        
        py::array_t<double> Sigma_i__i_({d - 1, d - 1});
        py::array_t<double> Sigma_i_i_({d - 1});
        py::array_t<double> Sigma_i__i({d - 1});
        
        auto S_ii_m = Sigma_i__i_.mutable_unchecked<2>();
        auto S_i_m = Sigma_i_i_.mutable_unchecked<1>();
        auto S__i_m = Sigma_i__i.mutable_unchecked<1>();
        
        for (int r = 0; r < d - 1; ++r) {
            S_i_m(r) = Sigma(i, i_indices[r]);
            S__i_m(r) = Sigma(i_indices[r], i);
            for (int c = 0; c < d - 1; ++c) {
                S_ii_m(r, c) = Sigma(i_indices[r], i_indices[c]);
            }
        }
        
        py::object inv_Sigma_i__i_ = nplinalg.attr("inv")(Sigma_i__i_);
        
        py::array_t<double> dot1 = np.attr("dot")(Sigma_i_i_, inv_Sigma_i__i_).cast<py::array_t<double>>();
        auto dot1_u = dot1.unchecked<1>();
        for(int j=0; j<d-1; ++j) cond_mean_factors[i][j] = dot1_u(j);
        
        py::object dot2 = np.attr("dot")(dot1, Sigma_i__i);
        cond_sigmas[i] = std::sqrt(Sigma(i, i) - dot2.cast<double>());
    }

    double cond_sigma_eta = 0.0;
    std::vector<double> cond_mean_factor_eta(d, 0.0);
    std::vector<double> eta_vec;
    if (eta_opt) {
        auto eta_u = eta_opt->unchecked<1>();
        for(int i=0; i<d; ++i) eta_vec.push_back(eta_u(i));
        
        py::object inv_Sigma = nplinalg.attr("inv")(Sigma_arr);
        py::object dot1 = np.attr("dot")(Sigma_arr, *eta_opt);
        cond_sigma_eta = std::sqrt(np.attr("dot")(*eta_opt, dot1).cast<double>());
        py::array_t<double> dot2 = np.attr("dot")(inv_Sigma, *eta_opt).cast<py::array_t<double>>();
        auto dot2_u = dot2.unchecked<1>();
        for(int i=0; i<d; ++i) cond_mean_factor_eta[i] = -dot2_u(i) * cond_sigma_eta * cond_sigma_eta;
    }

    py::array_t<double> samples({maxit, d});
    auto samples_m = samples.mutable_unchecked<2>();

    int flag_eta = 0, flag_PC = 0;

    for (int t = 0; t < maxit; ++t) {
        bool docoord = true;
        bool doeta = false;
        std::vector<double> theta_vec;
        double cond_mu = 0.0, cond_sigma = 0.0;
        
        flag_eta++;
        flag_PC++;

        if (eta_every > 0 && flag_eta == eta_every) {
            doeta = true; docoord = false;
            theta_vec = eta_vec;
            flag_eta = 0;
        } else if (PC_every > 0 && flag_PC == PC_every) {
            doeta = false; docoord = false;
            int pidx = (*pc_dist)(rng);
            theta_vec = PCs_vec[pidx];
            cond_sigma = std::sqrt(eigval[pidx]);
            flag_PC = 0;
        }

        double u = udist(rng);

        if (docoord) {
            int idx = coord_dist(rng);
            cond_mu = mu(idx);
            for (int j = 0; j < d; ++j) {
                if (j < idx) cond_mu += cond_mean_factors[idx][j] * (x_t[j] - mu(j));
                else if (j > idx) cond_mu += cond_mean_factors[idx][j - 1] * (x_t[j] - mu(j));
            }
            cond_sigma = cond_sigmas[idx];

            double lower_u = -cond_mu / cond_sigma;
            lower_u = (lower_u < 0) ? ndtr(lower_u) : (1.0 - ndtr(-lower_u));
            double upper_u = 1.0;
            double u_ = lower_u + (upper_u - lower_u) * u;
            u_ = u_ * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
            double zz = (u_ < 0.5) ? ndtri(u_) : -ndtri(1.0 - u_);
            x_t[idx] = cond_mu + cond_sigma * zz;
        } else {
            if (doeta) {
                cond_mu = 0.0;
                for (int j = 0; j < d; ++j) cond_mu += cond_mean_factor_eta[j] * (x_t[j] - mu(j));
                cond_sigma = cond_sigma_eta;
            } else {
                // PC direction: cond_sigma already set, need cond_mu
                cond_mu = 0.0;
                for (int j = 0; j < d; ++j) cond_mu += -theta_vec[j] * (x_t[j] - mu(j));
            }

            double lower = -1e12, upper = 1e12;
            for (int j = 0; j < d; ++j) {
                if (theta_vec[j] != 0) {
                    double val = -x_t[j] / theta_vec[j];
                    if (theta_vec[j] > 0) {
                        if (lower < val) lower = val;
                    } else {
                        if (upper > val) upper = val;
                    }
                }
            }

            if (lower < upper) {
                double lower_u = (lower - cond_mu) / cond_sigma;
                lower_u = (lower_u < 0) ? ndtr(lower_u) : (1.0 - ndtr(-lower_u));
                double upper_u = (upper - cond_mu) / cond_sigma;
                upper_u = (upper_u < 0) ? ndtr(upper_u) : (1.0 - ndtr(-upper_u));
                
                double u_ = lower_u + (upper_u - lower_u) * u;
                u_ = u_ * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
                double zz = (u_ < 0.5) ? ndtri(u_) : -ndtri(1.0 - u_);
                double step = cond_mu + cond_sigma * zz;
                for (int j = 0; j < d; ++j) x_t[j] += theta_vec[j] * step;
            }
        }
        for(int i=0; i<d; ++i) samples_m(t, i) = x_t[i];
    }

    return samples;
}

PYBIND11_MODULE(cpp_core, m) {
    m.def("sample_sov", &sample_sov, 
          py::arg("a"), py::arg("b"), py::arg("L"), py::arg("n"), 
          py::arg("seed") = 1, py::arg("spherical") = 1, py::arg("rqmc") = 1, 
          py::arg("shift") = py::none());
    m.def("joint_cdf_bivnormal", &joint_cdf_bivnormal);
    m.def("st_cdf", &st_cdf,
          py::arg("mu"), py::arg("L"), py::arg("c1"), py::arg("c2"), py::arg("n"), 
          py::arg("seed") = 1, py::arg("shift") = py::none(), py::arg("rqmc") = 1, py::arg("debug") = 1);
    m.def("Gibbs", &Gibbs,
          py::arg("mu"), py::arg("Sigma"), py::arg("initial"), py::arg("maxit"), 
          py::arg("seed"), py::arg("PC_every") = -1, py::arg("eigval") = py::none(), 
          py::arg("PCs") = py::none(), py::arg("eta") = py::none(), py::arg("eta_every") = -1);
}
