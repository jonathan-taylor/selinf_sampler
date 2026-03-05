#include "cpp_core.hpp"
#include <random>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/owens_t.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/sobol.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

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

// JoeKuoScrambledSobol uses LMS scrambling identical to the user's FastScrambledSobol 
// and matches Scipy's Owen scrambling technique for the Sobol sequence.
template<size_t MaxDims>
class JoeKuoScrambledSobol {
private:
    boost::random::sobol_engine<uint32_t, 32> engine;
    uint32_t L[MaxDims][32]; // Scrambling matrices
    uint32_t V[MaxDims];     // Digital shifts
    size_t current_dims;

public:
    JoeKuoScrambledSobol(size_t dims, uint32_t seed = 42) 
        : engine(dims), current_dims(dims) {
        
        if (dims > MaxDims) 
            throw std::runtime_error("Dimensions exceed MaxDims.");

        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> dist;

        for (size_t d = 0; d < dims; ++d) {
            V[d] = dist(rng);
            for (int i = 0; i < 32; ++i) {
                uint32_t mask = (1U << i) - 1;
                L[d][i] = (dist(rng) & mask) | (1U << i);
            }
        }
    }

    void next_point(double* out) {
        for (size_t d = 0; d < current_dims; ++d) {
            uint32_t x = engine(); 
            uint32_t s = 0;

            for (int i = 0; i < 32; ++i) {
                s ^= (L[d][i] * ((x >> i) & 1));
            }
            
            s ^= V[d];
            out[d] = s * 2.3283064365386963e-10; 
        }
    }
};

std::pair<std::vector<double>, std::vector<double>> sample_sov_cpp(
    const double* a, 
    const double* b, 
    const double* L, 
    int d,
    int n,
    unsigned int seed,
    int spherical,
    int rqmc,
    std::optional<std::vector<double>> shift_opt) {

    std::vector<double> shift(d, 0.0);
    if (shift_opt) {
        shift = shift_opt.value();
    }

    std::vector<double> samples_out(n * d);
    std::vector<double> weights_out(n);

    std::vector<std::vector<double>> U(n, std::vector<double>(d));
    if (rqmc == 1) {
        JoeKuoScrambledSobol<1000> sobol_gen(d, seed);
        std::vector<double> pt(d);
        for (int i = 0; i < n; ++i) {
            sobol_gen.next_point(pt.data());
            for (int j = 0; j < d; ++j) {
                double val = pt[j];
                U[i][j] = val * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
            }
        }
    } else {
        boost::random::mt19937 rng(seed);
        boost::random::uniform_01<double> udist;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                U[i][j] = udist(rng);
            }
        }
    }

    std::vector<double> x(d);
    std::vector<double> u(d);

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < d; ++j) {
            u[j] = U[k][j];
        }

        double w = 1.0;
        for (int j = 0; j < d; ++j) {
            double tmp = 0.0;
            for (int i = 0; i < j; ++i) {
                tmp += L[j * d + i] * x[i];
            }
            double aj_ = (a[j] - tmp) / L[j * d + j] - shift[j];
            double bj_ = (b[j] - tmp) / L[j * d + j] - shift[j];
            
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
            for (int j = 0; j < d; ++j) {
                samples_out[k * d + j] = x[j];
            }
        } else {
            for (int ii = 0; ii < d; ++ii) {
                double tmp = 0.0;
                for (int jj = 0; jj < d; ++jj) {
                    tmp += L[ii * d + jj] * x[jj];
                }
                samples_out[k * d + ii] = tmp;
            }
        }
        weights_out[k] = w;
    }
    return {std::move(samples_out), std::move(weights_out)};
}

std::pair<double, double> st_cdf_cpp(
    const double* mu, 
    const double* L, 
    const double* c1,
    double c2,
    int d,
    int n,
    unsigned int seed,
    std::optional<std::vector<double>> shift_opt,
    int rqmc,
    int debug) {

    std::vector<double> shift(d, 0.0);
    if (shift_opt) {
        shift = shift_opt.value();
    }

    std::vector<std::vector<double>> U(n, std::vector<double>(d - 1));
    if (rqmc == 0) {
        boost::random::mt19937 rng(seed);
        boost::random::uniform_01<double> udist;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d - 1; ++j) {
                U[i][j] = udist(rng);
            }
        }
    } else {
        if (debug == 1) {
            JoeKuoScrambledSobol<1000> sobol_gen(d, seed);
            std::vector<double> pt(d);
            for (int i = 0; i < n; ++i) {
                sobol_gen.next_point(pt.data());
                for (int j = 0; j < d - 1; ++j) {
                    double val = pt[j];
                    U[i][j] = val * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
                }
                // last dimension (pt[d-1]) is dropped automatically
            }
        } else {
            JoeKuoScrambledSobol<1000> sobol_gen(d - 1, seed);
            std::vector<double> pt(d - 1);
            for (int i = 0; i < n; ++i) {
                sobol_gen.next_point(pt.data());
                for (int j = 0; j < d - 1; ++j) {
                    double val = pt[j];
                    U[i][j] = val * (1.0 - MACHINE_EPS) + 0.5 * MACHINE_EPS;
                }
            }
        }
    }

    std::vector<double> weights(n);
    std::vector<double> numerators(n);

    std::vector<double> x(d);
    std::vector<double> u(d);

    std::vector<double> c1_t(d, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            c1_t[i] += L[j * d + i] * c1[j]; // L^T @ c1
        }
    }

    double c2_t = c2;
    for (int j = 0; j < d; ++j) {
        c2_t += c1[j] * mu[j];
    }
    double c1_ = c1_t[d - 1];

    std::vector<double> a(d);
    for(int i=0; i<d; ++i) a[i] = -mu[i];

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < d - 1; ++j) {
            u[j] = U[k][j];
        }

        double w = 1.0;
        double num = 0.0;
        
        for (int j = 0; j < d; ++j) {
            double tmp = 0.0;
            for (int i = 0; i < j; ++i) tmp += L[j * d + i] * x[i];
            
            double aj_ = (a[j] - tmp) / L[j * d + j] - shift[j];
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

    return {mean_num, mean_weights};
}
