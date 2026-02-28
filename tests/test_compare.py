import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

from src import cpp_core
from src import cython_core

def test_joint_cdf_bivnormal():
    print("Testing joint_cdf_bivnormal...")
    test_cases = [
        (0.5, -0.2, 0.4),
        (0.0, 0.0, 0.5),
        (-1.0, 1.0, -0.8),
        (2.0, 2.0, 0.99),
    ]
    for h, k, rho in test_cases:
        res_cython = cython_core.joint_cdf_bivnormal(h, k, rho)
        res_cpp = cpp_core.joint_cdf_bivnormal(h, k, rho)
        assert np.isclose(res_cython, res_cpp), f"Mismatch at ({h},{k},{rho}): Cython {res_cython} vs C++ {res_cpp}"
    print("Passed.")

def test_sample_sov():
    print("Testing sample_sov...")
    d = 3
    n = 1000
    a = np.array([-1.0, 0.0, 0.5])
    b = np.array([1.0, 2.0, 3.0])
    
    # Fixed L
    np.random.seed(42)
    L = np.tril(np.random.randn(d, d)) + np.eye(d) * 2
    
    samples_cy, weights_cy = cython_core.sample_sov(a, b, L, n, seed=42, spherical=1, rqmc=0)
    samples_cpp, weights_cpp = cpp_core.sample_sov(a, b, L, n, seed=42, spherical=1, rqmc=0)
    
    assert samples_cy.shape == samples_cpp.shape, "Shape mismatch in samples"
    assert weights_cy.shape == weights_cpp.shape, "Shape mismatch in weights"
    
    print(f"Cython mean weight: {np.mean(weights_cy):.6f}")
    print(f"C++ mean weight:    {np.mean(weights_cpp):.6f}")
    
    assert np.isclose(np.mean(weights_cy), np.mean(weights_cpp), atol=0.1)
    print("Passed.")

def test_st_cdf():
    print("Testing st_cdf...")
    d = 3
    n = 1000
    np.random.seed(42)
    mu = np.array([0.1, -0.2, 0.3])
    L = np.tril(np.random.randn(d, d)) + np.eye(d) * 2
    c1 = np.array([0.5, -0.5, 0.1])
    c2 = 0.2
    
    num_cy, den_cy = cython_core.st_cdf(mu, L, c1, c2, n, seed=42, rqmc=0)
    num_cpp, den_cpp = cpp_core.st_cdf(mu, L, c1, c2, n, seed=42, rqmc=0)
    
    print(f"Cython num / den: {num_cy:.6f} / {den_cy:.6f}")
    print(f"C++ num / den:    {num_cpp:.6f} / {den_cpp:.6f}")
    
    assert np.isclose(num_cy, num_cpp, atol=0.1)
    assert np.isclose(den_cy, den_cpp, atol=0.1)
    print("Passed.")

def test_gibbs():
    print("Testing Gibbs...")
    d = 3
    n = 1000
    np.random.seed(42)
    mu = np.array([0.1, -0.2, 0.3])
    Sigma = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    initial = np.array([0.5, 0.5, 0.5])
    
    samples_cy = cython_core.Gibbs(mu, Sigma, initial, n, seed=42)
    samples_cpp = cpp_core.Gibbs(mu, Sigma, initial, n, seed=42)
    
    print(f"Cython samples mean: {np.mean(samples_cy, axis=0)}")
    print(f"C++ samples mean:    {np.mean(samples_cpp, axis=0)}")
    
    assert np.allclose(np.mean(samples_cy, axis=0), np.mean(samples_cpp, axis=0), atol=0.2)
    print("Passed.")

if __name__ == '__main__':
    test_joint_cdf_bivnormal()
    print("")
    test_sample_sov()
    print("")
    test_st_cdf()
    print("")
    test_gibbs()
    print("All tests passed successfully.")
