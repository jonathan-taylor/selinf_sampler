import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    from src.cpp_core import sample_sov, joint_cdf_bivnormal
    print("Successfully imported cpp_core")
    
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    L = np.eye(2)
    samples, weights = sample_sov(a, b, L, 10, seed=42)
    print("Samples:\n", samples)
    print("Weights:\n", weights)
    
    p = joint_cdf_bivnormal(0.0, 0.0, 0.5)
    print("Joint CDF (0,0,0.5):", p)
    # Theoretical value for (0,0,rho) is 0.25 + arcsin(rho)/(2*pi)
    # arcsin(0.5) = pi/6. 0.25 + (pi/6)/(2*pi) = 0.25 + 1/12 = 3/12 + 1/12 = 4/12 = 1/3 ~ 0.3333
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
