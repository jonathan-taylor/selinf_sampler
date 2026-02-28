import os
from setuptools import setup, Extension

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

ext_modules = [
    Extension(
        "cpp_core",
        ["cpp_core.cpp"],
        include_dirs=[
            get_numpy_include(),
            get_pybind_include(),
            get_pybind_include(user=True),
            "../third_party",
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    )
]

setup(
    name="selinf_sampler",
    version="0.1.0",
    setup_requires=["pybind11", "numpy"],
    install_requires=["pybind11", "numpy"],
    ext_modules=ext_modules,
)
