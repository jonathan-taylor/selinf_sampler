project('selinf_sampler', 'cpp',
  version: run_command('python', '-c', 'import setuptools_scm; print(setuptools_scm.get_version())', check: true).stdout().strip(),
  default_options: ['cpp_std=c++17', 'warning_level=3'])

py = import('python').find_installation(pure: false)

inc_3rd = include_directories('third_party')

# Pybind11 dependency
pybind11_dep = dependency('pybind11', required: false)

if not pybind11_dep.found()
    pybind11_incdir = run_command(py,
      ['-c', 'import pybind11; print(pybind11.get_include())'],
      check: true
    ).stdout().strip()
    pybind11_dep = declare_dependency(include_directories: include_directories(pybind11_incdir))
endif


python_sources = [
  'python/core.py',
  'python/random_lasso.py',
  'python/utils.py',
]

py.install_sources(
  python_sources,
  subdir: 'src'
)

# Extension module
py.extension_module('cpp_core',
  ['src/pybind_core.cpp', 'src/cpp_core.cpp'],
  dependencies: [pybind11_dep],
  include_directories: inc_3rd,
  install: true,
  subdir: 'src'
)
