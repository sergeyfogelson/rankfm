import os
import sys
from setuptools import Extension, setup

NAME = 'rankfm'
VERSION = '0.4.1'

# define the extension packages to include
# ----------------------------------------

# the C source is always generated from the .pyx at build time (Cython is a build requirement in pyproject.toml),
# so there is no checked-in copy that can drift out of sync with it
try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython>=3.0 is required to build rankfm: pip install cython")


# add compiler arguments to optimize machine code and enable POSIX threads
if sys.platform == 'win32':
    compile_args = ['/O2']
    link_args = []
else:
    disabled_warnings = ['-Wno-unused-function', '-Wno-uninitialized']
    compile_args = ['-O3', '-ffast-math', '-fno-finite-math-only', '-funroll-loops', '-pthread'] + disabled_warnings
    # RANKFM_MARCH picks an explicit instruction-set baseline, e.g. x86-64-v3 (AVX2 + FMA). prefer it for wheels
    # and container images: RANKFM_ARCH_NATIVE targets whatever CPU runs the build, and the result crashes with
    # an illegal-instruction error on any machine whose CPU lacks those instructions.
    if os.environ.get('RANKFM_MARCH', ''):
        compile_args.append('-march={}'.format(os.environ['RANKFM_MARCH']))
    elif os.environ.get('RANKFM_ARCH_NATIVE', '') == '1':
        compile_args.append('-march=native')
    link_args = ['-pthread']

# define the _rankfm extension using fast_ops.h
extensions = [
    Extension(
        name='rankfm._rankfm',
        sources=['rankfm/_rankfm.pyx'],
        include_dirs=['rankfm'],
        depends=['rankfm/fast_ops.h'],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

extensions = cythonize(extensions)

# define the main package setup function
# --------------------------------------

setup(
    name=NAME,
    version=VERSION,
    description='a python implementation of the generic factorization machines model class '
                'adapted for collaborative filtering recommendation problems '
                'with implicit feedback user-item interaction data '
                'and (optionally) additional user/item side features',
    author='Eric Lundquist',
    author_email='e.t.lundquist@gmail.com',
    maintainer='Sergey Fogelson',
    url='https://github.com/sergeyfogelson/rankfm',
    keywords=['machine', 'learning', 'recommendation', 'factorization', 'machines', 'implicit'],
    license='GNU General Public License v3.0',
    packages=['rankfm'],
    ext_modules=extensions,
    zip_safe=False,
    python_requires='>=3.9',
    install_requires=['numpy>=1.15', 'pandas>=0.24'],
    extras_require={'blas': ['threadpoolctl>=3.0']}
)

