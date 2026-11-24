import os
import sys
import glob
from setuptools import Extension, setup

NAME = 'rankfm'
VERSION = '0.5.0'

# define the extension packages to include
# ----------------------------------------

# prefer cythonizing .pyx source when Cython is available; fall back to pre-generated C source otherwise
try:
    from Cython.Build import cythonize
    print("re-generating C source with cythonize...")
    use_cython = True
    ext = 'pyx'
except ImportError:
    if glob.glob('rankfm/_rankfm.c'):
        print("building extensions with pre-generated C source...")
        use_cython = False
        ext = 'c'
    else:
        raise RuntimeError("Cython is required to build rankfm from .pyx source")


# add compiler arguments to optimize machine code and enable POSIX threads
if sys.platform == 'win32':
    compile_args = ['/O2']
    link_args = []
else:
    disabled_warnings = ['-Wno-unused-function', '-Wno-uninitialized']
    compile_args = ['-O3', '-ffast-math', '-fno-finite-math-only', '-funroll-loops', '-pthread'] + disabled_warnings
    if os.environ.get('RANKFM_ARCH_NATIVE', '') == '1':
        compile_args.append('-march=native')
    link_args = ['-pthread']

# define the _rankfm extension using fast_ops.h
extensions = [
    Extension(
        name='rankfm._rankfm',
        sources=['rankfm/_rankfm.{ext}'.format(ext=ext)],
        include_dirs=['rankfm'],
        depends=['rankfm/fast_ops.h'],
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

# re-generate the C code if needed
if use_cython:
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
    url='https://github.com/etlundquist/rankfm',
    keywords=['machine', 'learning', 'recommendation', 'factorization', 'machines', 'implicit'],
    license='GNU General Public License v3.0',
    packages=['rankfm'],
    ext_modules=extensions,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=['numpy>=1.15', 'pandas>=0.24']
)

