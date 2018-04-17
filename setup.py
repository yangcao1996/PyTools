import os
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize

exec(open('pytools/version.py').read())
exts = [Extension(name='pytools.nms',
                  sources=["pytools/nms/_nms.pyx", "pytools/nms/nms.c"],
                  include_dirs=[numpy.get_include()])
        ]
setup(name='pytools',
  version=__version__,
  description='python tools',
  url='http://kaiz.xyz/pytools',
  author_email='kaiz.xyz@gmail.com',
  license='MIT',
  packages=['pytools'],
  ext_modules=cythonize(exts),
  zip_safe=False
)
# build submodules
# submodules = ['nms']
# for subm in submodules:
  # os.system("cd pytools/%s && make && cd -" % (subm)) 
