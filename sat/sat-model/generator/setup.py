# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.core import setup, Extension

import os
import numpy

DEBUG = True

envpath = os.environ["CONDA_PREFIX"]
numpypath = numpy.__path__[0]

sources = [os.path.join('src', f) for f in filter(
    lambda x: x.endswith('.cpp') and x != "generator_main.cpp", os.listdir('src'))]

module = Extension(
    '_satgenerator',
    sources=sources,
    include_dirs=[
        os.path.join(envpath, "include"),
        os.path.join(numpypath, "core/include"), ],
    library_dirs=[
        os.path.join(envpath, "lib")],
    libraries=[],
    depends=["opencv", "numpy"],
    extra_compile_args=[('-O0' if DEBUG else '-O3'),
                        '-ggdb', '-std=gnu++11', '-Wall'],
    extra_link_args=[('-O0' if DEBUG else '-O3'),
                     '-ggdb', '-Wall'],
    define_macros=([('SAT_DEBUG', '')] if DEBUG else []))

setup(name='_satgenerator',
      version='1.0',
      description='_satgenerator',
      ext_modules=[module])
