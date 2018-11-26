import os
import io
import re
import codecs
from setuptools import setup, find_packages

# List all of your Python package dependencies in the
# requirements.txt file

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readfile(filename, split=False):
    with io.open(filename, encoding="utf-8") as stream:
        if split:
            return stream.read().split("\n")
        return stream.read()


readme = readfile("README.rst", split=True)

# For requirements not hosted on PyPi place listings
# into the 'requirements.txt' file.
requires = [   # minimal requirements listing
    'numpy',
    'opencmiss.zinc',
]
source_license = readfile("LICENSE")


setup(
    name='sparc.electrodeprojection',
    version=find_version("src", "sparc", "electrodeprojection", "__init__.py"),
    description='',
    long_description='\n'.join(readme) + source_license,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
    ],
    author='Jesse Khorasanee',
    author_email='j.khorasanee@auckland.ac.nz',
    url='',
    license='APACHE',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=requires,
)
