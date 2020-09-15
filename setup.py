from glob import glob
from os.path import basename, splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="TorchUtils",
    version="0.1.0",
    license="MIT",
    description="Utilities for PyTorch",
    author="Akagawa Daisuke",
    url="https://www.github.com/Akasan",
    packages=find_packages("TorchUtils"),
    package_dir={"": "TorchUtils"},
    # py_modules=[splitext(basename(path))[0] for path in glob("TorchUtils/*.py")],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file("requirements.txt"),
    # setup_requires=["pytest-runner"],
    # tests_require=["pytest", "pytest-cov"]
)
