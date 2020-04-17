import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="residual-sample-nn",
    version="0.0.1",
    author="Ren Yuan Xue, Pierre McWhannel",
    author_email="pmcwhannel@uwaterloo.ca, ryxue@uwaterloo.ca",
    description="An implementation of a residual sample neural net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinceLuong/residual-sample-nn/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)