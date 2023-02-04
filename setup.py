import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchconvview",
    version="0.2.0",
    author="Paul Gavrikov",
    author_email="paul.gavrikov@hs-offenburg.de",
    description="A library for PyTorch convolution layer visualizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulgavrikov/torchconvview",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    python_requires=">=3.6",
)