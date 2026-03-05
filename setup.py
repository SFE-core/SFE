from setuptools import setup, find_packages

setup(
    name="sfe",
    version="0.1.0",
    description="Observability layer for coupling structure in multivariate time series",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jesus David Calderas Cervantes",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    extras_require={"dev": ["pytest", "pandas", "matplotlib", "scipy"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Paper":  "https://doi.org/10.5281/zenodo.18869381",
        "Source": "https://github.com/SFE-core/SFE",
    },
)
