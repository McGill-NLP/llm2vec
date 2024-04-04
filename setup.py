from setuptools import setup, find_packages

package_name = "llm2vec"
version = {}
with open(f"{package_name}/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="llm2vec",
    version=version["__version__"],
    python_requires=">=3.8",
    packages=find_packages(include=[f"{package_name}*"]),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "peft",
        "transformers>=4.38.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
