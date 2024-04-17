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
    author="McGill NLP",
    author_email=f"parishad.behnamghader@mila.quebec",
    url=f"https://github.com/McGill-NLP/{package_name}",
    description=f"The official {package_name} library",
    python_requires=">=3.8",
    packages=find_packages(include=[f"{package_name}*"]),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "peft",
        "transformers>=4.39.1",
        "datasets",
        "evaluate",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
)
