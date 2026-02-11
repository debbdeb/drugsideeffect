from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "readme.md").read_text(encoding="utf-8")

setup(
    name="drugsideeffect_test",
    version="0.1.1",
    author="Briti Deb",
    author_email="britideb@gmail.com",
    description="Visualize side effects from textual data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debbdeb/drugsideeffect",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "plotly>=5.11",
        "nltk>=3.8",
        "textblob>=0.17",
        "spacy>=3.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_data={
        "drugsideeffect": ["models/*.pkl"],
    },
    zip_safe=False,
)
