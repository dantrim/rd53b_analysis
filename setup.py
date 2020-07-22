from setuptools import setup, find_packages

setup(
    name = "rd53b_analysis",
    version = "0.1",
    description = "",
    long_description = "",
    url = "",
    author = "Daniel Joseph Antrim",
    author_email = "daniel.joseph.antrim@cern.ch",
    package_dir = { "": "python" },
    packages = find_packages(where = "python"),
    include_package_data = True,
    python_requires = ">=3.6",
    install_requires = [
        "pre-commit",
        "flake8",
        "black",
        "Click>=6.0",
        "numpy",
        "matplotlib",
        "scipy",
        "jsonschema"
    ],
    entry_points = {"console_scripts": ["rd53b=analysis.cli:cli"]},
)
