import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="girypy", # Replace with your username
    version="0.0.1",
    author="Drew Allen McNeely",
    author_email="drew.mcneely@utexas.edu",
    description="A library for making Markov categories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drewmcneely/girypy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
