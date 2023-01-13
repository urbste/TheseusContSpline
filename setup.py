import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='TheseusContSpline',  
     version='0.1',
     author="Steffen Urban",
     author_email="urbste@gmail.com",
     description="Fully differentiable continuous trajectory estimation in PyTorch",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/urbste/TheseusContSpline",
     packages=setuptools.find_packages(),
    #  classifiers=[
    #      "Programming Language :: Python :: 3",
    #      "License :: OSI Approved :: MIT License",
    #      "Operating System :: OS Independent",
    #  ],
 )