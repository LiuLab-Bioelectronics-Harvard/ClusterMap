from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
     name='ClusterMap',
     version='0.1',
     scripts=['ClusterMap'] ,
     author="Yichun He, Emma Bou Hanna, Jiahao Huang, Xin Tang",
     author_email="yichunhe00@gmail.com",
     description="A package for ClusterMap paper",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/LiuLab-Bioelectronics-Harvard/ClusterMap/",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
