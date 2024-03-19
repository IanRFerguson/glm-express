#!/bin/python
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(name='glm_express',
      version='1.8.9',
      description='Automated linear models for functional neuroimaging data',
      long_description=README,
      long_description_content_type="text/markdown",
      url='https://github.com/IanRFerguson/glm-express',
      author='Ian Richard Ferguson',
      author_email='IRF229@nyu.edu',
      license="MIT",
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7"
      ],
      packages=[
            'glm_express',
            'glm_express/aggregator',
            'glm_express/group_level',
            'glm_express/subject',
            'glm_express/rest',
            'glm_express/utils'
            ],
      include_package_data=True,
      install_requires=[
            'nilearn', 'pandas', 'tqdm', 'matplotlib', 'numpy', 'pybids', 'nibabel', 'nltools'
            ]
      )
