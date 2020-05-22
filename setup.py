from setuptools import setup, find_packages


import evaluation_framework

VERSION = evaluation_framework.__version__

# with open("README.rst", "r") as fh:
#     long_description = fh.read()

setup(
	name="ml-evaluation-framework", 
	version=VERSION,
	author="Jay Kim",
	description="",
	# long_description=long_description,
	# long_description_content_type="text/x-rst",
	url=None,
	license="DSB 3-clause",
	packages=find_packages(),
	# install_requires=["graphviz>=0.13.2"]
	)

# python setup.py bdist_wheel
# scp dist/evaluation_framework-0.0b0-py3-none-any.whl jkim2@10.174.62.128:/mnt/grubhub-gdp-delivery-data-assets-dev/notebooks/jkim2/model_experiments/evaluation_framework