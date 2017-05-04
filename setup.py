from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tqdm==4.11.2'
]

setup(name='entity_networks',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True)
