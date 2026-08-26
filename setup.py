"""Setup file for the Visual Geoloc Package to be installable by pip.
"""

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='visual_geoloc_package',
      version="0.0.1",
      description="Visual Geolocation Package (base package for explo)",
      license="MIT",
      author="Benoît Libeau, et al. see contributors",
      author_email="benoit.libeau@ensta.org",
      #url="https://github.com/lewagon/taxi-fare",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
