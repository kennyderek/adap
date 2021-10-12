from setuptools import setup

setup(name='adapenvs',
      version='0.0.1',
      install_requires=['gym', 'ray[rllib]==1.7.0', 'matplotlib', 'tqdm'] #And any other dependencies required
)