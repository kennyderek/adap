from setuptools import setup

setup(name='adap',
      version='0.0.2',
      install_requires=['gym', 'ray', 'ray[rllib]', 'adapenvs', 'torch', 'tensorflow', 'pyglet'] #And any other dependencies required
)