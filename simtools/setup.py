from setuptools import setup

setup(name='simtools',
      version='0.1',
      description='Useful tools in the context of tumor data simulations.',
      url='http://github.com/hdetering/simtools',
      author='Harald Detering',
      author_email='harald.detering@gmail.com',
      license='GPL-3.0',
      packages=['simtools'],
      install_requires=[
          'argparse',
          'numpy',
          'pandas',
          'pyaml',
          'uuid'
      ],
      zip_safe=False,
      include_package_data=True)