import setuptools
setuptools.setup(name='spym',
        version='0.1',
        description='A package to help analysis common ultrafast optics stuff.',
        url='https://github.uci.edu/stanfiem/spym',
        author='stanfiem',
        packages=setuptools.find_packages(),
		include_package_data=True,
        install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'pynlo',
	    'pandas',
            ],
        zip_safe=False)
