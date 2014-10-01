from distutils.core import setup

setup(
    name='jsonrpc20',
    version=__import__("jsonrpc20").__version__,
    packages=['jsonrpc20'],
    url='https://github.com/jet9/jsonrpc20',
    license='BSD',
    author='Dmitry Kurbatov',
    author_email='dk@dimcha.ru',
    requires=["ndict", "jsonschema"],
    description='Yet another jsonrpc 2.0 python implementation with wsgi support',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
    ]

)
