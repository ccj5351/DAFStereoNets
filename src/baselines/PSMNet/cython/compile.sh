MY_PYTHON=python3.7
rm ./__init__.py
rm ./__init__.pyc
rm *.so
rm *.c
$MY_PYTHON ./setup_KT15ErrLogColor.py clean
rm -rf build
$MY_PYTHON ./setup_KT15ErrLogColor.py build_ext --inplace


$MY_PYTHON ./setup_KT15FalseColor.py clean
rm -rf build
$MY_PYTHON ./setup_KT15FalseColor.py build_ext --inplace

cp ../__init__.py ./
