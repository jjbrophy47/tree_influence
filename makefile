get_deps:
	pip3 install -r requirements.txt

clean:
	cd tree_influence/explainers/parsers/; rm -rf *.so *.c *.html build/ __pycache__; cd -

build:
	cd tree_influence/explainers/parsers/; python3 setup.py build_ext --inplace; cd ..

all: clean get_deps build
