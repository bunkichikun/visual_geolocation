default: pylint

pylint:
	find . -iname "*.py" -not -path "./.git/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
