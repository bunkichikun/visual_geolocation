default: pylint

load_data_from_bucket:
	python -c 'from visual_geolocation.ml_logic.utils import load_data_from_bucket; load_data_from_bucket() '

run_main:
	python -m visual_geolocation.interface.main

baseline_perf_eval:
	python -c 'from visual_geolocation.interface.main import evaluate_baseline; evaluate_baseline() '

pylint:
	find . -iname "*.py" -not -path "./.git/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

reinstall_package:
	@pip uninstall -y visual_geoloc_package || :
	@pip install -e .

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
