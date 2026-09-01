default: pylint

load_data_from_bucket:
	python -c 'from visual_geolocation.ml_logic.utils import load_data_from_bucket; load_data_from_bucket() '

preprocess:
	python -c 'from visual_geolocation.interface.main import preprocess_offline; preprocess_offline() '

preprocess_test:
	python -c 'from visual_geolocation.interface.main import preprocess_offline; preprocess_offline(which="test") '

docker_update:
	docker build  --platform linux/amd64   -t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/visual-geolocation/$(GAR_IMAGE):prod .

docker_update_and_run: docker_update
	docker run -it -e PORT=8000 -e GOOGLE_APPLICATION_CREDENTIALS=/gcp.json   -p 8000:8000 --volume ${GOOGLE_APPLICATION_CREDENTIALS}:/gcp.json:ro  --env-file .env $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/visual-geolocation/$(GAR_IMAGE):prod

docker_update_and_deploy: docker_update
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/visual-geolocation/$(GAR_IMAGE):prod
  gcloud --project=$(GCP_PROJECT) run deploy --image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/visual-geolocation/$(GAR_IMAGE):prod --memory $(GAR_MEMORY) --region $(GCP_REGION) --env-vars-file .env.yaml

run_main:
	python -m visual_geolocation.interface.main

baseline_perf_eval:
	python -c 'from visual_geolocation.interface.main import evaluate_baseline; evaluate_baseline() '

pylint:
	find . -iname "*.py" -not -path "./.git/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

reinstall_package:
	@pip uninstall -y visual_geoloc_package || :
	@pip install -e .

run_api:
	uvicorn visual_geolocation.api.fast:app --reload

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
