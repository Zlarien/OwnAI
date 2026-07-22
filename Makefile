# OwnAI developer tasks. Everything goes through `python -m ...` so it works
# the same on Windows, macOS and Linux without extra tooling. Swap DOMAIN to
# target a different domain config; the code never changes, only the YAML.
DOMAIN ?= domains/mario-wii.yaml

.DEFAULT_GOAL := help
.PHONY: help install test demo chat serve clean

help: ## List the available targets
	@python -c "print('OwnAI tasks:\n  install  editable install with dev extras\n  test     run the test suite\n  demo     ingest+index+eval the mario-wii domain, then hint at chat\n  chat     chat with the domain AI in the terminal\n  serve    launch the local web chat UI\n  clean    remove build artifacts')"

install: ## Editable install with dev dependencies
	python -m pip install -e ".[dev]"

test: ## Run the full test suite
	python -m pytest -q

demo: ## Ingest, index and evaluate the mario-wii domain
	python -m ownai.cli ingest --domain $(DOMAIN)
	python -m ownai.cli index --domain $(DOMAIN)
	python -m ownai.cli eval --domain $(DOMAIN)
	@python -c "print('\nDemo ready. Try: make chat   (or: python examples/quickstart.py)')"

chat: ## Chat with the domain AI in the terminal (extractive)
	python -m ownai.cli chat --domain $(DOMAIN)

serve: ## Serve the local web chat UI at http://127.0.0.1:8000
	python -c "from ownai.web import serve; serve('$(DOMAIN)')"

clean: ## Remove generated artifacts and caches
	python -c "import shutil,glob; [shutil.rmtree(p, ignore_errors=True) for p in ['artifacts','build','ownai.egg-info']+glob.glob('**/__pycache__', recursive=True)+glob.glob('.pytest_cache')]"
