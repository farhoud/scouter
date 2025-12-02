.PHONY: evals neo4j-up eval-watch

evals:
	SCOUTER_ENV=development uv run pytest evals/ $(if $(LOGS),--log-cli-level=INFO -s,)

eval-watch:
	SCOUTER_ENV=development uv run pytest-watch ./src evals/ $(if $(LOGS),--log-cli-level=INFO -s,)

neo4j-up:
	docker-compose up neo4j -d