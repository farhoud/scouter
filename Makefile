.PHONY: evals neo4j-up

evals:
	SCOUTER_ENV=development uv run pytest evals/ $(if $(LOGS),--log-cli-level=INFO -s,)

neo4j-up:
	docker-compose up neo4j -d