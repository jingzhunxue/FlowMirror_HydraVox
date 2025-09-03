.PHONY: test-server-10000

kill-server-10000:
	@echo "Killing existing uvicorn processes listening on port 10000..."
	@lsof -i :10000 | awk 'NR>1 {print $$2}' | xargs -r sudo kill -9

test-server-10000: kill-server-10000
	nohup uvicorn main:app --host 0.0.0.0 --root-path /api/v1/math --proxy-headers --forwarded-allow-ips='*' --port 10000 --access-log --use-colors >> log.log 2>&1 &

.PHONY: format
format:
	black .
	isort .
