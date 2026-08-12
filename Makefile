.PHONY: help install lint format test cov clean

help: ## 显示全部可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## 安装全部依赖（uv sync，含 dev 扩展）
	uv sync --all-extras

lint: ## 静态检查（ruff check + 格式校验）
	uv run ruff check .
	uv run ruff format --check .

format: ## 自动格式化
	uv run ruff format .
	uv run ruff check --fix .

test: ## 运行全部测试
	uv run pytest

cov: ## 运行测试并输出覆盖率
	uv run pytest --cov=harness_agent --cov-report=term-missing

clean: ## 清理缓存与本地数据
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .data dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
