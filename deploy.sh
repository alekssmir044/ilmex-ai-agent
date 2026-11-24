#!/usr/bin/env bash
# Повторный деплой агента data-analyst на Mistral Cloud.
# Ключ берётся из .env (MISTRAL_API_KEY). Деплой собирает запушенный коммит,
# поэтому сначала git push.
set -euo pipefail
cd "$(dirname "$0")"
git pull --ff-only
uv run --env-file .env agents deploy --yes
echo
echo "Чат с агентом: uv run --env-file .env agents chat data-analyst"
