set quiet

# List available recipes
default:
    just --list

# Upgrade dependencies
upgrade:
    uv sync --upgrade
    uv export --quiet --no-hashes --no-dev --group docs --output-file docs/requirements.txt

# Serve the docs
[no-exit-message]
serve-docs:
    rm -rf docs/_build docs/generated
    uv run --with-requirements docs/requirements.txt sphinx-autobuild docs docs/_build
