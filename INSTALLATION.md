### As a user on a prepared machine:

```bash
cp .env.example .env
cp .envrc.example .envrc
direnv allow .
```

Set git config user.name and user.email. 

### Python package managment:

Evals are listed as `project.optional-dependences` in `pyproject.toml`. To install them, run `uv sync --extra example.eval`.

