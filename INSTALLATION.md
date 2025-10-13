### As a user on a prepared machine:

```bash
direnv allow .
```

Set git config user.name and user.email. 

### Python package managment:

Evals are listed as `project.optional-dependences` in `pyproject.toml`. To install them, run `uv sync --extra example.eval`.

