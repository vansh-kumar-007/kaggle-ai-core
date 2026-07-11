# kaggle-ai-core

Shared core modules used by both `kaggle-ai-notebook-agent` and `kaggle-ai-model-agent`: Kaggle API wrapper, Nemotron client, two-stage dataset selection/profiling, checkpointing, and structured run logging.

Consumed as a git submodule + editable pip install, not published to PyPI:

```powershell
git submodule add <this-repo-url> core
pip install -e ./core
```

Then import as `from kaggle_ai_core.xxx import yyy`.

## Status

Currently used by `kaggle-ai-model-agent`. **`kaggle-ai-notebook-agent` has NOT been migrated to this shared package yet** — it retains its own local copies of these modules (in `app/`), by deliberate choice: it was already working reliably in production when this package was extracted, and retrofitting a proven pipeline onto a not-yet-battle-tested abstraction was judged higher risk than the benefit of avoiding drift. Worth revisiting once this package has proven stable across a second real consumer.