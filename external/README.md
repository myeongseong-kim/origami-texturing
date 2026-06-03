# External Libraries

Place local third-party libraries, SDKs, source checkouts, or git submodules here
when they are needed for experiments but should stay separate from this package's
source code.

The `dinov3` directory is a git submodule pointing at
`https://github.com/facebookresearch/dinov3.git` and tracking the upstream `main`
branch.

Keep installable Python dependencies in `pyproject.toml` or `requirements.txt`
when possible, and use this directory for libraries that need to be referenced
from a local path.
