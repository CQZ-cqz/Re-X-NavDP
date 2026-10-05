# Vendored Ultralytics

- Upstream: <https://github.com/ultralytics/ultralytics>
- Revision: `82737b9e3aa61aaf104d61a055db4c773a4e7e8d`
- License: AGPL-3.0; see `LICENSE` in this directory.

Only the PyTorch YOLO26-Depth feature path is used by X-NavDP. The source is
vendored so model graph parsing and the semantic P3/stride-8 early exit remain
pinned; pretrained `.pt` weights are intentionally not stored in this repo.
