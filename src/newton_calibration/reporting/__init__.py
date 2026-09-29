"""Read-only, recipe-driven customer reports. Never authorizes robot activation."""

from .report import ReportError, build_report, load_recipe, render_report, write_bundle

__all__ = ["ReportError", "build_report", "load_recipe", "render_report", "write_bundle"]
