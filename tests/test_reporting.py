from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from newton_calibration.core.io import atomic_write_json
from newton_calibration.reporting import ReportError, build_report, load_recipe, render_report
from newton_calibration.reporting.adapters import export_toolkit_run
from newton_calibration.reporting.report import digest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "src/newton_calibration/recipes/arm_joint_response.v1.json"
spec = importlib.util.spec_from_file_location("report_demo", ROOT / "examples/reporting/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture
def bundle(tmp_path):
    return demo.make_demo(tmp_path / "bundle")


def mutate_record(bundle, update):
    manifest = json.loads(bundle.read_text())
    path = bundle.parent / manifest["records"]["synthetic"]["path"]
    data = json.loads(path.read_text())
    update(data)
    atomic_write_json(path, data)
    manifest["records"]["synthetic"]["sha256"] = digest(path.read_bytes())
    atomic_write_json(bundle, manifest)


def test_recipe_has_inputs_motions_and_report_contract():
    recipe = load_recipe(RECIPE)
    assert {x["id"] for x in recipe["setup_inputs"]} >= {"controller", "gripper_tool", "joint_mapping", "evidence"}
    assert "independent held-out multisine or different sweep" in recipe["collection"]["families"]
    assert "insertion" in recipe["execution"]["not_supported"]


def test_results_are_from_records_not_recipe(bundle, tmp_path):
    first = build_report(RECIPE, bundle)
    assert first["report_complete"]
    assert first["reduction_pct"] == 50
    assert first["values"]["activation_allowed"] is False
    mutate_record(bundle, lambda d: d.update(tuned_error=0.3))
    second = build_report(RECIPE, bundle)
    assert second["reduction_pct"] == pytest.approx(25)
    html = render_report(second, tmp_path / "report.html").read_text()
    assert "25.0%" in html and "Activation blocked" in html
    assert "SYNTHETIC EXAMPLE" in html and "0.3 deg" in html


def test_missing_facts_not_invented(bundle, tmp_path):
    mutate_record(bundle, lambda d: [d.pop(key) for key in ("tuned_error", "write.status", "activation_allowed")])
    model = build_report(RECIPE, bundle)
    assert not model["report_complete"] and model["reduction_pct"] is None
    assert len(model["issues"]) == 3
    html = render_report(model, tmp_path / "partial.html").read_text()
    assert "Report incomplete" in html and "Activation not recorded" in html
    assert "Comparison unavailable" in html


def test_worse_fit_is_not_called_improvement(bundle, tmp_path):
    mutate_record(bundle, lambda d: d.update(tuned_error=0.6))
    text = render_report(build_report(RECIPE, bundle), tmp_path / "worse.html").read_text()
    assert "higher error" in text and "did not improve" in text


def test_zero_baseline_is_not_a_percent_claim(bundle):
    mutate_record(bundle, lambda d: d.update(baseline_error=0))
    assert build_report(RECIPE, bundle)["reduction_pct"] is None


@pytest.mark.parametrize(
    "key,value",
    [("baseline_error", -1), ("tuned_error", True), ("validation_passed", "true"), ("fit.status", "success!")],
)
def test_invalid_semantics_rejected(bundle, key, value):
    mutate_record(bundle, lambda d: d.update({key: value}))
    with pytest.raises(ReportError):
        build_report(RECIPE, bundle)


def test_no_inline_results_in_manifest(bundle):
    data = json.loads(bundle.read_text())
    data["facts"]["tuned_error"] = {"value": 0.01}
    atomic_write_json(bundle, data)
    with pytest.raises(ReportError, match="not inline"):
        build_report(RECIPE, bundle)


def test_record_tampering_rejected(bundle):
    path = bundle.parent / "records/synthetic.json"
    path.write_text(path.read_text().replace('"tuned_error": 0.2', '"tuned_error": 0.1'))
    with pytest.raises(ReportError, match="Record changed"):
        build_report(RECIPE, bundle)


def test_recipe_edit_requires_new_binding(bundle, tmp_path):
    changed = tmp_path / "recipe.json"
    changed.write_text(RECIPE.read_text() + "\n")
    with pytest.raises(ReportError, match="Recipe ID/digest changed"):
        build_report(changed, bundle)


@pytest.mark.parametrize("relative", ["/etc/passwd", "../private.json"])
def test_path_escape_rejected(bundle, relative):
    data = json.loads(bundle.read_text())
    data["records"]["synthetic"]["path"] = relative
    atomic_write_json(bundle, data)
    with pytest.raises(ReportError):
        build_report(RECIPE, bundle)


def test_symlink_escape_rejected(bundle, tmp_path):
    outside = tmp_path / "private.json"
    outside.write_text("{}")
    (bundle.parent / "escape.json").symlink_to(outside)
    data = json.loads(bundle.read_text())
    data["records"]["synthetic"]["path"] = "escape.json"
    atomic_write_json(bundle, data)
    with pytest.raises(ReportError, match="escapes"):
        build_report(RECIPE, bundle)


def test_run_identity_rejected(bundle):
    data = json.loads(bundle.read_text())
    data["records"]["synthetic"]["run_id"] = "different-run"
    atomic_write_json(bundle, data)
    with pytest.raises(ReportError, match="identity"):
        build_report(RECIPE, bundle)


def test_markup_is_escaped(bundle, tmp_path):
    mutate_record(bundle, lambda d: d.update(inputs='<script>alert("x")</script>'))
    text = render_report(build_report(RECIPE, bundle), tmp_path / "safe.html").read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "Content-Security-Policy" in text
    assert "https://" not in text


def test_recipe_cannot_remove_required_status(tmp_path):
    data = json.loads(RECIPE.read_text())
    data["reporting"]["fields"] = [field for field in data["reporting"]["fields"] if field["id"] != "fit.status"]
    path = atomic_write_json(tmp_path / "recipe.json", data)
    with pytest.raises(ReportError, match="Missing core"):
        load_recipe(path)


def test_extension_requires_a_source(bundle, tmp_path):
    data = json.loads(RECIPE.read_text())
    data["reporting"]["fields"].append(
        {"id": "new_check", "label": "New recipe check", "type": "number", "stage": "validate", "required": True}
    )
    path = atomic_write_json(tmp_path / "extended.json", data)
    manifest = json.loads(bundle.read_text())
    manifest["recipe"]["sha256"] = digest(path.read_bytes())
    atomic_write_json(bundle, manifest)
    model = build_report(path, bundle)
    assert "Not recorded: New recipe check" in model["issues"]


@pytest.mark.parametrize("payload", ['{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_ambiguous_or_nonfinite_json_rejected(bundle, payload):
    manifest = json.loads(bundle.read_text())
    path = bundle.parent / manifest["records"]["synthetic"]["path"]
    path.write_text(payload)
    manifest["records"]["synthetic"]["sha256"] = digest(path.read_bytes())
    atomic_write_json(bundle, manifest)
    with pytest.raises(ReportError):
        build_report(RECIPE, bundle)


def test_preserves_existing_bundle(bundle):
    before = bundle.read_bytes()
    with pytest.raises(FileExistsError):
        demo.make_demo(bundle.parent)
    assert bundle.read_bytes() == before


def test_legacy_import_never_authorizes_activation(bundle, tmp_path):
    mutate_record(bundle, lambda d: d.update(activation_allowed=True))
    data = json.loads(bundle.read_text())
    data["execution_kind"] = "legacy_import"
    atomic_write_json(bundle, data)
    text = render_report(build_report(RECIPE, bundle), tmp_path / "legacy.html").read_text()
    assert "Activation not established" in text
    assert "Not five completed public API calls" in text


def test_partial_toolkit_records_not_status_surface(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    atomic_write_json(
        run / "analysis.json",
        {
            "run_id": "r1",
            "joints": ["j1"],
            "warnings": ["Mapping unknown"],
            "readiness": {"mapping_confirmed": False},
            "environment": {},
        },
    )
    atomic_write_json(run / "status.json", {"state": "COMPLETE", "activation_allowed": True})
    bundle = export_toolkit_run(run, RECIPE, tmp_path / "snapshot")
    model = build_report(RECIPE, bundle)
    assert model["values"]["analyze.status"] == "blocked"
    assert model["values"]["fit.status"] is None
    assert model["values"]["activation_allowed"] is None


def test_toolkit_adapter_rejects_cross_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    atomic_write_json(run / "analysis.json", {"run_id": "one"})
    atomic_write_json(run / "plan.json", {"run_id": "two"})
    with pytest.raises(ReportError, match="different runs"):
        export_toolkit_run(run, RECIPE, tmp_path / "snapshot")


def test_adapter_integrates_with_actual_five_calls(tmp_path):
    # This executes the existing analytic contract-test pipeline, not Newton and
    # not a real robot. It exercises the actual serialized dataclass structures.
    from test_five_calls import _analyze_synthetic_job

    from newton_calibration.isaaclab import tuning

    analysis = _analyze_synthetic_job(tmp_path)
    plan = tuning.plan(analysis)
    fit = tuning.fit(plan, generations=2, population=4, resume=False)
    validation = tuning.validate(fit)
    package = tuning.write(validation, output=tmp_path / "package")
    snapshot = export_toolkit_run(analysis.workdir, RECIPE, tmp_path / "snapshot", package_dir=package.output_dir)
    model = build_report(RECIPE, snapshot)
    assert model["report_complete"]
    assert model["values"]["baseline_error"] == validation.baseline_metrics["position_rmse_rad"]
    assert model["values"]["tuned_error"] == validation.calibrated_metrics["position_rmse_rad"]
    assert model["values"]["activation_allowed"] is False
    assert model["values"]["optimizer"]["name"] == "diagonal-cma-es"
    assert all(key in model["provenance"] for key in model["values"])


def test_imported_facts_have_provenance(bundle):
    model = build_report(RECIPE, bundle)
    assert model["provenance"]["baseline_error"]["pointer"] == "/baseline_error"
    assert len(model["provenance"]["baseline_error"]["sha256"]) == 64


def test_download_roundtrip_is_reproducible(bundle, tmp_path):
    model = build_report(RECIPE, bundle)
    original = render_report(model, tmp_path / "report.html").read_text()
    payload = re.search(r"data:application/zip;base64,([A-Za-z0-9+/=]+)", original).group(1)
    target = tmp_path / "download"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        archive.extractall(target)
    reopened = build_report(target / "recipe.json", target / "bundle.json")
    assert reopened["values"] == model["values"]
    assert reopened["recipe"]["_sha256"] == model["recipe"]["_sha256"]


def test_strict_cli_writes_gaps_and_exits_two(bundle, tmp_path):
    mutate_record(bundle, lambda d: d.pop("tuned_error"))
    output = tmp_path / "partial.html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "newton_calibration.reporting",
            "--recipe",
            str(RECIPE),
            "--bundle",
            str(bundle),
            "--output",
            str(output),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "Report incomplete" in output.read_text()


def test_archive_path_collision_rejected(bundle):
    manifest = json.loads(bundle.read_text())
    manifest["records"]["duplicate"] = dict(manifest["records"]["synthetic"])
    atomic_write_json(bundle, manifest)
    with pytest.raises(ReportError, match="archive path"):
        build_report(RECIPE, bundle)


def test_svg_media_is_rejected(bundle):
    manifest = json.loads(bundle.read_text())
    manifest["media"] = [{"kind": "simulation_preview", "mime": "image/svg+xml"}]
    atomic_write_json(bundle, manifest)
    with pytest.raises(ReportError, match="raster"):
        build_report(RECIPE, bundle)
