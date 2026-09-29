from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

from newton_calibration.core.io import atomic_write_json, atomic_write_text

STAGES = ("analyze", "plan", "fit", "validate", "write")
STATES = {"completed", "recorded", "blocked", "failed", "not_run", "unknown"}
KINDS = {"public_api", "legacy_import", "synthetic"}
TYPES = {"text", "number", "boolean", "object", "array", "status"}
CORE = {
    "inputs",
    "discoveries",
    "limitations",
    "next_action",
    "execution_note",
    "scope",
    "baseline_error",
    "tuned_error",
    "metric_label",
    "metric_unit",
    "validation_passed",
    "activation_allowed",
    "parameters",
    "objective",
    "optimizer",
    "collection",
    "runtime",
    "gates",
} | {f"{stage}.{suffix}" for stage in STAGES for suffix in ("status", "summary")}
_ID = re.compile(r"^[a-zA-Z0-9_.-]+$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class ReportError(ValueError):
    """A report input is malformed or no longer matches its recorded digest."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(raw: bytes) -> Any:
    def reject_constant(value):
        raise ReportError(f"Non-finite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReportError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)  # also rejects exponent overflow such as 1e999
        return value
    except (ValueError, UnicodeError) as exc:
        raise ReportError(f"Invalid finite JSON: {exc}") from exc


def load_recipe(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    recipe = _json(raw)
    if not isinstance(recipe, dict) or recipe.get("schema") != "newton.calibration.recipe/v1":
        raise ReportError("Unsupported recipe schema")
    for key in ("id", "title", "purpose"):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            raise ReportError(f"Recipe requires {key}")
    reporting = recipe.get("reporting", {})
    if reporting.get("schema") != "newton.calibration.reporting/v1":
        raise ReportError("Unsupported reporting contract")
    fields = reporting.get("fields")
    if not isinstance(fields, list):
        raise ReportError("Recipe reporting.fields must be an array")
    ids = set()
    for field in fields:
        if not isinstance(field, dict) or set(field) != {"id", "label", "stage", "type", "required"}:
            raise ReportError("A reporting field requires id, label, stage, type, required")
        if not isinstance(field["id"], str) or not _ID.fullmatch(field["id"]) or field["id"] in ids:
            raise ReportError("Invalid or duplicate reporting field ID")
        ids.add(field["id"])
        if field["stage"] not in (*STAGES, "overview") or field["type"] not in TYPES:
            raise ReportError("Invalid field stage or type")
        if not isinstance(field["required"], bool) or not isinstance(field["label"], str):
            raise ReportError("Invalid required flag or label")
    if CORE - ids:
        raise ReportError(f"Missing core reporting requirements: {sorted(CORE - ids)}")
    expected_types = {
        key: "text"
        for key in (
            "inputs",
            "discoveries",
            "limitations",
            "next_action",
            "execution_note",
            "scope",
            "metric_label",
            "metric_unit",
        )
    }
    expected_types.update(
        {
            "baseline_error": "number",
            "tuned_error": "number",
            "validation_passed": "boolean",
            "activation_allowed": "boolean",
        }
    )
    for stage in STAGES:
        expected_types[f"{stage}.status"] = "status"
        expected_types[f"{stage}.summary"] = "text"
    for field in fields:
        if field["id"] in CORE and field["required"] is not True:
            raise ReportError("Core reporting requirements cannot be made optional")
        if field["id"] in expected_types and field["type"] != expected_types[field["id"]]:
            raise ReportError(f"Incorrect type for {field['id']}")
    return {**recipe, "_sha256": digest(raw), "_source": raw.decode("utf-8")}


def _local(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ReportError("Record paths must be relative to the bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ReportError(f"Missing record or path escapes bundle: {relative}")
    return path


def _pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ReportError("Use a JSON Pointer, e.g. /heldout/baseline_rmse")
    if not pointer:
        return value
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ReportError("Invalid JSON Pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[token]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            value = value[int(token)]
        else:
            raise KeyError(token)
    return value


def _valid(value: Any, kind: str) -> bool:
    return {
        "text": lambda: isinstance(value, str) and bool(value.strip()),
        "number": lambda: type(value) in (float, int) and math.isfinite(value),
        "boolean": lambda: type(value) is bool,
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "status": lambda: isinstance(value, str) and value in STATES,
    }[kind]()


def build_report(recipe_path: str | Path, bundle_path: str | Path) -> dict:
    """Resolve declared facts from hash-checked records; missing facts stay missing.

    Digests establish bundle consistency, not producer authenticity or scientific
    validity. Only the existing package loader can decide whether to activate.
    """
    recipe = load_recipe(recipe_path)
    bundle_path = Path(bundle_path)
    raw = bundle_path.read_bytes()
    bundle = _json(raw)
    if not isinstance(bundle, dict) or bundle.get("schema") != "newton.calibration.report-bundle/v1":
        raise ReportError("Unsupported report bundle schema")
    if bundle.get("recipe") != {"id": recipe["id"], "sha256": recipe["_sha256"]}:
        raise ReportError("Recipe ID/digest changed; create a new report revision")
    if (
        bundle.get("execution_kind") not in KINDS
        or not isinstance(bundle.get("run_id"), str)
        or not bundle["run_id"].strip()
    ):
        raise ReportError("Bundle needs a run ID and explicit execution_kind")
    if not isinstance(bundle.get("facts"), dict) or not isinstance(bundle.get("records"), dict):
        raise ReportError("Bundle requires record and fact maps")
    values, provenance, issues, records = {}, {}, [], {}
    root = bundle_path.parent
    archive_paths = {"recipe.json", "bundle.json", "report-provenance.json"}
    for key, ref in bundle["records"].items():
        if not _ID.fullmatch(key) or not isinstance(ref, dict):
            raise ReportError("Invalid record ID/reference")
        expected = ref.get("sha256", "")
        if not isinstance(expected, str) or not _DIGEST.fullmatch(expected):
            raise ReportError(f"Record {key} needs a SHA-256 digest")
        raw_record = _local(root, ref.get("path")).read_bytes()
        archive_path = Path(ref["path"]).as_posix()
        if archive_path in archive_paths:
            raise ReportError("Duplicate or reserved archive path")
        archive_paths.add(archive_path)
        if digest(raw_record) != expected:
            raise ReportError(f"Record changed: {key}")
        obj = _json(raw_record)
        if "run_id" in ref and (not isinstance(obj, dict) or obj.get("run_id") != ref["run_id"]):
            raise ReportError(f"Mixed run identity for {key}")
        records[key] = {"value": obj, "sha256": expected, "text": raw_record.decode("utf-8"), "path": ref["path"]}
    declared = {field["id"] for field in recipe["reporting"]["fields"]}
    if set(bundle["facts"]) - declared:
        raise ReportError("Bundle contains facts not declared by its reporting contract")
    for field in recipe["reporting"]["fields"]:
        key = field["id"]
        ref = bundle["facts"].get(key)
        value = None
        if ref is not None:
            if not isinstance(ref, dict) or set(ref) != {"record", "pointer"}:
                raise ReportError(f"{key}: facts must reference recorded data, not inline results")
            if not isinstance(ref["record"], str) or ref["record"] not in records:
                raise ReportError(f"{key}: unknown source record")
            try:
                value = _pointer(records[ref["record"]]["value"], ref["pointer"])
            except (KeyError, IndexError):
                value = None
            provenance[key] = {**ref, "sha256": records[ref["record"]]["sha256"]}
        if value is not None and not _valid(value, field["type"]):
            raise ReportError(f"{key}: expected {field['type']}")
        if value is None and field["required"]:
            issues.append(f"Not recorded: {field['label']}")
        values[key] = value
    before, after = values["baseline_error"], values["tuned_error"]
    if any(v is not None and v < 0 for v in (before, after)):
        raise ReportError("Error metrics must be non-negative")
    reduction = 100 * (1 - after / before) if before is not None and before > 0 and after is not None else None
    if reduction is not None and not math.isfinite(reduction):
        raise ReportError("Metric reduction is not finite")
    media = []
    if not isinstance(bundle.get("media", []), list):
        raise ReportError("Bundle media must be an array")
    for entry in bundle.get("media", []):
        if not isinstance(entry, dict) or entry.get("kind") not in {"measured_simulation_trace", "simulation_preview"}:
            raise ReportError("Media requires an explicit trace/preview classification")
        mime = entry.get("mime")
        if mime not in {"image/png", "image/gif", "image/jpeg"}:
            raise ReportError("Only local raster PNG/GIF/JPEG media is supported")
        payload = _local(root, entry.get("path")).read_bytes()
        archive_path = Path(entry["path"]).as_posix()
        if archive_path in archive_paths:
            raise ReportError("Duplicate or reserved archive path")
        archive_paths.add(archive_path)
        if digest(payload) != entry.get("sha256"):
            raise ReportError("Media changed")
        signatures = {
            "image/png": (b"\x89PNG\r\n\x1a\n",),
            "image/gif": (b"GIF87a", b"GIF89a"),
            "image/jpeg": (b"\xff\xd8\xff",),
        }
        if not payload.startswith(signatures[mime]) or not isinstance(entry.get("caption"), str):
            raise ReportError("Invalid raster media or caption")
        media.append({**entry, "uri": f"data:{mime};base64," + base64.b64encode(payload).decode()})
    return {
        "schema": "newton.calibration.customer-report/v1",
        "run_id": bundle["run_id"],
        "execution_kind": bundle["execution_kind"],
        "recipe": recipe,
        "bundle_sha256": digest(raw),
        "bundle_source": raw.decode("utf-8"),
        "values": values,
        "provenance": provenance,
        "issues": issues,
        "report_complete": not issues,
        "reduction_pct": reduction,
        "records": records,
        "media": media,
    }


def write_bundle(
    recipe_path: str | Path,
    destination: str | Path,
    *,
    run_id: str,
    execution_kind: str,
    records: dict[str, Any],
    facts: dict[str, tuple[str, str]],
) -> Path:
    """Snapshot producer records in a NEW directory. Facts contain source pointers.

    This does not execute any of the five calls, create confirmations, or validate
    a package. Existing directories are rejected so previous results survive.
    """
    recipe = load_recipe(recipe_path)
    if execution_kind not in KINDS:
        raise ReportError("Unsupported execution kind")
    if any(not _ID.fullmatch(key) for key in records):
        raise ReportError("Invalid record ID")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    refs = {}
    for key, value in records.items():
        path = atomic_write_json(target / "records" / f"{key}.json", value)
        refs[key] = {"path": f"records/{key}.json", "sha256": digest(path.read_bytes())}
        if isinstance(value, dict) and isinstance(value.get("run_id"), str):
            refs[key]["run_id"] = value["run_id"]
    path = atomic_write_json(
        target / "bundle.json",
        {
            "schema": "newton.calibration.report-bundle/v1",
            "run_id": run_id,
            "execution_kind": execution_kind,
            "recipe": {"id": recipe["id"], "sha256": recipe["_sha256"]},
            "records": refs,
            "facts": {key: {"record": value[0], "pointer": value[1]} for key, value in facts.items()},
        },
    )
    build_report(recipe_path, path)
    return path


def render_report(model: dict, output: str | Path) -> Path:
    """Customer-first offline HTML. Raw records are behind a disclosure/download."""
    esc = lambda value: html.escape(str(value), quote=True)
    values = model["values"]

    def text(key):
        value = values.get(key)
        return esc("Not recorded" if value is None else value)

    def display(value):
        if value is None:
            return "Not recorded"
        return esc(json.dumps(value, indent=2, ensure_ascii=False) if isinstance(value, (dict, list)) else value)

    reduction = model["reduction_pct"]
    amount = "—" if reduction is None else f"{abs(reduction):.1f}%"
    direction = "Comparison unavailable" if reduction is None else ("lower error" if reduction >= 0 else "higher error")
    outcome = (
        "Measured comparison"
        if reduction is None
        else ("Closer to the measured motion." if reduction > 0 else "The motion gap did not improve.")
    )
    provenance_note = {
        "synthetic": "SYNTHETIC EXAMPLE · Not a robot result",
        "legacy_import": "EXPERIMENTAL IMPORT · Not five completed public API calls",
        "public_api": "TOOLKIT RUN RECORDS · Reporting is not package verification",
    }[model["execution_kind"]]
    activation = values.get("activation_allowed")
    package_status = (
        "Activation blocked"
        if activation is False
        else ("Activation not recorded" if activation is None else "Activation reported · verify with package loader")
    )
    if model["execution_kind"] != "public_api" and activation is True:
        package_status = "Activation not established by this report"
    validation = values.get("validation_passed")
    validation_status = "Not recorded" if validation is None else ("Passed (recorded)" if validation else "Not passed")
    cards = []
    for i, stage in enumerate(STAGES, 1):
        rows = []
        for field in model["recipe"]["reporting"]["fields"]:
            key = field["id"]
            if field["stage"] != stage or key in {f"{stage}.status", f"{stage}.summary"}:
                continue
            ref = model["provenance"].get(key)
            source = f"{ref['record']}#{ref['pointer']} · {ref['sha256'][:12]}" if ref else "No source record"
            rows.append(
                f'<div class="fact"><h4>{esc(field["label"])}</h4><pre>{display(values[key])}</pre>'
                f"<small>{esc(source)}</small></div>"
            )
        state = values[f"{stage}.status"] or "unknown"
        cards.append(
            f'<article class="stage"><span class="step">0{i}</span>'
            f'<span class="status">{esc(state.replace("_", " "))}</span>'
            f"<h3>{stage.title()}</h3><p>{text(f'{stage}.summary')}</p>"
            f"<details><summary>Evidence + details ↗</summary>{''.join(rows)}</details></article>"
        )
    bars = []
    scale = max(values.get("baseline_error") or 0, values.get("tuned_error") or 0)
    for label, key, color in (("Before tuning", "baseline_error", "before"), ("After tuning", "tuned_error", "after")):
        val = values[key]
        width = 0 if val is None or scale <= 0 else val / scale * 100
        formatted = "Not recorded" if val is None else f"{val:.4g} {values.get('metric_unit') or ''}"
        bars.append(
            f'<div class="bar-label"><span>{label}</span><b>{esc(formatted)}</b></div>'
            f'<div class="track"><i class="{color}" style="width:{width:.2f}%"></i></div>'
        )
    media = "".join(
        f'<figure><img src="{entry["uri"]}" alt="{esc(entry["caption"])}" loading="lazy">'
        f"<figcaption>{esc(entry['caption'])} · {esc(entry['kind'].replace('_', ' '))}. "
        "Not robot-camera footage.</figcaption></figure>"
        for entry in model["media"]
    )
    issue_html = ""
    if model["issues"]:
        issue_html = (
            '<aside class="warning"><b>Report incomplete</b><ul>'
            + "".join(f"<li>{esc(issue)}</li>" for issue in model["issues"])
            + "</ul></aside>"
        )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for record in model["records"].values():
            zipped.writestr(record["path"], record["text"])
        zipped.writestr("recipe.json", model["recipe"]["_source"])
        zipped.writestr("bundle.json", model["bundle_source"])
        for entry in model["media"]:
            zipped.writestr(entry["path"], base64.b64decode(entry["uri"].split(",", 1)[1]))
        zipped.writestr(
            "report-provenance.json",
            json.dumps(
                {
                    "recipe_sha256": model["recipe"]["_sha256"],
                    "bundle_sha256": model["bundle_sha256"],
                    "facts": model["provenance"],
                    "issues": model["issues"],
                },
                indent=2,
            ),
        )
    download = "data:application/zip;base64," + base64.b64encode(archive.getvalue()).decode()
    style = Path(__file__).with_name("customer.css").read_text()
    source = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>{esc(model["recipe"]["title"])} · Calibration report</title><style>{style}</style></head><body>
<header><a href="#overview">● NEWTON CALIBRATION <span>/ Customer report</span></a>
<a class="button" href="{download}" download="run-records.zip">Download records ↓</a></header><main>
<section id="overview"><p class="eyebrow">{provenance_note}</p><div class="heading">
<h1>{esc(model["recipe"]["title"])}</h1><span class="badge">{package_status}</span></div>
<p class="scope">{text("scope")}</p><div class="outcome"><div><p class="eyebrow">THE OUTCOME</p>
<h2>{outcome}</h2><p>{text("limitations")}</p><p class="boundary">Real-task transfer is not established by this report.</p></div>
<div class="score"><strong>{amount}</strong><p>{direction} · {text("metric_label")}</p>{"".join(bars)}</div></div>
<div class="context"><article><p class="eyebrow">01 / PROVIDED</p><p>{text("inputs")}</p></article>
<article><p class="eyebrow">02 / LEARNED FROM THE RECORDS</p><p>{text("discoveries")}</p></article>
<article><p class="eyebrow">03 / HELD-OUT CHECK</p><p>{validation_status}</p><small>Motion validation ≠ real-task transfer</small></article></div></section>
{issue_html}<section><p class="eyebrow">FIVE STAGES · WHAT WAS RECORDED</p><h2>The run, at a glance.</h2>
<div class="stages">{"".join(cards)}</div><p class="execution">{text("execution_note")}</p></section>
{('<section><p class="eyebrow">SEE THE EVIDENCE</p><h2>Measured vs. simulated.</h2><div class="media-grid">' + media + "</div></section>") if media else ""}
<section class="next"><p class="eyebrow">WHAT SHOULD HAPPEN NEXT?</p><h2>{text("next_action")}</h2></section>
<details class="audit"><summary>Traceability and reporting contract</summary>
<p>Run: {esc(model["run_id"])} · Recipe: {esc(model["recipe"]["id"])}</p>
<p>Recipe SHA-256: <code>{esc(model["recipe"]["_sha256"])}</code></p>
<p>Bundle SHA-256: <code>{esc(model["bundle_sha256"])}</code></p>
<p>Required facts: {"complete" if model["report_complete"] else "incomplete"}. Completeness is not a scientific pass.</p>
<p>Digests detect changes relative to this bundle; they do not authenticate the producer.
The report never activates a package or authorizes hardware motion.</p></details>
<footer>Recipe defines the questions. Run records supply the answers. Customer report explains the outcome.</footer>
</main></body></html>'''
    return atomic_write_text(output, source)
