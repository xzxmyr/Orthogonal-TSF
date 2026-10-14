"""Verify the compact R6 release without fitting or reading points.npz.

Run from any directory: python experiments/muse_formula_r6/verify_release.py
Use --root to verify a different directory containing model.py, run.py, and
artifacts/. All output goes to stdout; no source, model, or artifact is written.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Path escapes release directory: " + relative)
    return path


def verify_manifest(root):
    path = root / "SHA256SUMS"
    if not path.exists():
        return {"status": "not_present", "files_checked": 0}
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, "Invalid SHA256SUMS entry")
        expected, relative = match.groups()
        require(relative not in seen, "Duplicate SHA256SUMS entry: " + relative)
        seen.add(relative)
        require(sha256(inside(root, relative)) == expected.lower(),
                "SHA256SUMS mismatch: " + relative)
    require(bool(seen), "SHA256SUMS is empty")
    return {"status": "passed", "files_checked": len(seen)}


def verify_frozen_sources(root, config):
    names = ("model.py", "run.py", "PROTOCOL_FROZEN.md", "FORMULA_REVIEW.md")
    require(set(config["sha"]) == set(names), "Unexpected frozen source list")
    unchanged = 0
    redactions = 0
    for name in names:
        actual = sha256(root / name)
        original = config["sha"][name]
        if actual == original:
            unchanged += 1
            continue
        require(name == "PROTOCOL_FROZEN.md", "Frozen source SHA mismatch: " + name)
        provenance_path = root / "PUBLICATION_PROVENANCE.json"
        require(provenance_path.is_file(),
                "Protocol redaction has no publication provenance")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        entry = provenance.get("source_files", {}).get(name, {})
        require(entry.get("original_sha256") == original and
                entry.get("published_sha256") == actual and
                entry.get("change") == "remove private coordination metadata",
                "Protocol redaction provenance does not match frozen and published hashes")
        redactions += 1
    return {"frozen_sources_unchanged": unchanged, "publication_redactions": redactions}


def model_digest(model, np):
    digest = hashlib.sha256()
    arrays = [model.structure.weight, model.coef, model.h_weight,
              np.asarray(model.mu)]
    for group in ("T", "M", "J"):
        for field in ("scale", "rotation"):
            arrays.append(model.structure.states[group][field])
    for array in arrays:
        array = np.asarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    metadata = {"params": model.params, "certificate": model.structure.certificate,
                "anchor": list(model.anchor), "underfit": model.underfit,
                "estimator": model.estimator, "history": model.history,
                "start": model.start}
    digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    artifacts = root / "artifacts"
    manifest = verify_manifest(root)
    config = json.loads((artifacts / "config.json").read_text(encoding="utf-8"))
    source_verification = verify_frozen_sources(root, config)

    sys.path.insert(0, str(root))
    import numpy as np
    import model
    import run

    require(Path(model.__file__).resolve() == root / "model.py", "Wrong model import")
    require(Path(run.__file__).resolve() == root / "run.py", "Wrong run import")

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Release verification attempted fitting")

    # Guard every training entry point used by the original round.
    model.JointStructural.fit = forbidden
    model.StateHistory.fit_history = forbidden
    for module in (model, run):
        module.estimate_covariance = forbidden
        module.likelihood_covariance = forbidden
        module.feasible_gls = forbidden
    run.main = forbidden

    rows = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    variants = {"_".join(parts) for parts in itertools.product(
        config["estimators"], config["specs"], config["history"])}
    expected = {}
    for seed, design, period, interaction, ar in itertools.product(
            config["seeds"], config["designs"], config["period"],
            config["interaction"], config["ar"]):
        ident = f"s{seed}_{design}_p{period}_j{interaction}_ar{ar}"
        expected[ident] = (seed, design, period, interaction, ar)
    require(len(expected) == 72 and len(variants) == 36,
            "Expected the frozen 72-configuration, 36-variant R6 release")
    require(len(rows) == 2592, "Expected 2592 saved metric rows")
    grouped = {ident: {} for ident in expected}
    for row in rows:
        ident, variant = row["id"], row["variant"]
        require(ident in grouped and variant in variants, "Unexpected metric model")
        require(variant not in grouped[ident], "Duplicate metric model")
        require((row["seed"], row["design"], row["period"],
                 row["interaction"], row["ar"]) == expected[ident],
                "Metric configuration mismatch: " + ident)
        require(variant == "_".join((row["estimator"], row["spec"], row["history"])),
                "Metric variant mismatch: " + ident + "/" + variant)
        grouped[ident][variant] = row
    require(all(set(group) == variants for group in grouped.values()),
            "Missing metric variants")
    expected_npz = {ident + "/" + variant + ".npz"
                    for ident in expected for variant in variants}
    actual_npz = {str(path.relative_to(artifacts)) for path in artifacts.glob("*/*.npz")
                  if path.name not in ("points.npz", "frozen_base.npz")}
    require(actual_npz == expected_npz, "Checkpoint inventory does not match metrics")

    require(config["split"] == {"base": [1, 1200], "structure": [1201, 2400],
                               "history": [2401, 3600], "holdout": [3601, 4800]},
            "Unexpected frozen R6 split")
    holdout = np.arange(3601, 4801)
    all_idx = np.arange(1, 4801)
    maxima = {"prediction_mse_abs_error": 0.0, "source_mse_abs_error": 0.0,
              "structural_mse_abs_error": 0.0, "closure_abs_error": 0.0,
              "future_isolation_abs_error": 0.0, "singleton_abs_error": 0.0}
    models_checked = 0

    def close(actual, saved, field, context):
        error = abs(float(actual) - float(saved))
        maxima[field] = max(maxima[field], error)
        require(np.isclose(actual, saved, rtol=1e-10, atol=1e-8),
                context + ": metric mismatch " + field)

    def closure(error, context):
        value = float(np.max(np.abs(error)))
        maxima["closure_abs_error"] = max(maxima["closure_abs_error"], value)
        require(value <= 1e-8, context + ": ledger closure failed")

    for ident in sorted(expected):
        seed, design, period, interaction, ar = expected[ident]
        t, y, x, sources = run.dgp(seed, period, interaction, design, ar)
        base_path = artifacts / ident / "frozen_base.npz"
        base_sha = sha256(base_path)
        with np.load(base_path, allow_pickle=False) as stored_base:
            coef = stored_base["coef"].copy()
        f0 = coef[0] + coef[1] * t[holdout] / 3600 + coef[2] * y[holdout - 1]
        residual = y[holdout] - f0
        canonical = run.ledger(sources, coef, all_idx, t, x, interaction)
        full_f0 = coef[0] + coef[1] * t[all_idx] / 3600 + coef[2] * y[all_idx - 1]
        closure(y[all_idx] - full_f0 - sum(canonical.values()), ident)
        # A rolling prediction may use previous holdout labels. Test one current
        # point per model, changing only that point's label and all future data.
        probe_position = 37
        probe_idx = holdout[probe_position:probe_position + 1]
        changed_y = y.copy()
        changed_x = x.copy()
        changed_y[int(probe_idx[0]):] += 17000
        changed_x[int(probe_idx[0]) + 1:] += 7000

        for variant in sorted(variants):
            context = ident + "/" + variant
            row = grouped[ident][variant]
            path = artifacts / ident / (variant + ".npz")
            checkpoint_sha = sha256(path)
            loaded = model.StateHistory.load(path)
            stamp = model_digest(loaded, np)
            require(np.array_equal(loaded.coef, coef), context + ": base mismatch")
            require((loaded.estimator, loaded.history) ==
                    (row["estimator"], row["history"]), context + ": metadata mismatch")
            opts = config["specs"][row["spec"]]
            require(loaded.underfit == opts.get("underfit", False) and
                    tuple(loaded.anchor) == tuple(opts.get("anchor", (0.0, 0.0))),
                    context + ": specification mismatch")
            output = loaded.predict(t, y, x, holdout)
            require(all(np.all(np.isfinite(value)) for value in output.values()),
                    context + ": nonfinite prediction")
            close(run.score(output["prediction_total"], residual)["mse"],
                  row["prediction_mse"], "prediction_mse_abs_error", context)
            close(run.score(output["structural_total"], residual)["mse"],
                  row["structural_mse"], "structural_mse_abs_error", context)
            truth = run.ledger(sources, coef, holdout, t, x, interaction, loaded.anchor)
            for group in ("bias", "T", "M", "J"):
                close(run.score(output[group], truth[group])["mse"],
                      row["sources"][group]["mse"], "source_mse_abs_error", context)
            closure(output["structural_total"] - sum(output[group]
                    for group in ("bias", "T", "M", "J")), context)
            closure(output["prediction_total"] - output["structural_total"] - output["H"],
                    context)
            closure(output["H"] - output["H_state"] - output["H_remainder"] -
                    output["H_calibration"], context)
            singleton = loaded.predict(t, y, x, probe_idx)
            changed = loaded.predict(t, changed_y, changed_x, probe_idx)
            for key in output:
                error = float(np.max(np.abs(singleton[key] - output[key][probe_position])))
                maxima["singleton_abs_error"] = max(maxima["singleton_abs_error"], error)
                require(error <= 1e-8, context + ": singleton mismatch " + key)
                error = float(np.max(np.abs(singleton[key] - changed[key])))
                maxima["future_isolation_abs_error"] = max(
                    maxima["future_isolation_abs_error"], error)
                require(error <= 1e-8, context + ": future data affected " + key)
            require(model_digest(loaded, np) == stamp, context + ": parameters changed")
            require(sha256(path) == checkpoint_sha, context + ": checkpoint changed")
            models_checked += 1
        require(sha256(base_path) == base_sha, ident + ": base checkpoint changed")

    print(json.dumps({"passed": True, "configurations_checked": len(expected),
                      "models_checked": models_checked, "no_fit_calls": True,
                      "points_npz_required": False, **source_verification,
                      "manifest": manifest, "metric_atol": 1e-8, "metric_rtol": 1e-10,
                      "future_isolation_points_per_model": 1, "max_errors": maxima},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        sys.exit(1)
