from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from conformal_trees.data import load_dataset, make_splits
from conformal_trees.models import GBDTClassifier
from conformal_trees.recourse import (
    AllOfGate,
    OODGate,
    RecourseSpace,
    SearchConfig,
    find_recourse,
    make_donors,
)
from conformal_trees.signals import IsolationForestScore, KNNScore, LOFScore
from conformal_trees.utils import derive_seed, load_config

GateName = Literal["none", "gate_IF", "gate_LOF", "gate_ALL"]


class ServiceState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.dataset_name: str = "adult"
        self.seed: int = 0
        self.columns: list[str] = []
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.mutable_cols: list[str] = []
        self.immutable_cols: list[str] = []
        self.spiky_cols: list[str] = []
        self.monotone_up_cols: list[str] = []
        self.model: GBDTClassifier | None = None
        self.space: RecourseSpace | None = None
        self.donors: Any = None
        self.detectors: dict[str, Any] = {}
        self.gates: dict[str, Any] = {}
        self.test_X: pd.DataFrame | None = None
        self.test_y: pd.Series | None = None

    def build(self, dataset: str = "adult", seed: int = 0) -> None:
        base_cfg = load_config("base.yaml")
        model_cfg = base_cfg.get("model", base_cfg.get("gbdt", {}))
        rec_cfg = base_cfg

        ds = load_dataset(dataset)
        splits = make_splits(ds.X, ds.y, base_cfg["splits"], seed=derive_seed(seed, "split"))

        self.dataset_name = dataset
        self.seed = seed
        self.columns = list(ds.X.columns)
        self.numeric_cols = list(ds.numeric)
        self.categorical_cols = list(ds.categorical)
        self.test_X = splits.test.X.copy()
        self.test_y = splits.test.y.copy()

        self.model = GBDTClassifier.from_config(model_cfg, seed=derive_seed(seed, "gbdt")).fit(
            splits.train.X, splits.train.y
        )

        ds_cfg = load_config(f"datasets/{dataset}.yaml")
        rec_section = ds_cfg.get("recourse", {})
        extra_imm = rec_section.get("extra_immutable", [])
        mono_up = rec_section.get("monotone_increasing", [])

        rec_options = rec_cfg.get("recourse", {})
        self.space = RecourseSpace.from_dataset(
            ds,
            splits.train.X,
            extra_immutable=extra_imm,
            monotone_up=mono_up,
            spike_immutable=rec_options.get("spike_immutable", True),
            max_mode_share=rec_options.get("max_mode_share", 0.75),
        )
        self.mutable_cols = list(self.space.mutable)
        self.immutable_cols = [c for c in self.columns if c not in self.mutable_cols]
        self.spiky_cols = list(self.space.spiky)
        self.monotone_up_cols = list(self.space.monotone_up)

        self.donors = make_donors(self.space, splits.train.X, splits.train.y)

        det_specs = {
            "IF": IsolationForestScore(ds.numeric, ds.categorical, seed=derive_seed(seed, "det", "IF")),
            "LOF": LOFScore(ds.numeric, ds.categorical),
            "kNN": KNNScore(ds.numeric, ds.categorical),
        }
        gate_cfg = rec_cfg.get("gate", {})
        alpha = gate_cfg.get("alpha", 0.05)
        delta = gate_cfg.get("delta", 0.05)

        single_gates: dict[str, OODGate] = {}
        self.detectors = {}
        for name, det in det_specs.items():
            det.fit(splits.train.X)
            self.detectors[name] = det
            cal_scores = det.score(splits.calib.X)
            single_gates[name] = OODGate(det.score, cal_scores, alpha=alpha, delta=delta)

        self.gates = {
            "none": None,
            "gate_IF": single_gates["IF"],
            "gate_LOF": single_gates["LOF"],
            "gate_ALL": AllOfGate(list(single_gates.values())),
        }
        self.ready = True


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.build(dataset="adult", seed=0)
    yield


app = FastAPI(
    title="Conformal Tree OOD & Recourse API",
    version="0.4.0",
    description="Conformalized tree-based ensembles for OOD detection and constrained counterfactual recourse.",
    lifespan=lifespan,
)


class OODCheckRequest(BaseModel):
    record: dict[str, Any]
    gate: GateName = "gate_ALL"


class RecourseRequest(BaseModel):
    record: dict[str, Any]
    gate: GateName = "gate_ALL"
    target_prob: float = Field(0.6, gt=0.0, lt=1.0)
    generations: int = Field(40, ge=1, le=200)
    pop_size: int = Field(150, ge=20, le=500)
    seed: int = 0


def _clean_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _to_frame(record: dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in state.columns if c not in record]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    row = {}
    for col in state.columns:
        val = record[col]
        if col in state.categorical_cols:
            cats = pd.api.types.CategoricalDtype(
                categories=state.space.categories[col] if state.space and col in state.space.categories else None
            )
            row[col] = pd.Series([val], dtype=cats) if cats.categories is not None else [str(val)]
        else:
            row[col] = [np.nan if val is None or val == "" else float(val)]

    df = pd.DataFrame(row)[state.columns]
    for col in state.categorical_cols:
        if state.space and col in state.space.categories:
            df[col] = pd.Categorical(df[col], categories=state.space.categories[col])
    return df


def _detector_breakdown(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, det in state.detectors.items():
        score_val = float(det.score(df)[0])
        sub_gate = state.gates.get(f"gate_{name}")
        thresh = float(sub_gate.score_threshold) if (sub_gate is not None and sub_gate.score_threshold is not None) else None
        is_ood = bool(thresh is not None and score_val > thresh)
        rows.append(
            {
                "detector": name,
                "score": score_val,
                "conformal_threshold": thresh,
                "ood_flag": is_ood,
            }
        )
    return rows


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.ready else "initializing",
        "dataset": state.dataset_name,
        "seed": state.seed,
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if not state.ready or state.space is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    gate_thresholds: dict[str, Any] = {}
    for g_name, g_obj in state.gates.items():
        if g_obj is None:
            gate_thresholds[g_name] = None
        elif isinstance(g_obj, AllOfGate):
            gate_thresholds[g_name] = {
                sub_name: state.gates[f"gate_{sub_name}"].score_threshold
                for sub_name in state.detectors
            }
        else:
            gate_thresholds[g_name] = g_obj.score_threshold

    return {
        "dataset": state.dataset_name,
        "columns": state.columns,
        "mutable_features": state.mutable_cols,
        "immutable_or_non_actionable_features": state.immutable_cols,
        "spiky_features": state.spiky_cols,
        "monotone_increasing_features": state.monotone_up_cols,
        "gate_thresholds": gate_thresholds,
    }


@app.get("/sample-record")
def sample_record(
    offset: int = Query(0, ge=0, le=200),
    min_prob: float = Query(0.15, ge=0.0, le=0.55),
) -> dict[str, Any]:
    if not state.ready or state.model is None or state.test_X is None or state.test_y is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    unfav_mask = state.test_y == 0
    candidates = state.test_X[unfav_mask]
    if candidates.empty:
        raise HTTPException(status_code=404, detail="No unfavourable test records found.")

    probs = state.model.predict_proba(candidates)[:, 1]
    eligible_idx = np.where(probs >= min_prob)[0]
    pool = eligible_idx if len(eligible_idx) > 0 else np.arange(len(candidates))
    chosen_pos = int(pool[offset % len(pool)])
    row = candidates.iloc[chosen_pos]
    record = {col: _clean_value(row[col]) for col in state.columns}
    df = _to_frame(record)
    return {
        "index": int(row.name),
        "original_prob": float(probs[chosen_pos]),
        "record": record,
        "detectors": _detector_breakdown(df),
    }


@app.post("/ood-check")
def ood_check(req: OODCheckRequest) -> dict[str, Any]:
    if not state.ready or state.model is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    df = _to_frame(req.record)
    prob_pos = float(state.model.predict_proba(df)[0, 1])
    gate = state.gates[req.gate]

    if gate is None:
        passed = True
        violation = 0.0
    else:
        passed = bool(gate.passes(df)[0])
        violation = float(gate.violation(df)[0])

    breakdown = _detector_breakdown(df)
    return {
        "gate": req.gate,
        "in_distribution": passed,
        "is_ood": not passed,
        "violation": violation,
        "favourable_prob": prob_pos,
        "detectors": breakdown,
        "detector_scores": {item["detector"]: item["score"] for item in breakdown},
        "detector_flags": {f"flag_{item['detector']}": item["ood_flag"] for item in breakdown},
    }


@app.post("/recourse")
def recourse(req: RecourseRequest) -> dict[str, Any]:
    if not state.ready or state.model is None or state.space is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    df = _to_frame(req.record)
    x_row = df.iloc[0]
    orig_prob = float(state.model.predict_proba(df)[0, 1])

    cfg = SearchConfig(
        pop_size=req.pop_size,
        generations=req.generations,
        target_prob=req.target_prob,
    )
    gate = state.gates[req.gate]
    res = find_recourse(
        state.model,
        state.space,
        x_row,
        state.donors,
        cfg=cfg,
        rng=np.random.default_rng(req.seed),
        gate=gate,
    )

    if not res.found or res.x_cf is None:
        return {
            "found": False,
            "gate": req.gate,
            "original_prob": orig_prob,
            "target_prob": req.target_prob,
            "detectors": _detector_breakdown(df),
            "reason": "No feasible counterfactual found within search budget and actionability/gate constraints.",
        }

    cf_df = pd.DataFrame([res.x_cf])[state.columns]
    for col in state.categorical_cols:
        if state.space and col in state.space.categories:
            cf_df[col] = pd.Categorical(cf_df[col], categories=state.space.categories[col])

    changed = {
        col: {"from": _clean_value(req.record[col]), "to": _clean_value(res.x_cf[col])}
        for col in state.columns
        if str(req.record[col]) != str(res.x_cf[col])
    }

    cf_breakdown = _detector_breakdown(cf_df)
    return {
        "found": True,
        "gate": req.gate,
        "original_prob": orig_prob,
        "counterfactual_prob": float(res.prob),
        "cost": float(res.cost),
        "n_changed": int(res.n_changed),
        "violation": float(res.violation),
        "changed_features": changed,
        "counterfactual": {k: _clean_value(v) for k, v in res.x_cf.items()},
        "detectors": cf_breakdown,
        "detector_flags": {f"flag_{item['detector']}": item["ood_flag"] for item in cf_breakdown},
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Conformal OOD Detector</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    .container { max-width: 1060px; margin: 0 auto; }
    h1 { margin: 0 0 6px; font-size: 22px; color: #f8fafc; }
    .sub { color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; }
    label { display: flex; flex-direction: column; font-size: 12px; color: #cbd5e1; gap: 4px; }
    select, input { background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; font-size: 13px; }
    button { background: #2563eb; color: white; border: none; border-radius: 6px; padding: 9px 14px; font-size: 13px; font-weight: 600; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    button.secondary { background: #334155; }
    button.secondary:hover { background: #475569; }
    button.ood-btn { background: #b45309; }
    button.ood-btn:hover { background: #92400e; }
    .banner { padding: 12px 14px; border-radius: 8px; font-weight: 700; font-size: 14px; margin-top: 10px; }
    .banner-in { background: rgba(22, 163, 74, 0.18); border: 1px solid #22c55e; color: #86efac; }
    .banner-ood { background: rgba(220, 38, 38, 0.2); border: 1px solid #ef4444; color: #fca5a5; }
    .banner-err { background: rgba(217, 119, 6, 0.2); border: 1px solid #f59e0b; color: #fde68a; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }
    .kpi { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; }
    .kpi .k { font-size: 11px; color: #94a3b8; text-transform: uppercase; }
    .kpi .v { font-size: 18px; font-weight: 700; margin-top: 4px; color: #f8fafc; }
    .ok { color: #4ade80; font-weight: 700; }
    .bad { color: #f87171; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    th, td { border-bottom: 1px solid #334155; padding: 8px 10px; text-align: left; }
    th { color: #94a3b8; font-weight: 600; }
    textarea { width: 100%; height: 155px; background: #0f172a; color: #cbd5e1; border: 1px solid #334155; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; box-sizing: border-box; }
    .muted { color: #94a3b8; font-size: 13px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Conformal Out-of-Distribution (OOD) Detector</h1>
    <div class="sub">Detecting out-of-distribution records via conformal gates (IF, LOF, kNN) and auditing recourse plausibility</div>

    <div class="card">
      <div class="controls">
        <label>Conformal Gate
          <select id="gate">
            <option value="gate_ALL" selected>gate_ALL (All detectors must pass)</option>
            <option value="gate_IF">gate_IF (Isolation Forest)</option>
            <option value="gate_LOF">gate_LOF (Local Outlier Factor)</option>
            <option value="none">none (No OOD gate)</option>
          </select>
        </label>
        <label>Sample Offset
          <input id="offset" type="number" min="0" max="200" value="0" style="width:85px" />
        </label>
        <button type="button" class="secondary" onclick="loadSample()">Load Normal Sample</button>
        <button type="button" class="ood-btn" onclick="injectOOD()">Inject OOD Anomaly</button>
        <button type="button" onclick="checkOOD()">Check OOD Status</button>
        <button type="button" class="secondary" onclick="runRecourse()">Test Recourse + Gate</button>
      </div>
      <div style="margin-top:12px;">
        <label>Record Features (JSON)
          <textarea id="recordBox"></textarea>
        </label>
      </div>
    </div>

    <div class="card">
      <strong>OOD Detection Report</strong>
      <div id="oodBanner" class="banner banner-in" style="display:none;"></div>
      <div class="grid" id="oodKpis"></div>
      <div id="detectorTable"></div>
      <div id="recourseExtra" style="margin-top:14px;"></div>
    </div>
  </div>

  <script>
    function showError(msg) {
      const banner = document.getElementById("oodBanner");
      banner.style.display = "block";
      banner.className = "banner banner-err";
      banner.textContent = msg;
    }

    function parseRecordBox() {
      const raw = document.getElementById("recordBox").value;
      if (!raw || !raw.trim()) {
        throw new Error("Record JSON is empty. Click 'Load Normal Sample' first.");
      }
      return JSON.parse(raw);
    }

    async function loadSample() {
      try {
        const offset = Number(document.getElementById("offset").value || 0);
        const res = await fetch(`/sample-record?offset=${offset}&min_prob=0.15`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to load sample");
        document.getElementById("recordBox").value = JSON.stringify(data.record, null, 2);
        await checkOOD();
      } catch (err) {
        showError("Error loading sample: " + err.message);
      }
    }

    async function injectOOD() {
      try {
        const rec = parseRecordBox();
        if ("hours-per-week" in rec) rec["hours-per-week"] = 99;
        if ("capital-gain" in rec) rec["capital-gain"] = 99999;
        if ("age" in rec) rec["age"] = 90;
        document.getElementById("recordBox").value = JSON.stringify(rec, null, 2);
        await checkOOD();
      } catch (err) {
        showError("Error injecting OOD: " + err.message);
      }
    }

    function renderDetectorTable(detectors) {
      if (!detectors || !detectors.length) return "";
      const rows = detectors.map(d => {
        const thr = d.conformal_threshold !== null && d.conformal_threshold !== undefined
          ? Number(d.conformal_threshold).toFixed(4) : "—";
        const status = d.ood_flag
          ? `<span class="bad">OOD (score &gt; threshold)</span>`
          : `<span class="ok">IN-DIST (pass)</span>`;
        return `<tr><td><b>${d.detector}</b></td><td>${Number(d.score).toFixed(4)}</td><td>${thr}</td><td>${status}</td></tr>`;
      }).join("");
      return `
        <table>
          <thead><tr><th>OOD Detector</th><th>Anomaly Score</th><th>Conformal Threshold</th><th>Status</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    async function checkOOD() {
      try {
        const payload = {
          record: parseRecordBox(),
          gate: document.getElementById("gate").value
        };
        const res = await fetch("/ood-check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "OOD check failed");

        const banner = document.getElementById("oodBanner");
        banner.style.display = "block";
        if (data.in_distribution) {
          banner.className = "banner banner-in";
          banner.textContent = `IN-DISTRIBUTION under ${data.gate} (violation = ${Number(data.violation).toFixed(4)})`;
        } else {
          banner.className = "banner banner-ood";
          banner.textContent = `OUT-OF-DISTRIBUTION (OOD) DETECTED under ${data.gate} (violation = ${Number(data.violation).toFixed(4)})`;
        }

        document.getElementById("oodKpis").innerHTML = `
          <div class="kpi"><div class="k">Gate Decision</div><div class="v ${data.in_distribution ? 'ok' : 'bad'}">${data.in_distribution ? 'PASS (In-Dist)' : 'REJECT (OOD)'}</div></div>
          <div class="kpi"><div class="k">OOD Violation</div><div class="v">${Number(data.violation).toFixed(4)}</div></div>
          <div class="kpi"><div class="k">Model Prob</div><div class="v">${Number(data.favourable_prob).toFixed(4)}</div></div>
        `;
        document.getElementById("detectorTable").innerHTML = renderDetectorTable(data.detectors);
        document.getElementById("recourseExtra").innerHTML = "";
      } catch (err) {
        showError("OOD check error: " + err.message);
      }
    }

    async function runRecourse() {
      const extra = document.getElementById("recourseExtra");
      try {
        extra.innerHTML = `<div class="muted">Running recourse search under selected gate...</div>`;
        const payload = {
          record: parseRecordBox(),
          gate: document.getElementById("gate").value,
          target_prob: 0.6,
          generations: 40,
          pop_size: 150,
          seed: 0
        };
        const res = await fetch("/recourse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Recourse request failed");

        if (!data.found) {
          extra.innerHTML = `<div class="muted">Recourse check: no feasible counterfactual found under <b>${data.gate}</b> within current constraints.</div>`;
          return;
        }
        const changes = Object.entries(data.changed_features || {})
          .map(([k, v]) => `<li><b>${k}</b>: ${v.from} &rarr; ${v.to}</li>`).join("");
        extra.innerHTML = `
          <div><b>Counterfactual Recourse under ${data.gate}:</b> prob ${Number(data.original_prob).toFixed(3)} &rarr; ${Number(data.counterfactual_prob).toFixed(3)} | cost: ${Number(data.cost).toFixed(4)} | changed: ${data.n_changed}</div>
          <ul style="margin:6px 0 0; padding-left:18px;">${changes || "<li>No changes</li>"}</ul>
          ${renderDetectorTable(data.detectors)}
        `;
      } catch (err) {
        extra.innerHTML = "";
        showError("Recourse error: " + err.message);
      }
    }

    window.addEventListener("DOMContentLoaded", () => { loadSample().catch(() => {}); });
  </script>
</body>
</html>"""
