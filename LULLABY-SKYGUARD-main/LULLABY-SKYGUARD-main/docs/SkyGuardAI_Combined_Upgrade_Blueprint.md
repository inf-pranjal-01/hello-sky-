# SkyGuard AI — Combined Final Upgrade Blueprint (v3, merged)

This merges two independently-produced plans into one:
- The uploaded **Final Detailed Development Strategy** (intelligence-layer
  focus: TrueSHAP, station regime, network corroboration, incident
  interpretation).
- The **tiered scalability plan** worked out in this chat (Tier 0–4:
  live-path correctness → real-time delivery → storage → distributed
  state → real ingestion → prod ops), mapped against the actual SIH
  rubric weightage.

Both plans are decision frameworks, not specifications. Everything below
still needs to be verified against the current branch before being
trusted — neither source document is ground truth on its own.

---

## 1. The one finding that changes both plans' assumptions

The uploaded strategy doc's entire evaluation section (baseline
reproduction, TrueSHAP validation, ablation studies) implicitly assumes
`evaluate.py`'s reported metrics (precision 0.767 / recall 0.807 / F1
0.786, per-fault recalls) describe **the system as deployed**.

They don't, currently. Confirmed by reading the repo directly:

```python
# model/evaluate.py, evaluate_all()
predicted = predicted | helper_alert | frozen_helper_alert
```

`fault_helper.py`'s ExtraTrees model is OR'd into the **offline eval
pipeline's** verdict — but `detect.py`, the module `main.py`/`state.py`/
`simulator.py` actually call for live scoring, never imports
`fault_helper` at all. So the numbers above are not what a judge would
see if they ran your live demo and compared it to your report.

**This must be Day 1 work, before anything else in either plan,
including the doc's own "reproduce the baseline" step** — reproducing
`evaluate.py`'s baseline without fixing this just re-confirms a number
the live system can't actually produce. Fixing it is also the natural
foundation for the doc's Track 4 (Network Corroboration): `state.py`
already computes per-cluster contemporaneous peer buffers on every
`ingest_reading()` call (`_neighbor_buffers()`), currently passed into
`score_reading()` as an ignored kwarg (leftover from the removed
spatial-features design). That's the same plumbing both the fault-helper
wiring *and* Track 4's peer-eligibility/corroboration logic need —
build it once, use it twice.

---

## 2. Direction (merged)

> Network-aware, season-aware, explainable weather-sensor fault
> intelligence — built on an internally-consistent, honestly-scaled
> production path.

Not "add another model." The system already has: Isolation Forest +
deterministic rules + temporal/causal features + cross-parameter
physics + fault classification + per-parameter health/circuit-breaker +
replay/live modes + persistent history + SHAP-oriented explanation + a
second ExtraTrees fault-helper + station clusters + peer-plumbing + a
built frontend. Novelty this cycle is in the **interpretation and
delivery layer**, not in adding a third detector.

---

## 3. Unified track list

Each track below is tagged with: which SIH rubric line(s) it serves,
whether it's from the intelligence-layer doc ("doc") or the scalability
plan ("chat"), and a priority tier (P0 = do first, regardless of
anything else).

| # | Track | Rubric line(s) | Source | Tier |
|---|---|---|---|---|
| A | Fault-helper live-path wiring + baseline correctness | Detection Accuracy (20%) | chat finding, feeds doc's Track 1 | **P0** |
| B | Reproduce baseline, freeze it, regression-guard it | Detection Accuracy (20%) | doc Track 1 | **P0** |
| C | Live event push (WebSocket/SSE, replacing 3–5s poll) | Real-Time (15%) | chat Tier 0.5, doc Track 6 | **P0** |
| D | Storage layer decision (see §4 — conditional, not assumed) | Scalability + Deployability (20%) | chat Tier 1, doc flags as conditional | P1 |
| E | Network corroboration (peer eligibility → LOCALIZED / REGIONAL / INSUFFICIENT) | Innovation (25%), Explainability (10%) | doc Track 4, built on chat's peer-buffer finding | P1 |
| F | TrueSHAP — validated, model-specific explanation | Explainability (10%), Innovation (25%) | doc Track 2 | P1 |
| G | Sensor health/recovery validation (not new code, testing) | Detection Accuracy, Deployability | doc Track 7 | P1 |
| H | Unified incident object (model + rules + regime + peers + uncertainty) | Explainability, Innovation | doc Track 5 | P2 |
| I | Station regime metadata (context-only, Option A) | Innovation (25%) | doc Track 3, Option A only | P2 |
| J | Dashboard integration of E/F/H/I | Visualization/UI (5%), Explainability | doc §23 | P2 |
| K | Architecture roadmap for Tiers 2–4 (Redis, real ingestion, prod ops) | Scalability, Deployability | chat | P2, diagram-only |
| L | Regime-as-model-feature, adaptive thresholds, learned regime classifier | — | doc, explicitly conditional | **not this cycle** |
| M | Energy efficiency / edge-readiness narrative + lightweight rule-only inference path | Energy Efficiency (5%) | new — see §3.5 | P1 |

---

## 3.5 Rubric coverage audit

Checked against all 8 lines, not just the 35% scalability subset:

| Criterion | Weight | Current coverage | Gap |
|---|---|---|---|
| Innovation & Novelty | 25% | Strong — hybrid detector, network corroboration, TrueSHAP, regime metadata, physics-grounded injector | Minor: make sure the presentation states *why* Isolation Forest was chosen over deep learning (see Track M) — that's a novelty-supporting design decision, not just an efficiency one |
| Detection Accuracy | 20% | Strong once Day 1 lands | None outstanding |
| Real-Time Capability | 15% | Good once Day 3 lands | Judges can't see a number unless you show one — add a visible latency readout (see Day 3 update below) |
| Explainability | 10% | Strong — TrueSHAP, unified incident object, operator action | None outstanding |
| Scalability | 10% | Good — Tier roadmap + evidence-based DB decision | None outstanding |
| Practical Deployability | 10% | Decent but mostly assumed — no explicit deployment checklist yet | Add a concrete Day 9/10 deployability checklist (see below) |
| Visualization/UI | 5% | Decent — frontend already built, Day 8 wires new evidence in | Low weight; don't overinvest, just don't neglect it |
| **Energy Efficiency** | **5%** | **None — currently zero coverage anywhere in either source plan** | See Track M below |

### Track M in detail — Energy Efficiency

The PS explicitly suggests "Edge AI for low-power deployment on ESP32."
Building real firmware in this window is scope creep with hardware risk
for 5% weight — don't. Two cheap, real things instead:

1. **Narrative, backed by an actual design decision you already made:**
   Isolation Forest + deterministic rules was chosen over deep learning
   partly *because* forest/rule-based scoring is cheap enough to run on
   constrained hardware, unlike an LSTM/Transformer. This connects
   directly to §5's "no deep learning for novelty's own sake" call —
   reframe that decision as an energy-efficiency argument too, not only
   a scope-risk one.
2. **One small real artifact:** extract the hard-fact rules that need
   no ML and no pandas — `physical_bounds`, `frozen` floor-match,
   `fail_low` floor comparison — as a standalone pure-Python function
   with no dependencies beyond the stdlib. Position this explicitly as
   "the edge-deployable subset": the ESP32 runs this cheap first-pass
   filter locally (catches the unambiguous faults immediately, no
   network round-trip), while the full Isolation Forest + fusion +
   fault-helper stack stays server-side for the harder cases. A rough
   op-count or timing comparison between the two paths (even a
   back-of-envelope one) is enough evidence for the rubric line — this
   isn't a research claim, it's an architecture argument.

This is a 2–3 hour task, not a day. Slot it into Day 9 alongside
scenario testing (see updated Day 9 below).

---

## 4. Resolving a real tension: TimescaleDB (Track D)

The two source plans disagree here and it's worth surfacing rather than
picking silently.

- **Chat plan's case for it:** `HistoryStore`'s CSV-trim-on-append is a
  real full-file-rewrite pattern that *would* degrade at scale, and
  Timescale gives an honest "Scalability + Deployability" story for the
  rubric's 20% combined weight there.
- **Doc's objection:** "Database migration without performance
  evidence" is explicitly listed under *What Not to Prioritize* (§22),
  and DB migration is listed as *Conditional*, not *Strong* (§21) — i.e.
  don't do it on architectural taste alone.

**Resolution:** the doc is right that this shouldn't be done on taste.
Settle it with a 30-minute benchmark, not a debate — time
`HistoryStore.get_recent()`/`trim()` at your actual station count (20)
and a stress-simulated count (200–500) before deciding. If CSV holds up
fine at 500 simulated stations, don't migrate — cite the benchmark
numbers in the presentation instead ("we measured this and the current
design holds to N stations; here's the migration plan for beyond that,"
which is *more* credible to judges than an unmeasured migration). If it
visibly degrades, migrate with the actual evidence in hand. Either
outcome is a stronger deployability story than doing it unmeasured.
This benchmark slots into Day 1 alongside baseline reproduction.

---

## 5. What NOT to prioritize (merged, both plans agree)

- Deep learning / LSTM / Transformer for novelty's own sake
- Regime-as-model-feature or adaptive thresholds (Option C/B) — doc is
  explicit these need controlled experiments this cycle doesn't have
  room for; Option A (context as metadata) only
- Redis-backed distributed state, real ingestion queue, k8s, CI/CD —
  roadmap-only, not built (chat plan's Tier 2–4)
- Claims of real-world accuracy based only on synthetic injected faults
  — say "performance against injected synthetic faults + independent
  USCRN rule-layer validation," not "real-world accuracy"
- Claims that SHAP proves physical causation, or that peer agreement
  proves a regional weather event — both docs agree: evidence, not
  proof
- 3D visualization, chatbot without operational value, large regime
  taxonomies before a simple regime layer is validated

---

## 6. Single unified 10-day plan

Each day ends with the system in a fully demoable state — never a
half-finished intermediate.

**Day 1 — Baseline + the one real bug**
- Wire `fault_helper` into `detect.py`'s live path via the existing
  `_neighbor_buffers()` plumbing (§1 above)
- Move `HELPER_ALERT_THRESHOLD`/`FROZEN_HELPER_ALERT_THRESHOLD` into
  `config.py` so `evaluate.py` and `detect.py` can't drift apart on them
- Reproduce precision/recall/F1 + per-fault-type recall on the current
  branch, **with the fix above in place** — this is the number you
  actually present
- Run the `HistoryStore` benchmark from §4; decide Track D now, not
  later

**Day 2 — Reliability**
- Validate frozen-value behavior, recovery/circuit-breaker transitions,
  missing/stale data handling (doc Track 7's test list) — fix
  regressions only, no new features

**Day 3 — Real-time delivery**
- WebSocket/SSE push for new-anomaly, health-change, and recovery
  events, replacing the 3–5s poll
- Measure actual end-to-end latency (reading → detection →
  persistence → dashboard) and keep the number — this is a rubric line
  item you can now quote precisely instead of gesturing at
- Surface that number as a small visible readout in the dashboard
  itself (e.g. "detection latency: 42ms") — a judge reading a claimed
  number in a slide is weaker than a judge watching it update live

**Day 4 — TrueSHAP, definition + implementation**
- Define which of the doc's 5 "TrueSHAP" meanings you're building
  (recommend: #1 exact deployed model + #4 unified model/rule
  separation — skip full background-dataset experimentation, no time)
- Implement the three-layer explanation split (model / rule /
  operational interpretation) — don't merge into one opaque confidence
  number

**Day 5 — TrueSHAP validation + network corroboration, part 1**
- Feature-perturbation and rank-stability spot checks (not the full
  experimental suite — time-boxed to "does this look sane," not a
  publishable validation study)
- Peer eligibility logic (freshness, peer health, timestamp alignment)
  using Day 1's now-active neighbor buffers

**Day 6 — Network corroboration, part 2**
- LOCALIZED / REGIONAL / INSUFFICIENT_CORROBORATION states
- If Day 1's benchmark said migrate: TimescaleDB swap happens here
  (isolated, testable, doesn't block anything else)

**Day 7 — Unified incident object + regime metadata (Option A only)**
- Build the merged incident JSON (detection + attribution + explanation
  + regime + network + uncertainty — doc §15's shape)
- Regime as **metadata only** — a labeled state shown alongside the
  alert, not fed into the model or thresholds

**Day 8 — Dashboard integration**
- Wire E/F/H/I into the frontend per doc §23's layout (header /
  detection / station context / network evidence / explanation /
  operator action)
- Operator actions presented as suggestions, never confirmed diagnoses

**Day 9 — Scenario testing + roadmap diagram**
- Run a cut of the doc's 20-scenario matrix (prioritize: isolated
  spike, frozen sensor, drift, fail-low, multivariate, faulty-target-
  with-healthy-peers, faulty-target-with-faulty-peers, too-few-peers,
  recovery-after-repair — skip the exhaustive regime-transition/new-
  station edge cases if time is short)
- Draw the Tier 2–4 roadmap diagram (Redis, real ingestion, prod ops) —
  presentation asset, not code
- Track M: extract the hard-fact rules into a standalone dependency-
  free function, frame as the edge-deployable subset (§3.5)

**Day 10 — Freeze + rehearse**
- Freeze the build. No new code.
- Before/after numbers, architecture diagrams, honest limitations slide
- Practical Deployability checklist (Track K/D deployment items pulled
  together, not new work — just confirm and document): CORS locked to
  real frontend origin, env-driven config (no hardcoded secrets/URLs),
  health-check endpoint, one-command or clearly documented deploy
  steps for both backend and frontend
- Rehearse the demo end-to-end, including the judge-likely question
  "how does this scale to a national network" (answer from the Day 9
  diagram, not improvised)

---

## 7. Defensible novelty statement (from the doc, unchanged — still holds)

> SkyGuardAI combines hybrid machine-learning and rule-based anomaly
> detection with station-aware temporal context, explainable evidence,
> and network corroboration to support interpretation of potential
> weather-sensor faults and unusual environmental events.

If Day 7's regime metadata and Day 4–5's TrueSHAP both land:

> The system additionally evaluates station-specific environmental
> regimes and provides model-aligned feature attribution while
> distinguishing localized anomalies from regionally corroborated
> changes.

Avoid: "first system of its kind," guaranteed diagnosis, true causal
explanation, real-world accuracy proven only by synthetic injection.

---

## 8. Final reminder

Both source plans converge on the same discipline: verify against the
running system before trusting any number in this document (including
the ones in it now), protect the reliable baseline over forcing a
novelty feature to completion, and if something's unstable on Day 9,
cut it rather than demo it broken.
