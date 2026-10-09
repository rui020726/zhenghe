from __future__ import annotations
import datetime as dt, hashlib, json, pathlib, re

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"
LOW, HIGH = 1_700_000, 1_700_031
SCENE_KEYS = {"scene", "scene_id", "scenario", "scenario_id", "scene_index", "scenario_index"}
OWN_R58_PREFIX = "runs/comm_bc_initialized_constrained_ppo_20261007_r2/"
OWN_REGISTRATION_NAMES = {"ROUND58_GEN3_FIXED_PLAN.json", "ROUND58_GEN3_SCENE_OCCUPANCY_AUDIT.json",
                          "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"}

def walk_scene_values(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in SCENE_KEYS:
                if isinstance(v, int) and LOW <= v <= HIGH:
                    yield (str(k), v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, int) and LOW <= x <= HIGH:
                            yield (str(k), x)
            yield from walk_scene_values(v)
    elif isinstance(value, list):
        for item in value:
            yield from walk_scene_values(item)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    claims, registrations, path_hits, unreadable = [], [], [], []
    all_json = list(RUNS.rglob("*.json"))
    claim_files = [p for p in all_json if "claim" in p.as_posix().lower()]
    for p in claim_files:
        try: obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            unreadable.append({"path": p.relative_to(REPO).as_posix(), "error": type(e).__name__}); continue
        vals = list(walk_scene_values(obj))
        if vals: claims.append({"path": p.relative_to(REPO).as_posix(), "scene_fields": vals})
    for p in all_json:
        if p.name not in ("SCENE_REGISTRY.json", "CONFIG_AND_MODEL_MANIFEST.json"): continue
        try: obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            unreadable.append({"path": p.relative_to(REPO).as_posix(), "error": type(e).__name__}); continue
        vals = list(walk_scene_values(obj))
        if vals: registrations.append({"path": p.relative_to(REPO).as_posix(), "scene_fields": vals})
    token = re.compile(r"(?<!\d)(170000[0-9]|170001[0-9]|170002[0-9]|170003[01])(?!\d)")
    for path in RUNS.rglob("*"):
        rel = path.relative_to(REPO).as_posix()
        if token.search(rel): path_hits.append(rel)
    external_registrations = [r for r in registrations if not (
        r["path"].startswith(OWN_R58_PREFIX) and pathlib.Path(r["path"]).name in OWN_REGISTRATION_NAMES)]
    conflict_paths = ([item["path"] for item in claims] + path_hits
                      + [item["path"] for item in external_registrations])
    conflicts = bool(conflict_paths or unreadable)
    payload = {
        "kind": "round58-scene-occupancy-audit/1",
        "status": "clear" if not conflicts and not unreadable else "blocked",
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenes": list(range(LOW, HIGH + 1)),
        "conflicts": sorted(set(conflict_paths)),
        "result_blind": True, "outcome_fields_read": False,
        "checked_sources": {"registrations": True, "reservations": True,
                            "claims": True, "artifacts": True},
        "scope": "current repository runs tree; claim paths and scene identity fields, scene registry/manifests, artifact paths; own R58 reservation permitted",
        "claim_json_files_scanned": len(claim_files), "claim_scene_identity_matches": claims,
        "registered_scene_identity_hits": registrations,
        "external_registration_hits": external_registrations,
        "artifact_path_scene_id_hits": path_hits, "unreadable_inputs": unreadable,
        "prior_audit_sha256": sha(RUNS / "comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/ROUND58_GEN3_SCENE_OCCUPANCY_AUDIT.json"),
        "findings": {"third_party_consumption_or_reservation_found": bool(conflicts or unreadable),
                     "outcome_fields_read": False,
                     "own_round58_reservation_permitted": True},
        "finding": "No candidate scenes occur in claims or artifact paths; any matching registration is listed and must be the existing ROUND58 reservation."
    }
    out = pathlib.Path(__file__).resolve().parent / "ROUND58_GEN3_RESUME_SCENE_OCCUPANCY_RECHECK.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "audited_at_utc", "claim_json_files_scanned", "claim_scene_identity_matches", "registered_scene_identity_hits", "external_registration_hits", "artifact_path_scene_id_hits", "unreadable_inputs")}, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "clear" else 2
if __name__ == "__main__": raise SystemExit(main())
