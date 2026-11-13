"""Verify the saved current experiment matrix without FEM or optimisation.

Requires NumPy and SciPy; reports are written only to --output.
"""
import argparse
from pathlib import Path
from functools import lru_cache
import collections
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
import numpy as np
from scipy import stats
from scipy.signal import convolve2d

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
DOC = args.results.resolve() / 'documentation'
R2 = args.source.resolve()
HERE = args.output.resolve()
if HERE == DOC or HERE.is_relative_to(DOC) or HERE == R2 or HERE.is_relative_to(R2):
    parser.error('Reports must be outside the input data and source directories.')
HERE.mkdir(parents=True, exist_ok=True)
FAILURES = []
COUNTS = collections.Counter()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def emit(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n")


def csvout(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def check(category, condition, detail):
    COUNTS[category] += 1
    if not bool(condition):
        FAILURES.append({"category": category, "detail": str(detail)})


def equal(a, b, *, rtol=1e-10, atol=1e-11):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))


def summary(values):
    a = np.asarray(values, float)
    n = len(a)
    mean = float(np.mean(a)) if n else None
    sd = float(np.std(a, ddof=1)) if n > 1 else None
    half = float(stats.t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else None
    return {"n": n, "values": a.tolist(), "mean": mean, "std": sd,
            "ci95_mean": [mean-half, mean+half] if half is not None else None}


@lru_cache(maxsize=None)
def run(family, method, seed):
    return read(DOC / f"{family}/{method}/seed_{seed}/run.json")


def events(r):
    return r["extra"]["r2_evidence"]["fem_events"]


def anchors(r):
    # E3 baselines label their shared 20 initial designs "other"; use the
    # actual ordered design hashes and FEM values, not just the role label.
    selected = ([e for e in events(r) if e["role"] not in
                 ("diagnostic_data", "independent_test")][:20]
                if r["family"] == "example3" else
                [e for e in events(r) if e["role"] == "initial_data"])
    return [(e["design_sha256"], e["compliance"]) for e in selected]


def value(r):
    return r["best_compliance"] / r["extra"]["solid_compliance"] if r["family"] == "example1" else r["best_compliance"]


def topk(a, b, k):
    k = min(int(k), len(a))
    if k == 0:
        return None
    for v in (a, b):
        ordered = np.sort(v)
        if k < len(v) and ordered[k-1] == ordered[k]:
            return None
    return len(set(np.argsort(a)[:k]).intersection(np.argsort(b)[:k])) / k


def field_hash(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


print("1/6: Dateiintegrität und vollständige Experimentmatrix", flush=True)
manifest = read(DOC / "artifact_manifest.json")
for name, digest in manifest.items():
    path = DOC / name
    check("artifact_hash", path.is_file() and sha(path) == digest, name)
source_before = {name: sha(R2 / "code" / name) for name in ("solvers.py", "reproduce_all.py")}
for name, digest in source_before.items():
    check("source_identity", sha(DOC / "source" / name) == digest, name)
index = read(DOC / "run_index.json")
complete = read(DOC / "completeness.json")
plan = read(DOC / "experiment_plan.json")
check("completeness", complete["full_publication_complete"] and not complete["missing"] and not complete["errors"], complete)
groups = collections.defaultdict(list)
for row in index:
    groups[(row["family"], row["method"])].append(row)
for rows in groups.values():
    rows.sort(key=lambda row: row["seed"])
for key, expected in plan["selected"].items():
    family, method = key.split("/")
    check("planned_seeds", [r["seed"] for r in groups[(family, method)]] == expected, key)
check("planned_groups", len(groups) == len(plan["selected"]) == 66, len(groups))
check("unique_rows", len({(r['family'], r['method'], r['seed']) for r in index}) == len(index) == 526, len(index))

details, times, designs, accuracy, aggregate = [], [], [], [], []
trajectory_limits, simp_curves = [], []
max_phase_gap = 0.
for row in index:
    family, method, seed = row["family"], row["method"], row["seed"]
    r = run(family, method, seed)
    extra = r["extra"]
    audit = extra["r2_evidence"]
    ev = audit["fem_events"]
    config = audit["configuration"]
    key = f"{family}/{method}/seed_{seed}"
    roles = collections.Counter(e["role"] for e in ev)
    search = [e for e in ev if e["role"] not in ("diagnostic_data", "independent_test")]
    diagnostic = len(ev)-len(search)
    logical = r["fem_calls_total"]
    raw = extra.get("raw_run_fem_calls", logical)
    expected_recorded = raw
    note = "Suchbudget und zusätzliche Diagnostik getrennt."
    if family == "example1" and method.startswith("de"):
        expected_recorded += 1
        note = "Ein zusätzlicher terminaler FEM-Aufruf."
    if family == "example2" and method == "pca_projection_study":
        expected_recorded -= 81
        note = "81 FEM-Aufrufe des verschachtelten SIMP-Referenzlaufs im logischen Gesamtbudget; nicht im äußeren Ereigniskonto."
    if extra.get("fem_checkpoint"):
        note += " 156er-Zielwert ist ein Budgetausschnitt; Zeit/Ereignisse gehören zum vollständigen 160er-Ursprungslauf."
    check("budget_accounting", len(search) == expected_recorded, (key, logical, raw, len(search), note))
    check("event_sequence", [e["call"] for e in ev] == list(range(1, len(ev)+1)), key)
    check("index_metrics", equal(value(r), row["normalized_compliance"] if family == "example1" else row["compliance"])
          and logical == row["logical_fem_calls"] and len(ev) == row["recorded_fem_calls"]
          and dict(roles) == row["fem_calls_by_role"] and equal(row["solver_wall_time_s"], r["wall_time_s"]), key)
    check("run_source", config["source_sha256"] == source_before["solvers.py"], key)
    check("executed_sources", config.get("executed_sources") == source_before, key)
    check("execution_protocol", config["numerical_execution"] == plan["numerical_execution"], key)
    check("finite_endpoint", math.isfinite(value(r)), key)
    phases = audit["exclusive_seconds"]
    gap = sum(phases.values()) - audit["elapsed_s"]
    max_phase_gap = max(max_phase_gap, abs(gap))
    check("time_accounts", abs(gap) < .001 and all(math.isfinite(v) and v >= 0 for v in phases.values()), (key, gap))
    check("finite_runtime", math.isfinite(r["wall_time_s"]) and r["wall_time_s"] >= 0, key)
    finite_trajectory = [(int(n), float(v)) for n, v in r["trajectory"] if v is not None and math.isfinite(v)]
    if family != "mesh_refinement":
        check("trajectory_monotone", all(b[0] >= a[0] and b[1] <= a[1]+1e-8 for a,b in zip(finite_trajectory, finite_trajectory[1:])), key)
        if family == "example2" and method in ("simp", "simp_multistart_30"):
            # Stored history is only the best start's OC iterations, before
            # its final FEM evaluation. Verify every actual start endpoint.
            offset = 0
            for start in extra["starts"]:
                offset += start["fem_calls"]
                check("simp_start_endpoint", equal(search[offset-1]["compliance"], start["compliance"]), (key, start["start_idx"]))
            check("simp_final_endpoint", offset == logical == len(search)
                  and equal(min(s["compliance"] for s in extra["starts"]), r["best_compliance"]), key)
            check("simp_history", equal([v for _,v in finite_trajectory], np.minimum.accumulate(extra["objective_history"])), key)
            running = math.inf
            for call, event in enumerate(search, 1):
                running = min(running, event["compliance"])
                simp_curves.append({"method":method,"seed":seed,"cumulative_fem_calls":call,"best_compliance":running})
            note += " Gespeicherte trajectory enthält nur die OC-Historie des besten Einzelstarts ohne abschließenden FEM-Aufruf; keine kumulative Multistartkurve."
            trajectory_limits.append({"source":key,"stored_last":finite_trajectory[-1],"verified_final":r["best_compliance"],"note":note,
                                      "replacement":"simp_verlaeufe_aus_fem.csv: kumulativer Bestwert aller gespeicherten FEM-Ereignisse"})
        elif family == "example2" and method == "pca_projection_study":
            evaluations = [e for e in search if e["role"] != "initial_data"]
            projections = list(extra["projections"].values())
            check("projection_endpoints", len(evaluations) == len(projections)+1
                  and equal(evaluations[0]["compliance"], extra["simp_direct_binary_compliance"])
                  and all(equal(e["compliance"], p["projected_simp_binary_compliance"]) for e,p in zip(evaluations[1:],projections))
                  and equal(min(p["projected_simp_binary_compliance"] for p in projections), r["best_compliance"]), key)
            check("projection_history", equal([v for _,v in finite_trajectory], np.minimum.accumulate([e["compliance"] for e in evaluations])), key)
            note += " trajectory schließt die direkte SIMP-Referenz ein; finaler Projektionswert umfasst nur die sieben Projektionen. Keine Optimierungskurve."
            trajectory_limits.append({"source":key,"stored_last":finite_trajectory[-1],"verified_final":r["best_compliance"],"note":note,
                                      "replacement":"Sieben Projektionswerte und SIMP-Referenz separat tabellieren."})
        else:
            check("trajectory_endpoint", bool(finite_trajectory) and equal(finite_trajectory[-1][1], r["best_compliance"]), key)
    if family == "example3":
        check("e3_budget_and_selection", logical == len(search) == 150 and equal(min(e["compliance"] for e in search), r["best_compliance"]), key)
        expected_test = 16 if method.startswith(("pisco", "scalar")) else 0
        check("e3_test_budget", diagnostic == expected_test, (key, diagnostic))
    if family == "example1" and method.endswith("fixed_156"):
        check("e1_fixed_budget", logical == 156 and equal(min(e["compliance"] for e in search[:156]), r["best_compliance"]), key)
    if family == "example2" and "diagnostic_fem_calls_excluded" in extra:
        check("e2_diagnostic_budget", diagnostic == extra["diagnostic_fem_calls_excluded"] == 32, key)
    if family == "example2" and method != "pca_projection_study":
        binary_objective = method in ("pisco_binary", "pisco_strong_anchors", "latent_cmaes", "latent_ard_bo")
        eligible = [e for e in search if bool(e['binary']) == binary_objective]
        check("e2_final_selection", bool(eligible) and equal(min(e['compliance'] for e in eligible), r['best_compliance']), key)
        if method == "filtered_binary_direct_fem":
            check("binary_before_filter_semantics", not any(e['binary'] for e in search)
                  and extra['filter_order'] == 'decoder_clip -> exact_cardinality_binary_projection -> cone_filter -> volume_projection -> FEM', key)
            note += " Binär nur VOR dem Kegelfilter; FEM und Enddesign sind kontinuierlich. Nicht als binäre Hauptmetrik ausgeben."
    if family == "example2" and method == "no_periodic_verification":
        check("frozen_retraining_semantics", len(audit["training"]) == 1 and extra["actual_T"] == 12
              and len(audit["predictions"]) == 72 and roles["other"] == 72 and not config["config"]["active_retraining"], key)
    stored_design = r["best_design"]
    if stored_design is not None:
        arr = np.load(DOC / key / "best_design.npy", allow_pickle=False)
        check("design_json_npy", np.array_equal(arr, np.asarray(stored_design)), key)
        if family == "example1":
            x,y,rx,ry,angle = arr
            p = config["problem"]
            area_error = abs(rx*ry/p["target_hole_radius_mm"]**2-1)
            bx = math.hypot(rx*math.cos(angle), ry*math.sin(angle))
            by = math.hypot(rx*math.sin(angle), ry*math.cos(angle))
            margin = 2*p["Lx_mm"]/p["nelx"]
            feasible = x-bx >= margin-1e-8 and x+bx <= p["Lx_mm"]-margin+1e-8 and y-by >= margin-1e-8 and y+by <= p["Ly_mm"]-margin+1e-8
            check("ellipse_constraints", area_error <= 1e-6 and feasible, key)
            designs.append({"family":family,"method":method,"seed":seed,"area_relative_error":area_error,"margin_mm":margin,"feasible":feasible})
        elif family == "example2" or family == "mesh_refinement":
            check("density_bounds", np.isfinite(arr).all() and arr.min() >= -1e-7 and arr.max() <= 1+1e-7, key)
        elif family == "example3":
            check("e3_final_design_event", any(e['design_sha256'] == field_hash(arr)
                  and equal(e['compliance'],r['best_compliance']) for e in search), key)
    else:
        check("absent_design_justified", bool(extra.get("checkpoint_design_not_stored")), key)
    testfile = read(DOC / key / "independent_test.json")
    rawtest = testfile.get("raw", [])
    for rep in sorted({v.get("representation", "unspecified") for v in rawtest}):
        samples = [v for v in rawtest if v.get("representation", "unspecified") == rep]
        a = np.array([v["actual"] for v in samples]); b = np.array([v["predicted"] for v in samples])
        accuracy.append({"family":family,"method":method,"seed":seed,"representation":rep,"n":len(a),
                         "mean_relative_error":float(np.mean(abs(a-b)/np.maximum(abs(a),1e-12))),
                         "spearman":float(stats.spearmanr(a,b).statistic) if np.ptp(a) and np.ptp(b) else None})
    if rawtest:
        a = np.array([v["actual"] for v in rawtest]); b = np.array([v["predicted"] for v in rawtest])
        for metric, computed in (("mae",np.mean(abs(a-b))), ("rmse",np.sqrt(np.mean((a-b)**2))),
                                 ("mean_relative_error",np.mean(abs(a-b)/np.maximum(abs(a),1e-12)))):
            check("test_metrics", equal(computed,testfile[metric]), (key,metric))
    details.append({"family":family,"method":method,"seed":seed,"value":value(r),"logical_search_calls":logical,
                    "raw_run_search_calls":raw,"recorded_search_calls":len(search),"diagnostic_calls":diagnostic,
                    "constructor_calls":len(audit.get("construction",{}).get("fem_events",[])),
                    "roles":dict(roles),"note":note,"training_updates":len(audit["training"]),
                    "training_steps":sorted({t["steps_executed"] for t in audit["training"]}),
                    "stop_reason":extra.get("stop_reason"),"actual_cycles":extra.get("actual_T"),
                    "from_cache":extra.get("from_cache"),"resumed":audit.get("resumed")})
    times.append({"family":family,"method":method,"seed":seed,"solver_wall_time_s":r["wall_time_s"],
                  "audit_elapsed_s":audit["elapsed_s"],"exclusive_sum_s":sum(phases.values()),"sum_gap_s":gap,
                  "constructor_s":audit.get("construction",{}).get("elapsed_s"),
                  "training_s":phases.get("surrogate_training",0),"candidate_search_s":phases.get("candidate_search",0),
                  "all_fem_s":sum(v for k,v in phases.items() if k.startswith("fem_")),"exclusive_seconds":phases})
    if len(details) % 100 == 0:
        print(f"  {len(details)}/{len(index)} Ergebniszeilen gelesen", flush=True)

# Six sensitivity rows reuse three main runs twice. Do not call these independent experiments.
for method in ("pisco_starts_12_d35", "pisco_top_6_d35"):
    for seed in (0,1,2):
        a,b = run("example3",method,seed),run("example3","pisco_d35",seed)
        check("sensitivity_alias", all(a[k] == b[k] for k in ("best_compliance","best_design","trajectory","wall_time_s","extra")), (method,seed))

published = read(DOC / "statistics.json")["examples"]
for (family,method), rows in groups.items():
    rr = [run(family,method,row['seed']) for row in rows]
    s = summary([value(r) for r in rr]); saved=published[family][method]
    for key in ("n","mean","std"):
        check("method_summary", equal(s[key],saved[key]), (family,method,key))
    check("method_summary", equal(s["ci95_mean"],saved["ci95_mean"]), (family,method,"ci95"))
    check("method_summary", s['values'] == saved['per_seed'] and [r['seed'] for r in rr] == saved['seeds'], (family,method,"vectors"))
    tt = [r for r in times if r['family']==family and r['method']==method]
    aggregate.append({"family":family,"method":method,**s,"seeds":[r['seed'] for r in rr],
                      "search_fem_mean":float(np.mean([r['fem_calls_total'] for r in rr])),
                      "wall_time_mean_s":float(np.mean([r['wall_time_s'] for r in rr])),
                      "wall_time_sd_s":float(np.std([r['wall_time_s'] for r in rr],ddof=1)) if len(rr)>1 else None,
                      "training_mean_s":float(np.mean([r['training_s'] for r in tt])),
                      "search_mean_s":float(np.mean([r['candidate_search_s'] for r in tt])),
                      "fem_mean_s":float(np.mean([r['all_fem_s'] for r in tt])),
                      "training_share_of_solver_time":sum(r['training_s'] for r in tt)/sum(r['solver_wall_time_s'] for r in tt)})
csvout("laufpruefung.csv",details);csvout("laufzeiten.csv",times)
csvout("methodenvergleich.csv",aggregate);csvout("designpruefung.csv",designs)
csvout("testgenauigkeit.csv",accuracy)
emit("darstellungsgrenzen.json", trajectory_limits)
csvout("simp_verlaeufe_aus_fem.csv", simp_curves)
phase_summary = []
for method in ("pisco_dynamic_pca", "pisco_binary"):
    rows = [r for r in times if r['family'] == 'example2' and r['method'] == method]
    for phase in rows[0]['exclusive_seconds']:
        s = summary([r['exclusive_seconds'][phase] for r in rows])
        phase_summary.append({'method':method,'phase':phase,**s,
                              'share_of_audit_total':sum(s['values'])/sum(r['audit_elapsed_s'] for r in rows)})
csvout('zeitanteile_e2.csv',phase_summary)

print("2/6: Binäre Netzprüfung und kontinuierliche Designmetriken", flush=True)
ny,nx = 36,60
xx,yy = np.meshgrid((np.arange(nx)+.5)/nx,(np.arange(ny)+.5)/ny)
mask = ~((xx>.58)&(yy>.58))
rmin = 2.4
q = np.arange(-2,3)
kernel = np.maximum(0,rmin-np.hypot(q[:,None],q[None,:]))
denominator = convolve2d(np.ones((ny,nx)),kernel,mode="same")

def physical_and_binary(raw):
    filtered = convolve2d(np.where(mask,np.asarray(raw,float),0),kernel,mode="same")/denominator
    a = filtered[mask];lo,hi=-1.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        if np.mean(np.clip(a+mid,0,1)) < .45:lo=mid
        else:hi=mid
    physical=np.zeros((ny,nx));physical[mask]=np.clip(a+(lo+hi)/2,0,1)
    indices=np.argsort(physical[mask],kind="stable")[-round(.45*mask.sum()):]
    b=np.zeros(mask.sum());b[indices]=1
    binary=np.zeros((ny,nx));binary[mask]=b
    return physical,binary

post=read(DOC/'manuscript/postprocessing_evidence.json')
design_metrics=read(DOC/'manuscript/e2_design_metrics.json')
check('postprocessing_budget',len(post['fem_events'])==20,len(post['fem_events']))
postchecked=[]
for i,row in enumerate(design_metrics):
    r=run('example2',row['method'],row['seed'])
    rho,binary=physical_and_binary(r['extra']['raw_design'])
    for metric,expected in [('volume_fraction',rho[mask].mean()),('grayness',4*np.mean(rho[mask]*(1-rho[mask]))),('binary_volume_fraction',binary[mask].mean())]:
        check('design_metric',equal(row[metric],expected),(row['source'],metric))
    jc,jb=post['fem_events'][2*i:2*i+2]
    check('design_postprocessing_evidence',equal(row['common_filtered_compliance'],jc['compliance'])
          and equal(row['volume_preserving_binary_compliance'],jb['compliance'])
          and not jc['binary'] and jb['binary'] and jb['design_sha256']==field_hash(binary),row['source'])
    check('design_volume',abs(rho[mask].mean()-.45)<1e-12 and int(binary[mask].sum())==round(.45*mask.sum()),row['source'])
    check('continuous_endpoint_consistency',equal(row['common_filtered_compliance'],r['best_compliance'],rtol=1e-6),row['source'])
    postchecked.append({**row,'independent_grayness':float(4*np.mean(rho[mask]*(1-rho[mask]))),
                        'binary_hash_matches_recorded_fem':jb['design_sha256']==field_hash(binary)})

meshrows=[]
for method in ('pisco_binary','simp'):
    for row in groups[('mesh_refinement',method)]:
        seed=row['seed'];r=run('mesh_refinement',method,seed);source=run('example2',method,seed)
        coarse=np.asarray(source['best_design']) if method=='pisco_binary' else physical_and_binary(source['extra']['raw_design'])[1]
        fine=np.repeat(np.repeat(coarse,2,axis=0),2,axis=1)
        ev=events(r);a,b=r['extra']['mesh_refinement']
        check('binary_mesh_fields',np.isin(coarse,[0.,1.]).all() and field_hash(coarse)==ev[0]['design_sha256'] and field_hash(fine)==ev[1]['design_sha256'],(method,seed))
        check('binary_mesh_values',equal(a['compliance'],ev[0]['compliance']) and equal(b['compliance'],ev[1]['compliance']) and equal(b['compliance'],r['best_compliance']), (method,seed))
        if method=='pisco_binary':check('binary_main_endpoint',equal(a['compliance'],source['best_compliance']),seed)
        check('binary_mesh_protocol',a['binary'] and b['binary'] and not a['geometry_changed'] and not b['geometry_changed']
              and a['volume_fraction']==b['volume_fraction'] and a['total_vertical_force']==b['total_vertical_force']==-1.,(method,seed))
        check('binary_mesh_fraction',equal(a['volume_fraction'],coarse[mask].mean()),(method,seed))
        meshrows.append({'method':method,'seed':seed,'coarse':a['compliance'],'fine':b['compliance'],
                         'change_percent':100*(b['compliance']/a['compliance']-1),'volume':a['volume_fraction'],
                         'source':f'mesh_refinement/{method}/seed_{seed}/run.json'})
csvout('binaere_netzpruefung.csv',meshrows);csvout('designmetriken.csv',postchecked)

print('3/6: Learning curves, populations and current numerical summaries',flush=True)
learning=[]
for seed in range(7,16):
    r=run('example2','pisco_dynamic_pca_fixed_cycles',seed)
    cycles={c['cycle']:c for c in r['extra']['cycle_metrics']}
    check('holdout_population',cycles[1]['holdout_true_compliance']==cycles[12]['holdout_true_compliance']
          and cycles[1]['holdout_representations']==cycles[12]['holdout_representations'],seed)
    for cycle in (1,12):
        c=cycles[cycle];true=np.array(c['holdout_true_compliance']);pred=np.array(c['holdout_predicted_compliance']);reps=np.array(c['holdout_representations'])
        check('holdout_population',len(true)==32 and sum(reps=='continuous')==sum(reps=='binary')==16,(seed,cycle))
        for rep in ('continuous','binary','pooled'):
            select=np.ones(len(true),bool) if rep=='pooled' else reps==rep
            a,b=true[select],pred[select]
            learning.append({'seed':seed,'cycle':cycle,'representation':rep,'n_designs':len(a),
                             'rel_error_mean':float(np.mean(abs(a-b)/np.maximum(abs(a),1e-12))),
                             'spearman':float(stats.spearmanr(a,b).statistic) if np.ptp(a) and np.ptp(b) else None,
                             'top_k_overlap':topk(a,b,c['holdout_top_k']),
                             'equilibrium_residual_mean':c.get('holdout_equilibrium_residual_mean') if rep=='pooled' else None,
                             'source':f'example2/pisco_dynamic_pca_fixed_cycles/seed_{seed}/run.json'})
generated_learning=read(DOC/'manuscript/e2_learning_by_seed.json')
for row in learning:
    saved=next(r for r in generated_learning if all(r[k]==row[k] for k in ('seed','cycle','representation')))
    for k in ('n_designs','rel_error_mean','spearman','top_k_overlap','equilibrium_residual_mean'):
        check('learning_metrics',equal(row[k],saved[k]),(row['seed'],row['cycle'],row['representation'],k))
csvout('lernkurven.csv',learning)

claims=read(DOC/'manuscript/claims.json')
check('claim_coverage',len(claims)==92 and len({r['id'] for r in claims})==92,'92 current numerical summaries')
e2_alias={
 'pisco_pca':'pisco_dynamic_pca','pisco_pca_static':'pisco_static_pca',
 'pisco_pca_fixed_budget':'pisco_dynamic_pca_fixed_cycles','pisco_pca_static_fixed_budget':'pisco_static_pca_fixed_cycles',
 'pisco_vae':'pisco_vae','pisco_ae':'pisco_ae','pisco_vae_nophys':'pisco_vae_lambda_0','pisco_vae_strong':'pisco_vae_lambda_1',
 'pisco_vae_purephys':'pisco_vae_data_lambda_0','scalar_surrogate':'scalar_surrogate','no_reeval':'no_periodic_verification',
 'latent_fem':'latent_fem','simp_free_init':'pisco_random_basis','pisco_binary_objective':'pisco_binary',
 'pisco_strong_anchors':'pisco_strong_anchors','latent_cma_es':'latent_cmaes','latent_ard_bo':'latent_ard_bo'}

def endpoint(identifier):
    family='example'+identifier[1];suffix=identifier[3:]
    if family=='example1':
        method=suffix.removeprefix('fixed156_')+'_fixed_156' if suffix.startswith('fixed156_') else {'pisco_lam0':'pisco_lambda_0','pisco_lam0_001':'pisco_lambda_0.001','pisco_lam0_1':'pisco_lambda_0.1'}.get(suffix,suffix)
    elif family=='example2':method=e2_alias.get(suffix)
    else:method={'sklearn_bo_d35':'ard_bo_d35','scalar':'scalar_surrogate_d35','no_physics':'pisco_no_physics_d35','nested_pisco':'pisco_progressive','nested_bo':'ard_bo_progressive'}.get(suffix,suffix.replace('bo_ard_','ard_bo_'))
    return (family,method) if (family,method) in groups else None


def resolve(identifier):
    direct=endpoint(identifier)
    if direct:
        rr=[run(*direct,r['seed']) for r in groups[direct]]
        return [value(r) for r in rr],[r['seed'] for r in rr],[f'{direct[0]}/{direct[1]}/seed_{r["seed"]}/run.json' for r in rr]
    if 'min_feature_size' in identifier:return [],[],[]
    if identifier.startswith('e2_fine_mesh_'):
        method='simp' if identifier.endswith('simp') else 'pisco_binary'
        rows=[r for r in meshrows if r['method']==method]
        return [r['fine'] for r in rows],[r['seed'] for r in rows],[r['source'] for r in rows]
    if identifier.startswith(('e2_design_','e2_simp_design_')):
        method='simp' if identifier.startswith('e2_simp_') else 'pisco_dynamic_pca'
        prefix='e2_simp_design_' if method=='simp' else 'e2_design_'
        metric=identifier.removeprefix(prefix);rows=[r for r in postchecked if r['method']==method]
        return [r[metric] for r in rows],[r['seed'] for r in rows],[r['source'] for r in rows]
    if identifier.startswith('e2_decoder_'):
        metric,reducer={'raw_rho_min_global':('raw_min',min),'raw_rho_max_global':('raw_max',max),
                        'raw_clipped_fraction_mean':('clipped_fraction_mean',np.mean),
                        'projected_volume_max_abs_error':('projected_volume_max_abs_error',max)}[identifier.removeprefix('e2_decoder_')]
        rr=[run('example2','pisco_dynamic_pca',i) for i in range(7,16)]
        return [float(reducer([r['extra']['decoder_diagnostics_all_candidates'][metric] for r in rr]))],list(range(7,16)),[f'example2/pisco_dynamic_pca/seed_{i}/run.json' for i in range(7,16)]
    if identifier.startswith('e2_projected_simp_'):
        row=groups[('example2','pca_projection_study')][0];r=run('example2','pca_projection_study',row['seed'])
        return [r['extra']['projections'][identifier.removeprefix('e2_projected_simp_')]['projected_simp_binary_compliance']],[row['seed']],[row['evidence']]
    if identifier.startswith('e2_cycle'):
        head,metric=identifier.split('_holdout_');cycle=int(head.removeprefix('e2_cycle'));rep='pooled' if metric=='equilibrium_residual_mean' else 'continuous'
        valid={seed for seed in range(7,16) if all(any(r['seed']==seed and r['cycle']==c and r['representation']==rep and r[metric] is not None for r in learning) for c in (1,12))}
        rows=[r for r in learning if r['cycle']==cycle and r['representation']==rep and r['seed'] in valid]
        return [r[metric] for r in rows],[r['seed'] for r in rows],[r['source'] for r in rows]
    timekeys={'e2_pisco_wall_time':('pisco_dynamic_pca','wall_time_s'), 'e2_simp_wall_time':('simp','wall_time_s'),
              'e2_surrogate_training_s':('pisco_dynamic_pca','surrogate_training_s'),
              'e2_latent_optimisation_s':('pisco_dynamic_pca','latent_optimisation_s'), 'e2_active_fem_s':('pisco_dynamic_pca','active_fem_s')}
    method,key=timekeys[identifier];rr=[run('example2',method,row['seed']) for row in groups[('example2',method)]]
    return [r['wall_time_s'] if key=='wall_time_s' else r['extra']['timing_breakdown_s'][key] for r in rr],[r['seed'] for r in rr],[f'example2/{method}/seed_{r["seed"]}/run.json' for r in rr]


ledger=[]
for claim in claims:
    ident=claim['id'];vals,seeds,sources=resolve(ident);computed=summary(vals)
    for key in ('n','values','mean','std','ci95_mean'):
        check('claim_arithmetic',equal(computed[key],claim[key]),(ident,key))
    check('claim_provenance',seeds==claim['seeds'] and sources==claim['sources'],ident)
    if ident.startswith('e2_cycle'):
        check('learning_exclusions',claim['excluded_seeds']==sorted(set(range(7,16))-set(seeds)) and claim['cohort_n']==9,ident)
    ledger.append({'id':ident,**computed,'seeds':seeds,'sources':sources,'definition':claim['definition']})
csvout('current_summaries.csv',ledger)

print('4/6: 40 statistische Vergleiche, Paarung, Konfidenzintervalle und Holm-Familien',flush=True)
@lru_cache(maxsize=None)
def combinations(n,k):
    return np.array(list(itertools.combinations(range(n),k)))

def statistical_test(a,b,paired):
    a,b=np.asarray(a,float),np.asarray(b,float)
    result=stats.ttest_rel(a,b) if paired else stats.ttest_ind(a,b,equal_var=False)
    ci=result.confidence_interval(.95);observed=abs(a.mean()-b.mean());eps=1e-12*max(1,observed)
    if paired:
        perm=np.array(list(itertools.product((-1,1),repeat=len(a))))@(a-b)/len(a)
    else:
        v=np.r_[a,b];sums=v[combinations(len(v),len(a))].sum(axis=1)
        perm=sums/len(a)-(v.sum()-sums)/len(b)
    return {'mean_a':float(a.mean()),'mean_b':float(b.mean()),'mean_difference_a_minus_b':float(a.mean()-b.mean()),
            'ci95_difference_lo':float(ci.low),'ci95_difference_hi':float(ci.high),'p_raw':float(result.pvalue),
            'p_exact_permutation_sensitivity':float(np.mean(abs(perm)>=observed-eps)),'t_statistic':float(result.statistic)}

testdefs=[]
for suffix,family in (('', 'native'),('_fixed_156','fixed156')):
    for other in ('de','bo','cmaes'):
        testdefs.append(('example1','example1','pisco'+suffix,other+suffix,family,False,None))
for other in ('pisco_lambda_0','pisco_lambda_0.001','pisco_lambda_0.1'):
    testdefs.append(('example1','example1','pisco',other,'physics_weight',True,None))
for left,right,family in [
 ('pisco_dynamic_pca','pisco_static_pca','pca_representation'),
 ('pisco_dynamic_pca_fixed_cycles','pisco_static_pca_fixed_cycles','pca_representation'),
 ('pisco_dynamic_pca_fixed_cycles','no_periodic_verification','matched_controls'),
 ('pisco_dynamic_pca_fixed_cycles','field_no_physics_matched','matched_controls'),
 ('pisco_dynamic_pca_fixed_cycles','scalar_surrogate','matched_controls'),
 ('pisco_binary','latent_cmaes','binary_baselines'),('pisco_binary','latent_ard_bo','binary_baselines'),
 ('pisco_vae','pisco_vae_lambda_0','vae_loss'),('pisco_vae','pisco_vae_lambda_1','vae_loss')]:
    testdefs.append(('example2_exploratory','example2',left,right,family,True,None))
special={
 'field surrogate vs scalar surrogate at d=35':('pisco_d35','scalar_surrogate_d35'),
 'physics residual vs no physics at d=35':('pisco_d35','pisco_no_physics_d35'),
 'no-physics field output vs scalar output at d=35':('pisco_no_physics_d35','scalar_surrogate_d35'),
 'coarse-to-fine PISCO vs coarse-to-fine ARD-BO':('pisco_progressive','ard_bo_progressive'),
 'coarse-to-fine PISCO vs static PISCO at d=243':('pisco_progressive','pisco_d243'),
 'coarse-to-fine ARD-BO vs static ARD-BO at d=243':('ard_bo_progressive','ard_bo_d243')}
for t in read(Path(__file__).with_name('statistical_protocol.json'))['example3']:
    label=t['label']
    if label in special:left,right=special[label]
    else:
        d=label.split('d=')[-1];left='pisco_d'+d
        right=('ard_bo' if 'ARD-BO' in label else 'cmaes' if 'CMA-ES' in label else 'de' if ' vs DE' in label else 'random')+'_d'+d
    testdefs.append(('example3','example3',left,right,t['family'],'paired' in t['test'],label))
generated=read(DOC/'manuscript/statistical_tests.json')
tests=[]
for output,example,left,right,family,paired,label in testdefs:
    aa=[run(example,left,r['seed']) for r in groups[(example,left)]]
    bb=[run(example,right,r['seed']) for r in groups[(example,right)]]
    match=sum(anchors(a)==anchors(b) and bool(anchors(a)) for a,b in zip(aa,bb))
    if paired:check('statistical_pairing',match==len(aa)==len(bb) and [r['seed'] for r in aa]==[r['seed'] for r in bb],(output,left,right,match))
    row={'output':output,'family':family,'label':label or left+' vs '+right,'left':left,'right':right,
         'test':'paired' if paired else 'Welch','n_a':len(aa),'n_b':len(bb),'matching_initial_data_pairs':match,
         'same_search_budget_pairs':sum(a['fem_calls_total']==b['fem_calls_total'] for a,b in zip(aa,bb)),
         **statistical_test([value(a) for a in aa],[value(b) for b in bb],paired)}
    tests.append(row)
for key in {(r['output'],r['family']) for r in tests}:
    family=sorted([r for r in tests if (r['output'],r['family'])==key],key=lambda r:r['p_raw']);prev=0.
    for i,r in enumerate(family):
        prev=max(prev,min(1,r['p_raw']*(len(family)-i)));r['p_holm']=prev;r['significant_holm_005']=prev<.05
    sensitivity=sorted(family,key=lambda r:r['p_exact_permutation_sensitivity']);prev=0.
    for i,r in enumerate(sensitivity):
        prev=max(prev,min(1,r['p_exact_permutation_sensitivity']*(len(sensitivity)-i)))
        r['p_exact_permutation_holm_sensitivity']=prev
        r['significant_exact_holm_sensitivity']=prev<.05
        r['test_sensitive_decision']=r['significant_holm_005'] != (prev<.05)
for r in tests:
    saved=next(t for t in generated[r['output']] if t['label']==r['label'])
    for k in ('mean_a','mean_b','mean_difference_a_minus_b','ci95_difference_lo','ci95_difference_hi','p_raw','p_holm','p_exact_permutation_sensitivity','t_statistic'):
        check('statistical_arithmetic',equal(r[k],saved[k]),(r['label'],k,r[k],saved[k]))
    check('statistical_decision',r['significant_holm_005']==saved['significant_holm_005'] and r['family']==saved['family'],r['label'])
    r['interpretation_scope']='Explorative E2-Testfamilie, keine Präregistrierung; marginale 95%-Intervalle.' if r['output']=='example2_exploratory' else 'Festgelegte Vergleichsfamilie; marginale 95%-Intervalle; Nichtsignifikanz ist keine Äquivalenz.'
check('statistical_coverage',len(tests)==40 and len(generated['example3'])==22,len(tests))
csvout('statistik_40.csv',tests);emit('statistik_40.json',tests)

controls=[]
for right in ('pisco_static_pca_fixed_cycles','field_no_physics_matched','scalar_surrogate','no_periodic_verification'):
    for seed in range(7,16):
        a,b=run('example2','pisco_dynamic_pca_fixed_cycles',seed),run('example2',right,seed)
        ca,cb=a['extra']['r2_evidence']['configuration']['config'],b['extra']['r2_evidence']['configuration']['config']
        same=anchors(a)==anchors(b) and a['fem_calls_total']==b['fem_calls_total']
        check('e2_matched_controls',same, (right,seed))
        controls.append({'right':right,'seed':seed,'same_initial_and_budget':same,
                         'configuration_differences':{k:[v,cb.get(k)] for k,v in ca.items() if v!=cb.get(k)},
                         'updates_left':len(a['extra']['r2_evidence']['training']),
                         'updates_right':len(b['extra']['r2_evidence']['training'])})
emit('kontrollvergleiche.json',controls)

print('5/6: Current evidence and interpretation limits',flush=True)
check('source_unchanged',source_before=={name:sha(R2/'code'/name) for name in source_before},source_before)
e2_binary=[r['coarse'] for r in meshrows if r['method']=='pisco_binary']
e2_simp=next(r for r in meshrows if r['method']=='simp')
overview={'binary_pisco_coarse':summary(e2_binary),
          'binary_simp_coarse':e2_simp['coarse'],
          'pisco_relative_to_simp_percent':100*(np.mean(e2_binary)/e2_simp['coarse']-1),
          'binary_pisco_fine':summary([r['fine'] for r in meshrows if r['method']=='pisco_binary']),
          'binary_simp_fine':e2_simp['fine'],
          'pisco_mean_mesh_change_percent':float(np.mean([r['change_percent'] for r in meshrows if r['method']=='pisco_binary'])),
          'simp_mesh_change_percent':e2_simp['change_percent']}
emit('kernergebnisse.json',{'example2':overview,'methods':aggregate})
result={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Verification of saved current numerical evidence, budgets and statistics; no FEM, training or optimisation.',
        'passed':not FAILURES,'checks_by_category':dict(COUNTS),'failures':FAILURES,
        'artifact_count':len(manifest),'experiment_configurations':len(groups),'result_rows':len(index),
        'completed_solver_runs':len(index)-6,'reused_sensitivity_rows':6,
        'current_numeric_summaries':len(ledger),'statistical_contrasts':len(tests),
        'test_sensitive_comparisons':[{'label':r['label'],'p_holm':r['p_holm'],
                                      'p_exact_permutation_holm_sensitivity':r['p_exact_permutation_holm_sensitivity']}
                                     for r in tests if r['test_sensitive_decision']],
        'max_exclusive_time_account_gap_seconds':max_phase_gap,'source_sha256':source_before,
        'max_ellipse_area_relative_error':max(r['area_relative_error'] for r in designs),
        'ellipse_area_relative_tolerance':1e-6,'trajectory_limitations':trajectory_limits,
        'input_sha256':{str(p.relative_to(DOC)):sha(p) for p in [DOC/'artifact_manifest.json',DOC/'manuscript/claims.json',DOC/'manuscript/statistical_tests.json']},
        'limitations':[
            'Saved-data checks are not an independent rerun of the optimisation or FEM.',
            'Geometry diagnostics do not guarantee minimum strut thickness, manufacturability or continuum convergence.',
            'E2 learning residuals are pooled across representations; they cannot be separated from the saved aggregate.',
            'Confidence intervals are marginal; Holm correction is within the stated families; E2 tests are exploratory. Non-significance is not equivalence.',
            'E3 ARD-BO at d=35 and the physics ablation have test-sensitive decisions; report primary and exact-permutation sensitivity results together.',
            'Equal FEM budgets do not imply equal runtime. E1 DE/CMA checkpoint times belong to the full 160-evaluation run; E2 binary comparisons do not have identical budgets for every seed.',
            'SIMP trajectory omits the last FEM evaluation; the multistart trajectory represents the best start. Complete FEM events are used for cumulative curves.',
            'PCA projection diagnostics include a direct SIMP reference and are not optimisation trajectories.',
            'Numerical results can depend on the architecture and numerical-library builds.'
        ]}
emit('verification.json',result)
print('6/6:',json.dumps({k:result[k] for k in ('passed','artifact_count','result_rows','current_numeric_summaries','statistical_contrasts')}),flush=True)
if FAILURES:
    print(json.dumps(FAILURES[:15],ensure_ascii=False,indent=2),flush=True)
    sys.exit(1)
