#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Andreas Arnold, Arnold Invest GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Reproduce the numerical results and publication graphics of the paper.

The numerical implementation consists of this driver and ``solvers.py``.
Editable diagram templates are distributed in ``paper_support``.  Numerical algorithms are imported directly
from ``solvers.py``; no archive is extracted and no hidden helper script is
executed.  ``PublicationBuilder`` owns statistics, tables, plots and the
machine-readable verification manifest.

Typical commands
----------------
``python reproduce_all.py --list``
    Show the available benchmark families.
``python reproduce_all.py --all``
    Run all publication seeds, resume completed runs, then build and verify
    every result table and figure.
``python reproduce_all.py --example example1 --seeds 7``
    Run a short, explicitly labelled one-seed reproduction.
``python reproduce_all.py --build``
    Rebuild tables and figures from saved results without rerunning a solver.
``python reproduce_all.py --verify``
    Require the complete R2 experiment matrix, provenance and artifact hashes.
``python reproduce_all.py --verify-frozen``
    Check historical R1 reference arithmetic only.
``python reproduce_all.py --submission-assets``
    Extract the manuscript tables and verify the journal-upload figures
    without running a numerical experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
from typing import Any, Callable

# Worker count controls fixed packets, never native reduction trees.
for _native_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_native_variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np
from types import SimpleNamespace

import solvers
from solvers import PublicationExperimentSuite, SolverResult


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "reproduction_output_r2"
RESULT_BUNDLE = "publication_results.pkl"

# Aggregates frozen when the major-revision results were accepted into the
# manuscript.  Tolerances cover harmless floating-point/library differences;
# they do not conceal statistically relevant changes.
REFERENCE_MEANS: dict[str, tuple[float, float]] = {
    "example1/pisco": (1.0515027321212986, 5e-4),
    "example1/de": (1.0507375974433775, 5e-4),
    "example1/bo": (1.0549367000183065, 1e-3),
    "example1/cmaes": (1.050716049417423, 5e-4),
    "example1/pisco_lambda_0": (1.0524313337089812, 1e-3),
    "example1/pisco_lambda_0.001": (1.0519462593403486, 1e-3),
    "example1/pisco_lambda_0.1": (1.0516931030687504, 1e-3),
    "example1/pisco_fixed_156": (1.0513261530614466, 5e-4),
    "example1/de_fixed_156": (1.0524337668779438, 1e-3),
    "example1/bo_fixed_156": (1.053390394488088, 1e-3),
    "example1/cmaes_fixed_156": (1.0503445978903463, 5e-4),
    "example2/pisco_dynamic_pca": (84.44458658275254, 0.10),
    "example2/pisco_static_pca": (84.69522597880717, 0.15),
    "example2/pisco_vae": (85.61154396609865, 0.30),
    "example2/pisco_ae": (86.99265014992635, 0.15),
    "example2/pisco_binary": (71.0459737942775, 0.10),
    "example2/pisco_strong_anchors": (69.94667498958557, 0.10),
    "example2/pisco_dynamic_pca_fixed_cycles": (84.43779645041816, 0.15),
    "example2/pisco_static_pca_fixed_cycles": (84.44643226820644, 0.20),
    "example2/pisco_vae_lambda_0": (85.9247252812541, 0.30),
    "example2/pisco_vae_lambda_1": (85.67567039943681, 0.30),
    "example2/pisco_vae_data_lambda_0": (86.99265014992635, 0.20),
    "example2/scalar_surrogate": (86.99265014992635, 0.20),
    "example2/no_periodic_verification": (86.26503186248904, 0.35),
    "example2/latent_fem": (86.82386599775204, 0.25),
    "example2/latent_cmaes": (70.8247643023513, 0.15),
    "example2/latent_ard_bo": (71.01343045828929, 0.15),
    "example2/pisco_random_basis": (452.8169821520629, 0.50),
    "example3/pisco_d35": (43.446108446679126, 0.10),
    "example3/pisco_d62": (45.42373514364277, 0.10),
    "example3/pisco_d135": (45.01062874828395, 0.10),
    "example3/pisco_d243": (45.48652118905224, 0.10),
    "example3/ard_bo_d35": (37.966971029861504, 0.10),
    "example3/de_d35": (62.58288658717765, 0.10),
    "example3/random_d35": (63.37274015959054, 0.10),
    "example3/cmaes_d35": (47.967538804738686, 0.10),
    "example3/ard_bo_d62": (33.58704310210879, 0.10),
    "example3/de_d62": (64.0066591945704, 0.10),
    "example3/random_d62": (64.79299092443645, 0.10),
    "example3/cmaes_d62": (49.27485496778314, 0.10),
    "example3/ard_bo_d135": (36.40637355440074, 0.10),
    "example3/de_d135": (67.27113989109394, 0.10),
    "example3/random_d135": (70.99178479181988, 0.10),
    "example3/cmaes_d135": (51.51766194287216, 0.10),
    "example3/ard_bo_d243": (42.2386171126338, 0.10),
    "example3/de_d243": (72.96291809652933, 0.10),
    "example3/random_d243": (71.6134908989863, 0.10),
    "example3/cmaes_d243": (53.90757740315212, 0.10),
    "example3/scalar_surrogate_d35": (47.976782054564305, 0.10),
    "example3/pisco_no_physics_d35": (41.790347683384, 0.10),
    "example3/pisco_progressive": (38.26630443842234, 0.10),
    "example3/ard_bo_progressive": (43.23659639707223, 0.10),
    "example3/pisco_starts_6_d35": (43.65771280439418, 0.10),
    "example3/pisco_starts_12_d35": (44.27643715923299, 0.10),
    "example3/pisco_starts_24_d35": (46.06656972140524, 0.10),
    "example3/pisco_top_3_d35": (42.047372213630695, 0.10),
    "example3/pisco_top_6_d35": (44.27643715923299, 0.10),
    "example3/pisco_top_12_d35": (44.990531271342256, 0.10),
}

THREE_SEED_REFERENCE_METHODS = frozenset({
    "pisco_starts_6_d35", "pisco_starts_12_d35",
    "pisco_starts_24_d35", "pisco_top_3_d35", "pisco_top_6_d35",
    "pisco_top_12_d35",
})


# The complete frozen reviewer-facing record remains plain JSON text so it
# can be inspected without executing code.  It is reference data, not an
# embedded source bundle.  Verification recomputes all aggregate fields and
# statistical comparisons from the per-seed values below.
FROZEN_CLAIM_ROWS = json.loads(r'''[{"budget_fem":[108.0,108.0,132.0,140.0,132.0,124.0,108.0,132.0,108.0],"ci95_mean_hi":1.0522235382390839,"ci95_mean_lo":1.0507819260035134,"example":"E1","id":"e1_pisco","iqr":0.0012121105744631322,"mean":1.0515027321212986,"median":1.0512664290887568,"method":"PISCO","n":9,"per_seed_values":[1.0520095433970549,1.0508427470662036,1.0536219409697885,1.0507974328225917,1.0512664290887568,1.0507974328225917,1.0513577817271298,1.052033848374978,1.0507974328225917],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.000937733944334018},{"budget_fem":[210.0,250.0,340.0,255.0,355.0,260.0,260.0,230.0,225.0],"ci95_mean_hi":1.0517175393042935,"ci95_mean_lo":1.0497576555824615,"example":"E1","id":"e1_de","iqr":0.0017618606602995968,"mean":1.0507375974433775,"median":1.050449408150209,"method":"DE","n":9,"per_seed_values":[1.0500371206941495,1.049061489947192,1.050208237346607,1.0515027515742172,1.050449408150209,1.051798981354449,1.0520276605369092,1.0490501627638908,1.052502564622773],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0012748570299018435},{"budget_fem":[75.0,71.0,71.0,71.0,75.0,71.0,79.0,71.0,71.0],"ci95_mean_hi":1.0572719432161322,"ci95_mean_lo":1.052601456820481,"example":"E1","id":"e1_bo","iqr":0.002627622258021045,"mean":1.0549367000183065,"median":1.0550946097808755,"method":"BO","n":9,"per_seed_values":[1.0499473989936836,1.0548527584950678,1.0582987617903024,1.0545279008240447,1.0550946097808755,1.0571085440101364,1.0501973042399204,1.0572474989486624,1.0571555230820657],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.003038038608224925},{"budget_fem":[152.0,152.0,88.0,80.0,144.0,112.0,88.0,120.0,152.0],"ci95_mean_hi":1.0515325416616548,"ci95_mean_lo":1.0498995571731913,"example":"E1","id":"e1_cmaes","iqr":0.0005480230351644355,"mean":1.050716049417423,"median":1.0506010623995092,"method":"CMA-ES","n":9,"per_seed_values":[1.0492940992190556,1.049346675417245,1.0509652169439927,1.0504171939088283,1.0508638375257142,1.0505833578728259,1.0506010623995092,1.0516384185985905,1.052734582871045],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0010622169732052574},{"budget_fem":[116.0,132.0,108.0,108.0,108.0,108.0,108.0,108.0,108.0],"ci95_mean_hi":1.053438697870114,"ci95_mean_lo":1.0514239695478484,"example":"E1","id":"e1_pisco_lam0","iqr":0.0020285709663059315,"mean":1.0524313337089812,"median":1.0526516918427324,"method":"PISCO lambda=0","n":9,"per_seed_values":[1.0508427470662036,1.0515730755498538,1.0542504977196399,1.0536775611310527,1.0533863526934357,1.0533448628281914,1.0513577817271298,1.0526516918427324,1.0507974328225917],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0013105321179879544},{"budget_fem":[124.0,108.0,108.0,108.0,108.0,124.0,108.0,132.0,108.0],"ci95_mean_hi":1.0530416887725584,"ci95_mean_lo":1.0508508299081387,"example":"E1","id":"e1_pisco_lam0_001","iqr":0.002543605627232104,"mean":1.0519462593403486,"median":1.0513577817271298,"method":"PISCO lambda=0.001","n":9,"per_seed_values":[1.0515092743894474,1.0508427470662036,1.0508427470662036,1.0536775611310527,1.0533863526934357,1.0507974328225917,1.0513577817271298,1.0543050043444817,1.0507974328225917],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0014251007821105854},{"budget_fem":[108.0,148.0,116.0,140.0,132.0,156.0,108.0,156.0,108.0],"ci95_mean_hi":1.0525476094877262,"ci95_mean_lo":1.0508385966497746,"example":"E1","id":"e1_pisco_lam0_1","iqr":0.001609188417190488,"mean":1.0516931030687504,"median":1.0513577817271298,"method":"PISCO lambda=0.1","n":9,"per_seed_values":[1.0526021655301518,1.0506792514057486,1.0524066212397822,1.0507974328225917,1.0507974328225917,1.0518773451652712,1.0513577817271298,1.0539224640828955,1.0507974328225917],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0011116715784642564},{"budget_fem":[292,286,287,303,310,302,297,300,323],"ci95_mean_hi":84.71839531750878,"ci95_mean_lo":84.17077784799629,"example":"E2","id":"e2_pisco_pca","iqr":0.5545888675658546,"mean":84.44458658275254,"median":84.4354876893913,"method":"PISCO adaptive PCA","n":9,"per_seed_values":[84.14020436893331,84.4354876893913,84.82891591141319,84.4314356004133,84.10318350983974,83.83743332253499,84.86852259196161,84.66130301378615,84.69479323649917],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3562119389677602},{"budget_fem":[274,250,299,291,274,272,315,300,293],"ci95_mean_hi":84.94715539278654,"ci95_mean_lo":84.4432965648278,"example":"E2","id":"e2_pisco_pca_static","iqr":0.47508782920009196,"mean":84.69522597880717,"median":84.62054042873237,"method":"PISCO static PCA","n":9,"per_seed_values":[84.18726426744172,84.9641668915915,84.61398673033926,84.62054042873237,84.48907906239141,85.13558627262495,84.44090796597135,84.648199324011,85.15730286616102],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3277479994072998},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":84.7079034832624,"ci95_mean_lo":84.16768941757392,"example":"E2","id":"e2_pisco_pca_fixed_budget","iqr":0.5210986448528416,"mean":84.43779645041816,"median":84.4354876893913,"method":"PISCO adaptive PCA fixed T","n":9,"per_seed_values":[84.14020436893331,84.4354876893913,84.82891591141319,84.4314356004133,84.10318350983974,83.83743332253499,84.86852259196161,84.66130301378615,84.63368204548985],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3513962035722743},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":84.56784357752362,"ci95_mean_lo":84.32502095888925,"example":"E2","id":"e2_pisco_pca_static_fixed_budget","iqr":0.1716276879522809,"mean":84.44643226820644,"median":84.44090796597135,"method":"PISCO static PCA fixed T","n":9,"per_seed_values":[84.18726426744172,84.43547849103277,84.61398673033926,84.62054042873237,84.22328394929232,84.39929526194962,84.44090796597135,84.52621036919675,84.5709229499019],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.15795024926062612},{"budget_fem":[262,280,269,267,310,278,297,288,329],"ci95_mean_hi":86.2449541831826,"ci95_mean_lo":84.9781337490147,"example":"E2","id":"e2_pisco_vae","iqr":1.187300353344554,"mean":85.61154396609865,"median":85.72976281017729,"method":"PISCO VAE","n":9,"per_seed_values":[85.02377327627218,84.73691926521914,86.54410204132195,86.91622798873777,84.64942092019663,86.09662742855606,85.8977348891953,85.72976281017729,84.90932707521151],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.8240360987399534},{"budget_fem":[262,250,269,267,274,272,279,270,293],"ci95_mean_hi":87.07497426187076,"ci95_mean_lo":86.91032603798193,"example":"E2","id":"e2_pisco_ae","iqr":0.1668332707267126,"mean":86.99265014992635,"median":87.00130426491964,"method":"PISCO AE","n":9,"per_seed_values":[87.18432461241864,87.00130426491964,87.05836889249024,86.91622798873777,86.89153562176352,86.88527060296789,86.87545223070808,87.05607366363873,87.06529347169266],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.1070996933886186},{"budget_fem":[286,268,269,285,280,272,279,270,293],"ci95_mean_hi":86.59631259194714,"ci95_mean_lo":85.25313797056106,"example":"E2","id":"e2_pisco_vae_nophys","iqr":1.8261668587181106,"mean":85.9247252812541,"median":85.98474021210731,"method":"PISCO VAE lambda=0","n":9,"per_seed_values":[84.84463846821153,84.91722344886864,86.81768763715473,85.98474021210731,84.99152077843662,86.88527060296789,86.02901335059425,85.78713956125306,87.06529347169266],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.8737026535734065},{"budget_fem":[262,250,269,267,298,272,297,294,299],"ci95_mean_hi":86.25689650041681,"ci95_mean_lo":85.0944442984568,"example":"E2","id":"e2_pisco_vae_strong","iqr":1.0467617390930428,"mean":85.67567039943681,"median":85.94738503222248,"method":"PISCO VAE lambda=1","n":9,"per_seed_values":[84.82354429900346,86.29238943482102,86.93122500314531,85.96185137478855,84.64102656222542,85.60616090822265,85.96210635979769,85.94738503222248,84.91534462070464],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.7561470841792802},{"budget_fem":[262,250,269,267,274,272,279,270,293],"ci95_mean_hi":87.07497426187076,"ci95_mean_lo":86.91032603798193,"example":"E2","id":"e2_pisco_vae_purephys","iqr":0.1668332707267126,"mean":86.99265014992635,"median":87.00130426491964,"method":"PISCO VAE data lambda=0","n":9,"per_seed_values":[87.18432461241864,87.00130426491964,87.05836889249024,86.91622798873777,86.89153562176352,86.88527060296789,86.87545223070808,87.05607366363873,87.06529347169266],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.1070996933886186},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":87.07497426187076,"ci95_mean_lo":86.91032603798193,"example":"E2","id":"e2_scalar_surrogate","iqr":0.1668332707267126,"mean":86.99265014992635,"median":87.00130426491964,"method":"scalar surrogate","n":9,"per_seed_values":[87.18432461241864,87.00130426491964,87.05836889249024,86.91622798873777,86.89153562176352,86.88527060296789,86.87545223070808,87.05607366363873,87.06529347169266],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.1070996933886186},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":86.93542146493651,"ci95_mean_lo":85.59464226004157,"example":"E2","id":"e2_no_reeval","iqr":1.2438835193548243,"mean":86.26503186248904,"median":86.87545223070808,"method":"fixed surrogate","n":9,"per_seed_values":[85.95925336843341,85.6476521024087,84.94825893804133,85.05649676274706,86.89153562176352,86.88527060296789,86.87545223070808,87.05607366363873,87.06529347169266],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.8721444929951783},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":87.13864209985786,"ci95_mean_lo":86.50908989564621,"example":"E2","id":"e2_latent_fem","iqr":0.19907034231607668,"mean":86.82386599775204,"median":86.88527060296789,"method":"latent FEM search","n":9,"per_seed_values":[87.18432461241864,87.00130426491964,87.05836889249024,86.696619275596,86.85700332132265,86.88527060296789,86.87545223070808,87.05607366363873,85.80037711570645],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":0.4095085051674828},{"budget_fem":[152.0,134.0,152.0,134.0,134.0,134.0,152.0,134.0,152.0],"ci95_mean_hi":465.03993873584193,"ci95_mean_lo":440.5940255682839,"example":"E2","id":"e2_simp_free_init","iqr":25.480279550872524,"mean":452.8169821520629,"median":456.2388191881769,"method":"SIMP-free anchors","n":9,"per_seed_values":[424.87462860856806,464.86543097002937,438.5063894593526,468.237235513671,458.5835042654954,473.4649144239571,451.19676552015846,456.2388191881769,439.38515141915684],"role":"ablation","seeds":[7,8,9,10,11,12,13,14,15],"std":15.90147614721882},{"budget_fem":156,"ci95_mean_hi":1.051965732758534,"ci95_mean_lo":1.0506865733643593,"example":"E1","id":"e1_fixed156_pisco","iqr":0.0005603489045380439,"mean":1.0513261530614466,"median":1.0508427470662036,"method":"PISCO","n":9,"per_seed_values":[1.0507974328225917,1.0508427470662036,1.053244840005586,1.0507974328225917,1.0512664290887568,1.0507974328225917,1.0513577817271298,1.052033848374978,1.0507974328225917],"role":"fixed_budget_control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0008320622942385437},{"budget_fem":156,"ci95_mean_hi":1.053926527846042,"ci95_mean_lo":1.0509410059098456,"example":"E1","id":"e1_fixed156_de","iqr":0.0028117648446166488,"mean":1.0524337668779438,"median":1.0525522809674202,"method":"DE","n":9,"per_seed_values":[1.0500433907302484,1.051224931255223,1.0553871333526093,1.0515027515981976,1.0543756517668936,1.0540366960998397,1.0529519198463815,1.0498291462846803,1.0525522809674202],"role":"fixed_budget_control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0019420099192050183},{"budget_fem":156,"ci95_mean_hi":1.0548966644599531,"ci95_mean_lo":1.051884124516223,"example":"E1","id":"e1_fixed156_bo","iqr":0.0013679171769169773,"mean":1.053390394488088,"median":1.0541644978469111,"method":"BO","n":9,"per_seed_values":[1.0499473989936836,1.0548527584950678,1.0531599836471277,1.0545279008240447,1.0550946097808755,1.0541644978469111,1.0501973042399204,1.054486539973631,1.0540825565915317],"role":"fixed_budget_control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0019595844806217025},{"budget_fem":156,"ci95_mean_hi":1.051289704886241,"ci95_mean_lo":1.0493994908944515,"example":"E1","id":"e1_fixed156_cmaes","iqr":0.0016006154646277526,"mean":1.0503445978903463,"median":1.0498611697850182,"method":"CMA-ES","n":9,"per_seed_values":[1.0492940992190556,1.049346675417245,1.0509472908818727,1.049099833648856,1.0508638375257142,1.0498611697850182,1.0494472459637525,1.0516384185985905,1.0526028099730123],"role":"fixed_budget_control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0012295385530317986},{"budget_fem":[298,280,305,303,292,290,309,306,317],"ci95_mean_hi":71.2879949727821,"ci95_mean_lo":70.8039526157729,"example":"E2","id":"e2_pisco_binary_objective","iqr":0.5149358051141064,"mean":71.0459737942775,"median":71.1599611657006,"method":"PISCO binary objective","n":9,"per_seed_values":[70.63695012871648,70.77457700647825,71.1599611657006,71.16769829071809,71.00140280330888,71.43774698466933,71.28951281159236,71.35952598577302,70.58638897154046],"role":"main","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3148578636219591},{"budget_fem":[840,828,847,845,852,850,857,848,871],"ci95_mean_hi":69.99450931598886,"ci95_mean_lo":69.89884066318228,"example":"E2","id":"e2_pisco_strong_anchors","iqr":0.06825794149142439,"mean":69.94667498958557,"median":69.97919822898484,"method":"PISCO strong anchors","n":9,"per_seed_values":[69.82615631715488,69.91085606965979,70.03805338718453,69.9826452273273,69.91382893049558,69.97919822898484,69.91094028749342,69.97919822898484,69.97919822898484],"role":"control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.06223014825476504},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":70.9526043537136,"ci95_mean_lo":70.696924250989,"example":"E2","id":"e2_latent_cma_es","iqr":0.2038179694847031,"mean":70.8247643023513,"median":70.80655925837895,"method":"latent CMA-ES","n":9,"per_seed_values":[70.90547183054912,70.80655925837895,70.95875890903027,70.80220126060219,70.62112347216018,70.88838562958091,71.12560616710601,70.61311833268954,70.70165386106441],"role":"control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.16631373215336814},{"budget_fem":[298,286,305,303,310,308,315,306,329],"ci95_mean_hi":71.18176305206629,"ci95_mean_lo":70.8450978645123,"example":"E2","id":"e2_latent_ard_bo","iqr":0.18403999663479453,"mean":71.01343045828929,"median":70.95260308716013,"method":"latent ARD-BO","n":9,"per_seed_values":[70.87352227279297,70.95260308716013,71.05756226942776,71.12444180175157,71.04429554835282,70.77494542368957,71.51732626098513,70.84963920327286,70.92653825717088],"role":"control","seeds":[7,8,9,10,11,12,13,14,15],"std":0.21899257404682237},{"budget_fem":null,"ci95_mean_hi":0.44999999966803916,"ci95_mean_lo":0.44999996889674787,"example":"E2","id":"e2_design_volume_fraction","iqr":1.7869784485480267e-08,"mean":0.4499999842823935,"median":0.4499999873903619,"method":"PISCO final volume fraction","n":9,"per_seed_values":[0.4499999873903619,0.4500000193897708,0.4499999789193048,0.4499999946415038,0.4499999505729729,0.44999996416912486,0.4499999990605903,0.4499999767717193,0.44999998762619264],"role":"design_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":2.001598183134923e-08},{"budget_fem":null,"ci95_mean_hi":0.2804441783112271,"ci95_mean_lo":0.27139504739204295,"example":"E2","id":"e2_design_grayness","iqr":0.008025294076910994,"mean":0.275919612851635,"median":0.2736505768687089,"method":"PISCO final grayness","n":9,"per_seed_values":[0.2700045894974341,0.27949331552102874,0.27146802144411775,0.28438414120129885,0.2724378902883637,0.2697190497395728,0.28489383953896075,0.2772250915652299,0.2736505768687089],"role":"design_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":0.005886241126612021},{"budget_fem":null,"ci95_mean_hi":5.0,"ci95_mean_lo":5.0,"example":"E2","id":"e2_design_min_feature_size_elements","iqr":0.0,"mean":5.0,"median":5.0,"method":"PISCO minimum feature size","n":9,"per_seed_values":[5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0],"role":"design_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":0.0},{"budget_fem":null,"ci95_mean_hi":84.71839531750878,"ci95_mean_lo":84.17077784799629,"example":"E2","id":"e2_design_common_filtered_compliance","iqr":0.5545888675658546,"mean":84.44458658275254,"median":84.4354876893913,"method":"PISCO filtered compliance","n":9,"per_seed_values":[84.14020436893331,84.4354876893913,84.82891591141319,84.4314356004133,84.10318350983974,83.83743332253499,84.86852259196161,84.66130301378615,84.69479323649917],"role":"design_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3562119389677602},{"budget_fem":null,"ci95_mean_hi":71.94280640099228,"ci95_mean_lo":71.4160310606357,"example":"E2","id":"e2_design_volume_preserving_binary_compliance","iqr":0.38036871197671474,"mean":71.67941873081399,"median":71.600345502743,"method":"PISCO volume-preserving binary compliance","n":9,"per_seed_values":[71.56558898499364,71.600345502743,72.03243846019438,71.58131778755497,71.1639945315784,71.20675639745167,71.94595769697035,72.0733532076763,71.94501600816318],"role":"design_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3426546372888028},{"budget_fem":null,"ci95_mean_hi":88.92279750042887,"ci95_mean_lo":88.2462210429395,"example":"E2","id":"e2_fine_mesh_pisco","iqr":0.6647836033339161,"mean":88.58450927168418,"median":88.56403574763824,"method":"PISCO fine-mesh compliance","n":9,"per_seed_values":[88.29320385720791,88.56403574763824,89.02267457516231,88.44208901983684,88.21662283423336,87.809122614524,89.1606118109325,88.95798746054183,88.79423552508064],"role":"mesh_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":0.4400966462899293},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_simp_design_volume_fraction","iqr":null,"mean":0.45,"median":0.45,"method":"SIMP final volume fraction","n":1,"per_seed_values":[0.45],"role":"deterministic_design_diagnostic","seeds":[0],"std":null,"value":0.45},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_simp_design_grayness","iqr":null,"mean":0.3027906096843308,"median":0.3027906096843308,"method":"SIMP final grayness","n":1,"per_seed_values":[0.3027906096843308],"role":"deterministic_design_diagnostic","seeds":[0],"std":null,"value":0.3027906096843308},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_simp_design_min_feature_size_elements","iqr":null,"mean":4.0,"median":4.0,"method":"SIMP minimum feature size","n":1,"per_seed_values":[4.0],"role":"deterministic_design_diagnostic","seeds":[0],"std":null,"value":4.0},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_simp_design_common_filtered_compliance","iqr":null,"mean":85.14576055323822,"median":85.14576055323822,"method":"SIMP filtered compliance","n":1,"per_seed_values":[85.14576055323822],"role":"deterministic_design_diagnostic","seeds":[0],"std":null,"value":85.14576055323822},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_simp_design_volume_preserving_binary_compliance","iqr":null,"mean":69.32671948818714,"median":69.32671948818714,"method":"SIMP volume-preserving binary compliance","n":1,"per_seed_values":[69.32671948818714],"role":"deterministic_design_diagnostic","seeds":[0],"std":null,"value":69.32671948818714},{"budget_fem":81,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_fine_mesh_simp","iqr":null,"mean":89.29538098881483,"median":89.29538098881483,"method":"SIMP fine-mesh compliance","n":1,"per_seed_values":[89.29538098881483],"role":"deterministic_mesh_diagnostic","seeds":[0],"std":null,"value":89.29538098881483},{"budget_fem":null,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_decoder_raw_rho_min_global","iqr":null,"mean":-0.6564170122146606,"median":-0.6564170122146606,"method":"PISCO decoder global minimum","n":1,"per_seed_values":[-0.6564170122146606],"role":"aggregate_decoder_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":null,"value":-0.6564170122146606},{"budget_fem":null,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_decoder_raw_rho_max_global","iqr":null,"mean":1.7037996053695679,"median":1.7037996053695679,"method":"PISCO decoder global maximum","n":1,"per_seed_values":[1.7037996053695679],"role":"aggregate_decoder_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":null,"value":1.7037996053695679},{"budget_fem":null,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_decoder_raw_clipped_fraction_mean","iqr":null,"mean":0.5767381376590727,"median":0.5767381376590727,"method":"PISCO decoder clipped fraction","n":1,"per_seed_values":[0.5767381376590727],"role":"aggregate_decoder_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":null,"value":0.5767381376590727},{"budget_fem":null,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_decoder_projected_volume_max_abs_error","iqr":null,"mean":1.311302185169616e-07,"median":1.311302185169616e-07,"method":"PISCO maximum projected-volume error","n":1,"per_seed_values":[1.311302185169616e-07],"role":"aggregate_decoder_diagnostic","seeds":[7,8,9,10,11,12,13,14,15],"std":null,"value":1.311302185169616e-07},{"budget_fem":226,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors40_iters6","iqr":null,"mean":70.494,"median":70.494,"method":"projected SIMP reference (anchors40_iters6)","n":1,"per_seed_values":[70.494],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":70.494},{"budget_fem":341,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors60_iters6","iqr":null,"mean":70.306,"median":70.306,"method":"projected SIMP reference (anchors60_iters6)","n":1,"per_seed_values":[70.306],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":70.306},{"budget_fem":459,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors80_iters6","iqr":null,"mean":69.686,"median":69.686,"method":"projected SIMP reference (anchors80_iters6)","n":1,"per_seed_values":[69.686],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":69.686},{"budget_fem":700,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors120_iters6","iqr":null,"mean":69.953,"median":69.953,"method":"projected SIMP reference (anchors120_iters6)","n":1,"per_seed_values":[69.953],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":69.953},{"budget_fem":466,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors40_iters12","iqr":null,"mean":69.591,"median":69.591,"method":"projected SIMP reference (anchors40_iters12)","n":1,"per_seed_values":[69.591],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":69.591},{"budget_fem":939,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors80_iters12","iqr":null,"mean":69.57,"median":69.57,"method":"projected SIMP reference (anchors80_iters12)","n":1,"per_seed_values":[69.57],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":69.57},{"budget_fem":786,"ci95_mean_hi":null,"ci95_mean_lo":null,"example":"E2","id":"e2_projected_simp_anchors40_iters20","iqr":null,"mean":69.412,"median":69.412,"method":"projected SIMP reference (anchors40_iters20)","n":1,"per_seed_values":[69.412],"role":"single_seed_representation_diagnostic","seeds":[7],"std":null,"value":69.412},{"budget_fem":[298,268,287,297,292,308,297,300,329],"ci95_mean_hi":1121.40363999478,"ci95_mean_lo":807.5795218856608,"example":"E2","id":"e2_pisco_wall_time","iqr":366.8757503879897,"mean":964.4915809402204,"median":1009.8769285510061,"method":"PISCO wall time","n":9,"per_seed_values":[1297.8345538600115,772.9884416240093,770.7791646920086,1022.489850146987,763.1875017329876,1142.34237153898,763.2705012359947,1009.8769285510061,1137.6549150799983],"role":"timing","seeds":[7,8,9,10,11,12,13,14,15],"std":204.13500998427364},{"budget_fem":null,"ci95_mean_hi":1097.3355959638554,"ci95_mean_lo":789.503677249909,"example":"E2","id":"e2_surrogate_training_s","iqr":362.8052719249972,"mean":943.4196366068822,"median":990.3535391589976,"method":"surrogate training time","n":9,"per_seed_values":[1265.4656542769226,755.4474715999386,752.1165009899996,1001.0227757790126,745.0434680239996,1121.1119968080602,745.2935499100131,990.3535391589976,1114.9217729149968],"role":"timing","seeds":[7,8,9,10,11,12,13,14,15],"std":200.2372289892005},{"budget_fem":null,"ci95_mean_hi":14.504999821762084,"ci95_mean_lo":9.49858315136165,"example":"E2","id":"e2_latent_optimisation_s","iqr":1.9112647629517596,"mean":12.001791486561867,"median":10.580846327968175,"method":"latent optimisation time","n":9,"per_seed_values":[20.121911239926703,10.41869651802699,10.302986240043538,12.594848105072742,10.486819401092362,12.131591066980036,9.04846319896751,10.580846327968175,12.32996128097875],"role":"timing","seeds":[7,8,9,10,11,12,13,14,15],"std":3.256553139240477},{"budget_fem":null,"ci95_mean_hi":1.9728795198792917,"ci95_mean_lo":1.508463289447241,"example":"E2","id":"e2_active_fem_s","iqr":0.4235447138780728,"mean":1.7406714046632663,"median":1.647495719982544,"method":"active FEM time","n":9,"per_seed_values":[2.2623316610697657,1.4140377990843263,1.5751328449114226,1.647495719982544,1.5057110601046588,1.9292557739827316,1.48977042795741,1.701721100922441,2.1405862539540976],"role":"timing","seeds":[7,8,9,10,11,12,13,14,15],"std":0.3020915422540643},{"budget_fem":81,"ci95_mean_hi":2.4915867376238126,"ci95_mean_lo":1.9269533607720841,"example":"E2","id":"e2_simp_wall_time","iqr":0.26127011395874433,"mean":2.2092700491979484,"median":2.174074278998887,"method":"SIMP wall time","n":5,"per_seed_values":[2.56370613600302,2.0166092830186244,2.174074278998887,2.2778793969773687,2.014081150991842],"role":"timing","seeds":[0,1,2,3,4],"std":0.22736963368050758},{"budget_fem":null,"ci95_mean_hi":0.16710909729326745,"ci95_mean_lo":-0.025440101160726453,"example":"E2","id":"e2_cycle1_holdout_rel_error_mean","iqr":0.03734266342989265,"mean":0.0708344980662705,"median":0.08280852518283323,"method":"cycle 1 holdout relative error","n":3,"per_seed_values":[0.08280852518283323,0.10219014793788178,0.027504821078096476],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.03875574158856178},{"budget_fem":null,"ci95_mean_hi":1.0252440344873799,"ci95_mean_lo":0.9092490165896666,"example":"E2","id":"e2_cycle1_holdout_spearman","iqr":0.021002632995140047,"mean":0.9672465255385232,"median":0.9790209790209792,"method":"cycle 1 holdout Spearman correlation","n":3,"per_seed_values":[0.9790209790209792,0.9403566658021552,0.9823619317924353],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.023347139200262528},{"budget_fem":null,"ci95_mean_hi":1.220176848649964,"ci95_mean_lo":0.6464898180167025,"example":"E2","id":"e2_cycle1_holdout_top_k_overlap","iqr":0.09999999999999998,"mean":0.9333333333333332,"median":1.0,"method":"cycle 1 holdout top-k overlap","n":3,"per_seed_values":[0.8,1.0,1.0],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.11547005383792514},{"budget_fem":null,"ci95_mean_hi":23.024660123793694,"ci95_mean_lo":6.406346943568767,"example":"E2","id":"e2_cycle1_holdout_equilibrium_residual_mean","iqr":3.175929029782612,"mean":14.71550353368123,"median":13.503488779067993,"method":"cycle 1 holdout relative equilibrium residual","n":3,"per_seed_values":[13.503488779067993,18.497439940770466,12.14558188120524],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":3.3448856521959125},{"budget_fem":null,"ci95_mean_hi":0.018371707281406628,"ci95_mean_lo":0.007637572524135784,"example":"E2","id":"e2_cycle12_holdout_rel_error_mean","iqr":0.0019347742358550232,"mean":0.013004639902771205,"median":0.011894347616209814,"method":"cycle 12 holdout relative error","n":3,"per_seed_values":[0.011625011810196878,0.015494560281906924,0.011894347616209814],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.0021605353653496814},{"budget_fem":null,"ci95_mean_hi":1.0237109902223902,"ci95_mean_lo":0.9016969351855353,"example":"E2","id":"e2_cycle12_holdout_spearman","iqr":0.02447552447552459,"mean":0.9627039627039627,"median":0.965034965034965,"method":"cycle 12 holdout Spearman correlation","n":3,"per_seed_values":[0.9860139860139862,0.9370629370629372,0.965034965034965],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.024558633456533186},{"budget_fem":null,"ci95_mean_hi":1.220176848649964,"ci95_mean_lo":0.6464898180167025,"example":"E2","id":"e2_cycle12_holdout_top_k_overlap","iqr":0.09999999999999998,"mean":0.9333333333333332,"median":1.0,"method":"cycle 12 holdout top-k overlap","n":3,"per_seed_values":[0.8,1.0,1.0],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":0.11547005383792514},{"budget_fem":null,"ci95_mean_hi":17.96044140067568,"ci95_mean_lo":0.4782960712227151,"example":"E2","id":"e2_cycle12_holdout_equilibrium_residual_mean","iqr":3.481227219104767,"mean":9.219368735949198,"median":8.627540270487467,"method":"cycle 12 holdout relative equilibrium residual","n":3,"per_seed_values":[6.034055749575297,12.99651018778483,8.627540270487467],"role":"surrogate_diagnostic","seeds":[7,12,15],"std":3.5187552700399607},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":45.60644787975771,"ci95_mean_lo":41.28576901360054,"example":"E3","id":"e3_pisco_d35","iqr":2.5819373771258043,"mean":43.446108446679126,"median":43.63174290845342,"method":"PISCO d=35","n":9,"per_seed_values":[41.6166171120679,49.654530177517906,41.55816418811315,39.534424923017184,44.198554489193704,44.25709703043376,42.533001174855926,43.63174290845342,44.03084401645924],"role":"main","seeds":[0,1,2,3,4,5,6,7,8],"std":2.810497258133474},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":38.547818930054596,"ci95_mean_lo":37.386123117309786,"example":"E3","id":"e3_sklearn_bo_d35","iqr":0.7464719373908011,"mean":37.96697102368219,"median":37.9926674037852,"method":"ARD-BO d=35","n":9,"per_seed_values":[37.570021977271225,39.67243447258494,37.21905638883408,37.309533651361114,38.134470301385704,38.085889006810426,38.33066764711216,37.9926674037852,37.3879983639949],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":0.7556550712615899},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":64.54360547238154,"ci95_mean_lo":60.62216801484228,"example":"E3","id":"e3_de_d35","iqr":2.2651393468322922,"mean":62.58288674361191,"median":62.94062586195334,"method":"DE d=35","n":9,"per_seed_values":[62.14876182010686,59.08757795081785,65.18911235805906,65.37721146482889,62.765022067473254,62.94062586195334,58.00180111675918,63.321966885569545,64.41390116693916],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.550800363498961},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":65.70903601835047,"ci95_mean_lo":61.03644429073038,"example":"E3","id":"e3_random_d35","iqr":5.016761666262333,"mean":63.37274015454042,"median":63.141796371325135,"method":"Random search d=35","n":9,"per_seed_values":[67.94078906466329,62.9809868100478,63.141796371325135,60.29100736812269,65.6922347521277,58.75546093384277,65.61037979154617,65.34838817390443,60.59361812528383],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":3.0394080758270587},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":49.531377358913964,"ci95_mean_lo":46.4036998531708,"example":"E3","id":"e3_cmaes_d35","iqr":2.2116984520053506,"mean":47.96753860604238,"median":48.41953112064238,"method":"CMA-ES d=35","n":9,"per_seed_values":[44.054930018655334,46.45999029100099,48.57002188938223,48.67168874300634,48.382934282481095,50.36596317223528,50.41428902136613,48.41953112064238,46.36849891561168],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.034478684141376},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":53.60987288198456,"ci95_mean_lo":37.237596705394495,"example":"E3","id":"e3_pisco_d62","iqr":13.052959738646202,"mean":45.423734793689526,"median":43.83992871674846,"method":"PISCO d=62","n":9,"per_seed_values":[33.76247891387416,43.83992871674846,31.254340974645487,57.46712231174767,52.33816217439892,62.794324495507766,39.285202435752716,48.03308037017759,40.03897275035296],"role":"main","seeds":[0,1,2,3,4,5,6,7,8],"std":10.649770262754009},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":34.36643694185654,"ci95_mean_lo":32.80764934444607,"example":"E3","id":"e3_bo_ard_d62","iqr":1.0452896065814912,"mean":33.587043143151305,"median":33.38484460759153,"method":"ARD-BO d=62","n":9,"per_seed_values":[33.38484460759153,35.07542249931986,32.31832918257232,33.35382830204713,34.174692931750116,35.05079339335827,32.324519424594996,33.47155462195891,33.129403325168624],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":1.0139536874285335},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":51.27019104268433,"ci95_mean_lo":47.27951896569209,"example":"E3","id":"e3_cmaes_d62","iqr":2.241238973595813,"mean":49.27485500418821,"median":49.35671327109352,"method":"CMA-ES d=62","n":9,"per_seed_values":[48.11222288417876,53.41955760661959,49.3650927207324,49.35671327109352,52.686084923745426,47.929917675043626,45.119727556066636,50.17115664863944,47.31322175157453],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.5958358114385547},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":67.05839240733651,"ci95_mean_lo":60.95492589845687,"example":"E3","id":"e3_de_d62","iqr":3.65857208374905,"mean":64.00665915289669,"median":64.50930726129809,"method":"DE d=62","n":9,"per_seed_values":[64.04650213314571,61.71411653223156,60.630033496807116,69.88176476662608,67.99735354094031,65.3726886159806,65.32676110444055,64.50930726129809,56.581404924600214],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":3.970157565441176},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":68.50525592800348,"ci95_mean_lo":61.08072648132525,"example":"E3","id":"e3_random_d62","iqr":3.83787803971331,"mean":64.79299120466436,"median":64.51722355101795,"method":"Random search d=62","n":9,"per_seed_values":[63.04728982452615,55.14974700334743,63.005507268956336,64.51722355101795,64.82467399430317,70.78953430495224,63.41356646565051,66.88516786423946,71.50421056498604],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":4.8294771028376},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":54.67615865143666,"ci95_mean_lo":35.34509998435015,"example":"E3","id":"e3_pisco_d135","iqr":14.956303054845641,"mean":45.010629317893404,"median":41.28306884978908,"method":"PISCO d=135","n":9,"per_seed_values":[52.25149753873912,33.1691557693188,71.13089864939653,54.58386440865856,43.863261651849044,40.8801349542787,37.29519448389348,41.28306884978908,30.63858755511728],"role":"main","seeds":[0,1,2,3,4,5,6,7,8],"std":12.574386818288382},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":37.924086244456184,"ci95_mean_lo":34.88866188677482,"example":"E3","id":"e3_bo_ard_d135","iqr":2.4634688116495624,"mean":36.4063740656155,"median":36.09881181310182,"method":"ARD-BO d=135","n":9,"per_seed_values":[40.28516829663448,37.23208835782871,34.03533431009833,34.768619546179146,38.05930300627153,34.66284202075187,35.46816838373789,36.09881181310182,37.04703085593584],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":1.9744702392387214},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":54.318420373557515,"ci95_mean_lo":48.716904283333804,"example":"E3","id":"e3_cmaes_d135","iqr":4.352452041250359,"mean":51.51766232844566,"median":51.36332750792246,"method":"CMA-ES d=135","n":9,"per_seed_values":[48.51540652950418,48.404438389435064,51.76380410981815,59.22153326632898,52.86785857075454,47.923867554409036,51.36332750792246,54.365751125835224,49.2329739020033],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":3.6436509401973174},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":69.68320230677212,"ci95_mean_lo":64.85907810437429,"example":"E3","id":"e3_de_d135","iqr":4.847793539085117,"mean":67.2711402055732,"median":67.59710156258325,"method":"DE d=135","n":9,"per_seed_values":[66.01032527777156,64.22738563969521,69.08545069206421,63.03600212288636,72.53858750896686,70.04781423865751,67.59710156258325,64.2376571529791,68.65993765455498],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":3.1379762911639753},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":73.16604940870076,"ci95_mean_lo":68.81752112473694,"example":"E3","id":"e3_random_d135","iqr":1.1608602123251615,"mean":70.99178526671885,"median":72.00108730170018,"method":"Random search d=135","n":9,"per_seed_values":[72.00108730170018,72.1895737160678,71.38827233002851,73.30513758180322,72.85922157272232,71.02871350374264,72.03220830906213,70.26808602583,63.85376705951281],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.82861263185386},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":53.55078878247524,"ci95_mean_lo":37.42225293252376,"example":"E3","id":"e3_pisco_d243","iqr":16.950518222371365,"mean":45.4865208574995,"median":39.15606186009975,"method":"PISCO d=243","n":9,"per_seed_values":[34.304130389894254,36.313233804835825,37.93384061173205,39.15606186009975,59.83870451151052,53.636354207639506,53.160826958652514,58.34969938786292,36.68583598526814],"role":"main","seeds":[0,1,2,3,4,5,6,7,8],"std":10.491223066599261},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":43.983916137004314,"ci95_mean_lo":40.493317169998775,"example":"E3","id":"e3_bo_ard_d243","iqr":3.3244741748510336,"mean":42.238616653501545,"median":43.01575466129219,"method":"ARD-BO d=243","n":9,"per_seed_values":[39.752572580635125,41.8891847643682,43.07704675548616,39.57569160625735,46.44760316542368,43.590339425474234,39.72751850786849,43.071838414708466,43.01575466129219],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.270550330147058},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":55.706642875412946,"ci95_mean_lo":52.10851093044523,"example":"E3","id":"e3_cmaes_d243","iqr":3.245999514390981,"mean":53.90757690292909,"median":54.15078232474032,"method":"CMA-ES d=243","n":9,"per_seed_values":[54.15078232474032,57.48583840044586,49.96334339992635,53.549354376861025,52.037763220752645,55.283762735143625,55.8423287827429,51.74216761720325,55.11285126854577],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.3404979353923503},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":75.43483976230094,"ci95_mean_lo":70.49099563475441,"example":"E3","id":"e3_de_d243","iqr":3.8967375865027662,"mean":72.96291769852768,"median":72.83686077292742,"method":"DE d=243","n":9,"per_seed_values":[66.2720154562615,72.83686077292742,70.72592851310394,74.82919617787243,76.5442343878196,71.75558640473226,75.65232399123502,72.33838110147099,75.711732481326],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":3.215851211239581},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":73.6879625150015,"ci95_mean_lo":69.53902013832938,"example":"E3","id":"e3_random_d243","iqr":1.0792533353066034,"mean":71.61349132666544,"median":71.22847919058232,"method":"Random search d=243","n":9,"per_seed_values":[76.75822311356693,71.7237234619004,68.91485484094221,70.89276463249897,74.93117450464753,68.185191833144,71.22847919058232,71.24254023611289,70.6444701265938],"role":"baseline","seeds":[0,1,2,3,4,5,6,7,8],"std":2.698786819155974},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":53.19101128810529,"ci95_mean_lo":42.76255286511551,"example":"E3","id":"e3_scalar","iqr":9.427958763738097,"mean":47.9767820766104,"median":49.49874771344114,"method":"scalar surrogate","n":9,"per_seed_values":[52.46210787924256,46.06747647350576,52.42099431064484,38.16406092778913,50.00990719581887,59.785854827429176,42.993035546906746,40.388853814715326,49.49874771344114],"role":"ablation","seeds":[0,1,2,3,4,5,6,7,8],"std":6.7834603571080345},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":47.60379621487837,"ci95_mean_lo":35.97689934761797,"example":"E3","id":"e3_no_physics","iqr":1.7199277750372275,"mean":41.79034778124817,"median":39.15922283757444,"method":"PISCO lambda=0","n":9,"per_seed_values":[39.59158520305886,39.15922283757444,40.52988576419511,38.80995798915788,38.92215010072008,61.823412122803425,38.23774356179831,40.71976156061873,38.31941089130669],"role":"ablation","seeds":[0,1,2,3,4,5,6,7,8],"std":7.563015622843409},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":41.32899579225418,"ci95_mean_lo":35.20361177905665,"example":"E3","id":"e3_nested_pisco","iqr":4.067674756972956,"mean":38.26630378565542,"median":38.19550415380146,"method":"coarse-to-fine PISCO","n":9,"per_seed_values":[30.42895113749443,43.62356569827087,40.902315722924484,38.60777579415961,37.569579968267924,36.83464096595153,42.71072257362699,35.52367805640146,38.19550415380146],"role":"control","seeds":[0,1,2,3,4,5,6,7,8],"std":3.984414372692708},{"budget_fem":[150,150,150,150,150,150,150,150,150],"ci95_mean_hi":45.200707166396334,"ci95_mean_lo":41.27248519922136,"example":"E3","id":"e3_nested_bo","iqr":3.510560986596481,"mean":43.236596182808846,"median":43.16585179850715,"method":"coarse-to-fine ARD-BO","n":9,"per_seed_values":[46.93321267426282,43.16585179850715,39.67099068971312,45.163517213330586,41.652956226734105,43.51786360207509,39.82701618180221,46.050176278323974,43.14778098053058],"role":"control","seeds":[0,1,2,3,4,5,6,7,8],"std":2.555213523171235},{"budget_fem":[150,150,150],"ci95_mean_hi":60.46919555561416,"ci95_mean_lo":26.846230053174203,"example":"E3","id":"e3_pisco_starts_6_d35","iqr":6.307182126284985,"mean":43.65771280439418,"median":40.824822631128946,"method":"PISCO, 6 search starts","n":3,"per_seed_values":[38.766975764741815,51.381340017311786,40.824822631128946],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":6.767532521123622},{"budget_fem":[150,150,150],"ci95_mean_hi":55.84669825237069,"ci95_mean_lo":32.706176066095274,"example":"E3","id":"e3_pisco_starts_12_d35","iqr":4.048182994702373,"mean":44.276437159232984,"median":41.6166171120679,"method":"PISCO, 12 search starts","n":3,"per_seed_values":[41.6166171120679,49.654530177517906,41.55816418811315],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":4.657656875626785},{"budget_fem":[150,150,150],"ci95_mean_hi":58.22557252724573,"ci95_mean_lo":33.90756691556474,"example":"E3","id":"e3_pisco_starts_24_d35","iqr":4.537568459527037,"mean":46.06656972140524,"median":48.185733909343696,"method":"PISCO, 24 search starts","n":3,"per_seed_values":[49.54455608696305,48.185733909343696,40.46941916790896],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":4.894657308379749},{"budget_fem":[150,150,150],"ci95_mean_hi":51.16869693799051,"ci95_mean_lo":32.92604748927088,"example":"E3","id":"e3_pisco_top_3_d35","iqr":3.2934456506754586,"mean":42.047372213630695,"median":40.17279434586219,"method":"PISCO, 3 FEM-verified candidates per cycle","n":3,"per_seed_values":[40.17279434586219,46.27810679819041,39.6912154968395],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":3.671827323104768},{"budget_fem":[150,150,150],"ci95_mean_hi":55.84669825237069,"ci95_mean_lo":32.706176066095274,"example":"E3","id":"e3_pisco_top_6_d35","iqr":4.048182994702373,"mean":44.276437159232984,"median":41.6166171120679,"method":"PISCO, 6 FEM-verified candidates per cycle","n":3,"per_seed_values":[41.6166171120679,49.654530177517906,41.55816418811315],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":4.657656875626785},{"budget_fem":[150,150,150],"ci95_mean_hi":54.85438527083049,"ci95_mean_lo":35.12667727185402,"example":"E3","id":"e3_pisco_top_12_d35","iqr":3.9597178978834293,"mean":44.990531271342256,"median":45.331852666867334,"method":"PISCO, 12 FEM-verified candidates per cycle","n":3,"per_seed_values":[45.331852666867334,48.77958847146314,40.86015267569628],"role":"exploratory_sensitivity","seeds":[0,1,2],"std":3.97073558073322}]''')
FROZEN_E3_COMPARISONS = json.loads(r'''[{"ci95_difference_hi":7.1555067815546565,"ci95_difference_lo":3.8027680644392206,"family":"dimension_pisco_vs_ard_bo","label":"PISCO vs ARD-BO at d=35","mean_a":43.446108446679126,"mean_b":37.96697102368219,"mean_difference_a_minus_b":5.4791374229969385,"n_pairs":9,"p_exact_permutation_sensitivity":0.00390625,"p_holm":0.00026756987640337397,"p_raw":6.689246910084349e-05,"pairing":"common seed and common random initial anchors","significant_holm_005":true,"t_statistic":7.537070210859388,"test":"two-sided paired t-test"},{"ci95_difference_hi":-2.0501201220607195,"ci95_difference_lo":-6.992740196665792,"degrees_of_freedom":14.577948470525458,"family":"dimension_pisco_vs_cmaes","label":"PISCO vs CMA-ES at d=35","mean_a":43.446108446679126,"mean_b":47.96753860604238,"mean_difference_a_minus_b":-4.5214301593632555,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.0020567667626491155,"p_holm":0.0058585909804060605,"p_raw":0.0014646477451015151,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-3.9094885163225643,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-16.452739050074726,"ci95_difference_lo":-21.820817543790838,"degrees_of_freedom":15.851918815584263,"family":"dimension_pisco_vs_de","label":"PISCO vs DE at d=35","mean_a":43.446108446679126,"mean_b":62.58288674361191,"mean_difference_a_minus_b":-19.136778296932782,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":4.113533525298231e-05,"p_holm":3.092260132774628e-10,"p_raw":7.73065033193657e-11,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-15.126075142589347,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-16.999940790070063,"ci95_difference_lo":-22.85332262565253,"degrees_of_freedom":15.902894872922472,"family":"dimension_pisco_vs_random","label":"PISCO vs random search at d=35","mean_a":43.446108446679126,"mean_b":63.37274015454042,"mean_difference_a_minus_b":-19.926631707861297,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":4.113533525298231e-05,"p_holm":5.880686260483873e-10,"p_raw":1.4701715651209683e-10,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-14.440727225568939,"test":"two-sided Welch t-test"},{"ci95_difference_hi":19.554168565330748,"ci95_difference_lo":4.119214735745694,"family":"dimension_pisco_vs_ard_bo","label":"PISCO vs ARD-BO at d=62","mean_a":45.423734793689526,"mean_b":33.587043143151305,"mean_difference_a_minus_b":11.83669165053822,"n_pairs":9,"p_exact_permutation_sensitivity":0.01171875,"p_holm":0.022965415774423868,"p_raw":0.007655138591474623,"pairing":"common seed and common random initial anchors","significant_holm_005":true,"t_statistic":3.536837258425613,"test":"two-sided paired t-test"},{"ci95_difference_hi":4.421910837362564,"ci95_difference_lo":-12.124151258359934,"degrees_of_freedom":8.947247957320688,"family":"dimension_pisco_vs_cmaes","label":"PISCO vs CMA-ES at d=62","mean_a":45.423734793689526,"mean_b":49.27485500418821,"mean_difference_a_minus_b":-3.851120210498685,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.30744549568078977,"p_holm":0.33786478218313754,"p_raw":0.31951796922103043,"pairing":"independent stochastic runs","significant_holm_005":false,"t_statistic":-1.0539879665531797,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-10.16180011367924,"ci95_difference_lo":-27.004048604735083,"degrees_of_freedom":10.181458940868763,"family":"dimension_pisco_vs_de","label":"PISCO vs DE at d=62","mean_a":45.423734793689526,"mean_b":64.00665915289669,"mean_difference_a_minus_b":-18.582924359207162,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.00028794734677087616,"p_holm":0.0011735471122293034,"p_raw":0.0005867735561146517,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-4.904989487968553,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-10.804763378735453,"ci95_difference_lo":-27.933749443214218,"degrees_of_freedom":11.156829182460974,"family":"dimension_pisco_vs_random","label":"PISCO vs random search at d=62","mean_a":45.423734793689526,"mean_b":64.79299120466436,"mean_difference_a_minus_b":-19.369256410974835,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.00032908268202385847,"p_holm":0.00041717355594086454,"p_raw":0.00040506051791957084,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-4.969172848737851,"test":"two-sided Welch t-test"},{"ci95_difference_hi":18.79363721849173,"ci95_difference_lo":-1.5851267139359546,"family":"dimension_pisco_vs_ard_bo","label":"PISCO vs ARD-BO at d=135","mean_a":45.010629317893404,"mean_b":36.4063740656155,"mean_difference_a_minus_b":8.604255252277888,"n_pairs":9,"p_exact_permutation_sensitivity":0.0625,"p_holm":0.1747205102104122,"p_raw":0.0873602551052061,"pairing":"common seed and common random initial anchors","significant_holm_005":false,"t_statistic":1.9472670921451023,"test":"two-sided paired t-test"},{"ci95_difference_hi":3.31116813376431,"ci95_difference_lo":-16.325234154868824,"degrees_of_freedom":9.334039755196292,"family":"dimension_pisco_vs_cmaes","label":"PISCO vs CMA-ES at d=135","mean_a":45.010629317893404,"mean_b":51.51766232844566,"mean_difference_a_minus_b":-6.507033010552256,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.15660222130810367,"p_holm":0.33786478218313754,"p_raw":0.16893239109156877,"pairing":"independent stochastic runs","significant_holm_005":false,"t_statistic":-1.4911102532571539,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-12.486747751721222,"ci95_difference_lo":-32.034274023638375,"degrees_of_freedom":8.992577808420668,"family":"dimension_pisco_vs_de","label":"PISCO vs DE at d=135","mean_a":45.010629317893404,"mean_b":67.2711402055732,"mean_difference_a_minus_b":-22.2605108876798,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.00037021801727684083,"p_holm":0.0011735471122293034,"p_raw":0.0006023743736641425,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-5.152888383500093,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-16.23000592333357,"ci95_difference_lo":-35.73230597431732,"degrees_of_freedom":8.807574612495447,"family":"dimension_pisco_vs_random","label":"PISCO vs random search at d=135","mean_a":45.010629317893404,"mean_b":70.99178526671885,"mean_difference_a_minus_b":-25.981155948825446,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.00016454134101192923,"p_holm":0.00041717355594086454,"p_raw":0.00020858677797043227,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-6.047469140404676,"test":"two-sided Welch t-test"},{"ci95_difference_hi":10.573698652367668,"ci95_difference_lo":-4.07789024437176,"family":"dimension_pisco_vs_ard_bo","label":"PISCO vs ARD-BO at d=243","mean_a":45.4865208574995,"mean_b":42.238616653501545,"mean_difference_a_minus_b":3.2479042039979538,"n_pairs":9,"p_exact_permutation_sensitivity":0.328125,"p_holm":0.3365235555927343,"p_raw":0.3365235555927343,"pairing":"common seed and common random initial anchors","significant_holm_005":false,"t_statistic":1.0223710995376116,"test":"two-sided paired t-test"},{"ci95_difference_hi":-0.2866672373434316,"ci95_difference_lo":-16.555444853515745,"degrees_of_freedom":8.794346234874922,"family":"dimension_pisco_vs_cmaes","label":"PISCO vs CMA-ES at d=243","mean_a":45.4865208574995,"mean_b":53.90757690292909,"mean_difference_a_minus_b":-8.421056045429587,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.0295351707116413,"p_holm":0.13174477954340325,"p_raw":0.043914926514467746,"pairing":"independent stochastic runs","significant_holm_005":false,"t_statistic":-2.3502534137419646,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-19.266847659727397,"ci95_difference_lo":-35.685946022328956,"degrees_of_freedom":9.49019244732173,"family":"dimension_pisco_vs_de","label":"PISCO vs DE at d=243","mean_a":45.4865208574995,"mean_b":72.96291769852768,"mean_difference_a_minus_b":-27.476396841028176,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":4.113533525298231e-05,"p_holm":8.174020278312403e-05,"p_raw":2.7246734261041345e-05,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-7.511978506705846,"test":"two-sided Welch t-test"},{"ci95_difference_hi":-17.96592625017297,"ci95_difference_lo":-34.28801468815891,"degrees_of_freedom":9.054161861021715,"family":"dimension_pisco_vs_random","label":"PISCO vs random search at d=243","mean_a":45.4865208574995,"mean_b":71.61349132666544,"mean_difference_a_minus_b":-26.12697046916594,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":4.113533525298231e-05,"p_holm":0.00014224639135609126,"p_raw":4.741546378536376e-05,"pairing":"independent stochastic runs","significant_holm_005":true,"t_statistic":-7.235527871688718,"test":"two-sided Welch t-test"},{"ci95_difference_hi":0.7751382124766772,"ci95_difference_lo":-9.836485472339199,"family":"architecture_ablations","label":"field surrogate vs scalar surrogate at d=35","mean_a":43.446108446679126,"mean_b":47.9767820766104,"mean_difference_a_minus_b":-4.530673629931261,"n_pairs":9,"p_exact_permutation_sensitivity":0.08984375,"p_holm":0.16891844868460112,"p_raw":0.08445922434230056,"pairing":"common seed and common random initial anchors","significant_holm_005":false,"t_statistic":-1.9691147059486471,"test":"two-sided paired t-test"},{"ci95_difference_hi":7.653870051012651,"ci95_difference_lo":-4.342348720150724,"family":"architecture_ablations","label":"physics residual vs no physics at d=35","mean_a":43.446108446679126,"mean_b":41.79034778124817,"mean_difference_a_minus_b":1.6557606654309633,"n_pairs":9,"p_exact_permutation_sensitivity":0.5703125,"p_holm":0.5421886566257487,"p_raw":0.5421886566257487,"pairing":"common seed and common random initial anchors","significant_holm_005":false,"t_statistic":0.6365657402931675,"test":"two-sided paired t-test"},{"ci95_difference_hi":-1.5942287938131967,"ci95_difference_lo":-10.778639796911254,"family":"architecture_mechanism_sensitivity","label":"no-physics field output vs scalar output at d=35","mean_a":41.79034778124817,"mean_b":47.9767820766104,"mean_difference_a_minus_b":-6.186434295362225,"n_pairs":9,"p_exact_permutation_sensitivity":0.03125,"p_holm":0.01452096432778434,"p_raw":0.01452096432778434,"pairing":"common seed and common random initial anchors","significant_holm_005":true,"t_statistic":-3.1065558939951656,"test":"two-sided paired t-test"},{"ci95_difference_hi":-0.2651648576344474,"ci95_difference_lo":-9.675419936672414,"family":"nested_control","label":"coarse-to-fine PISCO vs coarse-to-fine ARD-BO","mean_a":38.26630378565542,"mean_b":43.236596182808846,"mean_difference_a_minus_b":-4.970292397153431,"n_pairs":9,"p_exact_permutation_sensitivity":0.0390625,"p_holm":0.040820778749457415,"p_raw":0.040820778749457415,"pairing":"common seed and common random initial anchors","significant_holm_005":true,"t_statistic":-2.4359626226372972,"test":"two-sided paired t-test"},{"ci95_difference_hi":1.0861508471165386,"ci95_difference_lo":-15.526584990804704,"degrees_of_freedom":10.260758838508975,"family":"nested_strategy","label":"coarse-to-fine PISCO vs static PISCO at d=243","mean_a":38.26630378565542,"mean_b":45.4865208574995,"mean_difference_a_minus_b":-7.220217071844083,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.07091731797614151,"p_holm":0.16335162622872598,"p_raw":0.08167581311436299,"pairing":"independent stochastic runs","significant_holm_005":false,"t_statistic":-1.9301336844138837,"test":"two-sided Welch t-test"},{"ci95_difference_hi":3.416159696111211,"ci95_difference_lo":-1.4202006374966087,"degrees_of_freedom":15.78186185019037,"family":"nested_strategy","label":"coarse-to-fine ARD-BO vs static ARD-BO at d=243","mean_a":43.236596182808846,"mean_b":42.238616653501545,"mean_difference_a_minus_b":0.997979529307301,"n_a":9,"n_b":9,"p_exact_permutation_sensitivity":0.3878445084327437,"p_holm":0.3942389919916011,"p_raw":0.3942389919916011,"pairing":"independent stochastic runs","significant_holm_005":false,"t_statistic":0.8758657490644313,"test":"two-sided Welch t-test"}]''')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _close(actual: Any, expected: Any, *, tolerance: float = 1e-10) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return bool(np.isclose(
            float(actual), float(expected), rtol=tolerance,
            atol=tolerance, equal_nan=True))
    return actual == expected


def _claim_summary(values: np.ndarray) -> dict[str, Any]:
    from scipy.stats import t

    values = np.asarray(values, dtype=float)
    if values.size == 1:
        value = float(values[0])
        return {
            "n": 1, "mean": value, "std": None, "median": value,
            "iqr": None, "ci95_mean_lo": None, "ci95_mean_hi": None,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = float(t.ppf(0.975, values.size - 1)
                       * std / math.sqrt(values.size))
    return {
        "n": int(values.size), "mean": mean, "std": std,
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75)
                     - np.percentile(values, 25)),
        "ci95_mean_lo": mean - half_width,
        "ci95_mean_hi": mean + half_width,
    }


def _verify_frozen_claim_rows() -> list[str]:
    """Recompute all 94 descriptive rows from their frozen seed vectors."""
    failures: list[str] = []
    identifiers: set[str] = set()
    for row in FROZEN_CLAIM_ROWS:
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier in identifiers:
            failures.append(f"invalid or duplicate frozen claim id {identifier!r}")
            continue
        identifiers.add(identifier)
        values = row.get("per_seed_values", [])
        if values:
            vector = np.asarray(values, dtype=float)
            if not np.isfinite(vector).all():
                failures.append(f"{identifier}: non-finite seed value")
                continue
            current = _claim_summary(vector)
            for field, actual in current.items():
                if not _close(actual, row.get(field)):
                    failures.append(
                        f"{identifier}.{field}: recomputed {actual!r}, "
                        f"frozen {row.get(field)!r}")
        else:
            if row.get("n") != 1 or not _close(
                    row.get("mean"), row.get("value")):
                failures.append(
                    f"{identifier}: invalid deterministic aggregate")
    if len(FROZEN_CLAIM_ROWS) != 94:
        failures.append(
            f"frozen claim matrix has {len(FROZEN_CLAIM_ROWS)} instead of "
            "94 rows")
    return failures


def _exact_sign_flip(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    observed = abs(float(difference.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    exceedances = 0
    for mask in range(1 << difference.size):
        signed_sum = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(difference))
        if abs(signed_sum / difference.size) >= observed - tolerance:
            exceedances += 1
    return exceedances / (1 << difference.size)


def _exact_label_permutation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate((a, b))
    observed = abs(float(a.mean() - b.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    pooled_sum = float(pooled.sum())
    exceedances = 0
    total = math.comb(pooled.size, a.size)
    for selected in combinations(range(pooled.size), a.size):
        sum_a = sum(float(pooled[index]) for index in selected)
        difference = sum_a / a.size - (pooled_sum - sum_a) / b.size
        if abs(difference) >= observed - tolerance:
            exceedances += 1
    return exceedances / total


def _paired_comparison(a: np.ndarray, b: np.ndarray, *, label: str,
                       family: str) -> dict[str, Any]:
    from scipy import stats

    difference = np.asarray(a, float) - np.asarray(b, float)
    valid = len(difference)>1 and np.std(difference)>0
    statistic, p_value = stats.ttest_rel(a, b) if valid else (float("nan"), float("nan"))
    summary = _claim_summary(difference)
    return {
        "label": label, "family": family,
        "test": "two-sided paired t-test",
        "pairing": "common seed and common random initial anchors",
        "n_pairs": int(difference.size),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_difference_a_minus_b": summary["mean"],
        "ci95_difference_lo": summary["ci95_mean_lo"],
        "ci95_difference_hi": summary["ci95_mean_hi"],
        "t_statistic": float(statistic), "p_raw": float(p_value),
        "p_holm": None,
        "p_exact_permutation_sensitivity": _exact_sign_flip(difference),
    }


def _welch_comparison(a: np.ndarray, b: np.ndarray, *, label: str,
                      family: str) -> dict[str, Any]:
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    difference = float(a.mean() - b.mean())
    variance = float(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    standard_error = math.sqrt(variance)
    denominator = ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                   + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    degrees = variance ** 2 / denominator if denominator>0 else float("nan")
    critical = float(stats.t.ppf(0.975, degrees))
    statistic, p_value = stats.ttest_ind(a, b, equal_var=False) if variance>0 else (float("nan"), float("nan"))
    return {
        "label": label, "family": family,
        "test": "two-sided Welch t-test",
        "pairing": "independent stochastic runs",
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference_a_minus_b": difference,
        "ci95_difference_lo": difference - critical * standard_error,
        "ci95_difference_hi": difference + critical * standard_error,
        "degrees_of_freedom": degrees,
        "t_statistic": float(statistic), "p_raw": float(p_value),
        "p_holm": None,
        "p_exact_permutation_sensitivity": _exact_label_permutation(a, b),
    }


def _holm_adjust(comparisons_: list[dict[str, Any]]) -> None:
    for family in sorted({entry["family"] for entry in comparisons_}):
        members = [entry for entry in comparisons_
                   if entry["family"] == family]
        ordered = sorted([entry for entry in members if math.isfinite(entry["p_raw"])],
                         key=lambda entry: entry["p_raw"])
        running = 0.0
        for rank, entry in enumerate(ordered):
            adjusted = min(1.0, (len(members) - rank) * entry["p_raw"])
            running = max(running, adjusted)
            entry["p_holm"] = running
            entry["significant_holm_005"] = bool(running < 0.05)


def _recompute_e3_comparisons(values_by_id=None) -> list[dict[str, Any]]:
    rows = {row["id"]: row for row in FROZEN_CLAIM_ROWS}

    def vector(identifier: str) -> np.ndarray:
        return np.asarray(rows[identifier]["per_seed_values"] if values_by_id is None
                          else values_by_id[identifier], dtype=float)

    comparisons_: list[dict[str, Any]] = []
    for dimension in (35, 62, 135, 243):
        pisco = vector(f"e3_pisco_d{dimension}")
        bo = (vector("e3_sklearn_bo_d35") if dimension == 35
              else vector(f"e3_bo_ard_d{dimension}"))
        comparisons_.append(_paired_comparison(
            pisco, bo, label=f"PISCO vs ARD-BO at d={dimension}",
            family="dimension_pisco_vs_ard_bo"))
        for key, method in (
                ("cmaes", "CMA-ES"), ("de", "DE"),
                ("random", "random search")):
            comparisons_.append(_welch_comparison(
                pisco, vector(f"e3_{key}_d{dimension}"),
                label=f"PISCO vs {method} at d={dimension}",
                family=f"dimension_pisco_vs_{key}"))
    comparisons_.extend([
        _paired_comparison(
            vector("e3_pisco_d35"), vector("e3_scalar"),
            label="field surrogate vs scalar surrogate at d=35",
            family="architecture_ablations"),
        _paired_comparison(
            vector("e3_pisco_d35"), vector("e3_no_physics"),
            label="physics residual vs no physics at d=35",
            family="architecture_ablations"),
        _paired_comparison(
            vector("e3_no_physics"), vector("e3_scalar"),
            label="no-physics field output vs scalar output at d=35",
            family="architecture_mechanism_sensitivity"),
        _paired_comparison(
            vector("e3_nested_pisco"), vector("e3_nested_bo"),
            label="coarse-to-fine PISCO vs coarse-to-fine ARD-BO",
            family="nested_control"),
        _welch_comparison(
            vector("e3_nested_pisco"), vector("e3_pisco_d243"),
            label="coarse-to-fine PISCO vs static PISCO at d=243",
            family="nested_strategy"),
        _welch_comparison(
            vector("e3_nested_bo"), vector("e3_bo_ard_d243"),
            label="coarse-to-fine ARD-BO vs static ARD-BO at d=243",
            family="nested_strategy"),
    ])
    _holm_adjust(comparisons_)
    return comparisons_


def _verify_frozen_comparisons() -> list[str]:
    failures: list[str] = []
    current = _recompute_e3_comparisons()
    if len(current) != 22 or len(FROZEN_E3_COMPARISONS) != 22:
        return [
            f"statistical comparison count is {len(current)}/"
            f"{len(FROZEN_E3_COMPARISONS)}, expected 22/22"]
    for actual, expected in zip(current, FROZEN_E3_COMPARISONS):
        if set(actual) != set(expected):
            failures.append(
                f"{expected.get('label')}: comparison fields differ")
            continue
        for field, frozen_value in expected.items():
            if not _close(actual[field], frozen_value, tolerance=1e-9):
                failures.append(
                    f"{expected['label']}.{field}: recomputed "
                    f"{actual[field]!r}, frozen {frozen_value!r}")
    return failures


class ProgressDisplay:
    """Dependency-free progress display used for long resumable stages."""

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.completed = 0
        self.started = time.perf_counter()

    def _bar(self) -> str:
        width = 28
        filled = round(width * self.completed / self.total)
        return "█" * filled + "░" * (width - filled)

    def run(self, label: str, operation: Callable[[], Any]) -> Any:
        print(
            f"\n[{self._bar()}] {self.completed / self.total:6.1%} | "
            f"{self.completed + 1}/{self.total} | {label}", flush=True)
        started = time.perf_counter()
        import threading
        stopped = threading.Event()
        def heartbeat():
            while not stopped.wait(30):
                print(f"[{self._bar()}] {label} | running {(time.perf_counter()-started)/60:.1f} min", flush=True)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            result = operation()
        finally:
            stopped.set()
            thread.join()
        self.completed += 1
        elapsed = time.perf_counter() - self.started
        print(
            f"[{self._bar()}] {self.completed / self.total:6.1%} | "
            f"step {time.perf_counter() - started:.1f} s | "
            f"total {elapsed:.1f} s", flush=True)
        return result


class PublicationBuilder:
    """Prepare all paper-facing output from numerical ``SolverResult`` data.

    The class has no optimisation logic.  It validates results, computes
    descriptive statistics and confidence intervals, writes machine-readable
    tables, renders every manuscript figure and records hashes in a manifest.
    """

    FIGURE_NAMES = (
        "fig_02_designfeld_zu_physik_en.png",
        "fig_02_designfeld_zu_physik_en.jpeg",
        "fig_08_ellipse_resultate_en.png",
        "fig_08_ellipse_resultate_en.jpeg",
        "fig_09_ellipse_verlauf_en.png",
        "fig_09_ellipse_verlauf_en.jpeg",
        "fig_11_lbracket_resultate_en.png",
        "fig_11_lbracket_resultate_en.jpeg",
        "fig_13_fin_setup_en.png",
        "fig_13_fin_setup_en.jpeg",
        "fig_14_fin_results_en.png",
        "fig_14_fin_results_en.jpeg",
        "schema_workflow_detail_en.pdf",
        "schema_workflow_detail_en.jpeg",
        "schema_verschaltung_en.pdf",
        "schema_verschaltung_en.jpeg",
        "ellipse_setup_en.pdf",
        "ellipse_setup_en.jpeg",
        "graphical_abstract.pdf",
        "graphical_abstract.jpeg",
        "Figure_01.jpeg",
        "Figure_02.jpeg",
        "Figure_03.jpeg",
        "Figure_04.jpeg",
        "Figure_05.jpeg",
        "Figure_06.jpeg",
        "Figure_07.jpeg",
        "Figure_08.jpeg",
        "Figure_09.jpeg",
        "Graphical_Abstract.jpeg",
    )

    SUBMISSION_FIGURE_MAP = (
        ("fig_02_designfeld_zu_physik_en.jpeg", "Figure_01.jpeg"),
        ("schema_workflow_detail_en.jpeg", "Figure_02.jpeg"),
        ("schema_verschaltung_en.jpeg", "Figure_03.jpeg"),
        ("ellipse_setup_en.jpeg", "Figure_04.jpeg"),
        ("fig_08_ellipse_resultate_en.jpeg", "Figure_05.jpeg"),
        ("fig_09_ellipse_verlauf_en.jpeg", "Figure_06.jpeg"),
        ("fig_11_lbracket_resultate_en.jpeg", "Figure_07.jpeg"),
        ("fig_13_fin_setup_en.jpeg", "Figure_08.jpeg"),
        ("fig_14_fin_results_en.jpeg", "Figure_09.jpeg"),
        ("graphical_abstract.jpeg", "Graphical_Abstract.jpeg"),
    )
    JOURNAL_FIGURE_SELECTION = {
        "Figure_01": ".jpeg",
        "Figure_02": ".jpeg",
        "Figure_03": ".jpeg",
        "Figure_04": ".jpeg",
        "Figure_05": ".jpeg",
        "Figure_06": ".jpeg",
        "Figure_07": ".jpeg",
        "Figure_08": ".jpeg",
        "Figure_09": ".jpeg",
        "Graphical_Abstract": ".jpeg",
    }
    LEGACY_SUBMISSION_FIGURES = (
        "Figure_01.pdf", "Figure_01.tiff",
        "Figure_02.pdf", "Figure_02.eps",
        "Figure_03.pdf", "Figure_03.eps",
        "Figure_04.pdf", "Figure_04.eps",
        "Figure_05.pdf", "Figure_05.tiff",
        "Figure_06.pdf", "Figure_06.tiff",
        "Figure_07.pdf", "Figure_07.tiff",
        "Figure_08.pdf", "Figure_08.tiff",
        "Figure_09.pdf", "Figure_09.tiff",
        "Graphical_Abstract.eps",
    )
    LEGACY_GENERATED_FIGURES = (
        "fig_02_designfeld_zu_physik_en.pdf",
        "fig_02_designfeld_zu_physik_en.tiff",
        "fig_08_ellipse_resultate_en.pdf",
        "fig_08_ellipse_resultate_en.tiff",
        "fig_09_ellipse_verlauf_en.pdf",
        "fig_09_ellipse_verlauf_en.tiff",
        "fig_11_lbracket_resultate_en.pdf",
        "fig_11_lbracket_resultate_en.tiff",
        "fig_13_fin_setup_en.pdf",
        "fig_13_fin_setup_en.tiff",
        "fig_14_fin_results_en.pdf",
        "fig_14_fin_results_en.tiff",
        "schema_workflow_detail_en.eps",
        "schema_verschaltung_en.eps",
        "ellipse_setup_en.eps",
        "graphical_abstract.eps",
    )
    DATA_FIGURE_NAMES = (
        "fig_02_designfeld_zu_physik_en.png",
        "fig_08_ellipse_resultate_en.png",
        "fig_09_ellipse_verlauf_en.png",
        "fig_11_lbracket_resultate_en.png",
        "fig_13_fin_setup_en.png",
        "fig_14_fin_results_en.png",
    )

    def __init__(self, output_root: Path | str = DEFAULT_OUTPUT):
        self.output_root = Path(output_root)
        self.results_dir = self.output_root / "results"
        self.figure_dir = self.output_root / "figures"
        self.table_dir = self.output_root / "tables"
        for directory in (
                self.results_dir, self.figure_dir, self.table_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def bundle_path(self) -> Path:
        return self.results_dir / RESULT_BUNDLE

    def save_results(self, results: dict[str, Any]) -> Path:
        # An example-specific rerun updates that family without discarding the
        # already completed examples.  This is essential for long resumable
        # reproductions, where correcting one protocol must not force the
        # other two examples to run again.
        merged: dict[str, Any] = {}
        if self.bundle_path.is_file():
            try:
                merged.update(self.load_results())
            except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as error:
                raise RuntimeError("Existing bundle cannot be merged; choose a new output directory") from error
        for key in ("source_sha256", "training_policy", "numerical_execution"):
            signatures = {json.dumps(r.extra.get("r2_evidence", {}).get("configuration", {}).get(key), sort_keys=True)
                          for collection in (merged, results) for methods in collection.values()
                          for runs in methods.values() for r in runs}
            if len(signatures)>1:
                raise RuntimeError(f"Cannot mix {key}; choose a new output directory")
        for family, methods in results.items():
            target = merged.setdefault(family, {})
            for method, runs in methods.items():
                by_seed = {r.seed:r for r in target.get(method, [])}
                by_seed.update({r.seed:r for r in runs})
                target[method] = [by_seed[seed] for seed in sorted(by_seed)]
        payload = pickle.dumps({
            "protocol": solvers.PUBLICATION_PROTOCOL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "results": merged,
        }, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_bytes(self.bundle_path, payload)
        return self.bundle_path

    def load_results(self) -> dict[str, Any]:
        if not self.bundle_path.is_file():
            raise FileNotFoundError(
                f"No saved numerical result bundle at {self.bundle_path}. "
                "Run --all or --example first.")
        with self.bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("protocol") != solvers.PUBLICATION_PROTOCOL:
            raise RuntimeError("Saved results use a different protocol")
        return bundle["results"]

    @staticmethod
    def _values(runs: list[SolverResult]) -> np.ndarray:
        # Example 1 is reported relative to the compliance of the unweakened
        # plate.  The other examples have no such reference and therefore use
        # the solver-verified compliance directly.  Detecting the reference
        # from the result contract keeps this rule local to the data instead
        # of relying on method-name special cases.
        normalized = [result.best_compliance_normalized for result in runs]
        if all(value is not None for value in normalized):
            return np.asarray(normalized, dtype=float)
        if any(value is not None for value in normalized):
            raise RuntimeError(
                "A result family mixes normalized and unnormalized values")
        return np.asarray([
            result.best_compliance for result in runs], dtype=float)

    @staticmethod
    def _ci95(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        if len(values) < 2:
            return mean, mean
        from scipy.stats import t

        half_width = float(t.ppf(0.975, len(values) - 1)
                           * values.std(ddof=1) / math.sqrt(len(values)))
        return mean - half_width, mean + half_width

    def summaries(self, results: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "protocol": solvers.PUBLICATION_PROTOCOL,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "examples": {},
        }
        for example, methods in results.items():
            output["examples"][example] = {}
            for method, runs in methods.items():
                if not runs:
                    continue
                values = self._values(runs)
                lower, upper = self._ci95(values)
                output["examples"][example][method] = {
                    "n": len(values),
                    "seeds": [int(run.seed) for run in runs],
                    "mean": float(values.mean()),
                    "std": (float(values.std(ddof=1))
                            if len(values) > 1 else 0.0),
                    "median": float(np.median(values)),
                    "ci95_mean": [lower, upper],
                    "per_seed": values.tolist(),
                    "fem_calls": [int(run.fem_calls_total) for run in runs],
                    "wall_time_s": [float(run.wall_time_s) for run in runs],
                }
        return output

    def write_tables(self, summaries: dict[str, Any]) -> list[Path]:
        json_path = self.table_dir / "publication_results.json"
        _atomic_text(json_path, json.dumps(
            summaries, indent=2, sort_keys=True) + "\n")
        csv_path = self.table_dir / "publication_results.csv"
        rows = []
        for example, methods in summaries["examples"].items():
            for method, values in methods.items():
                rows.append({
                    "example": example,
                    "method": method,
                    "n": values["n"],
                    "mean_compliance": values["mean"],
                    "std_compliance": values["std"],
                    "ci95_low": values["ci95_mean"][0],
                    "ci95_high": values["ci95_mean"][1],
                    "mean_fem_calls": float(np.mean(values["fem_calls"])),
                    "mean_wall_time_s": float(np.mean(values["wall_time_s"])),
                })
        temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(csv_path)

        seed_path = self.table_dir / "per_seed_results.csv"
        seed_rows = []
        for example, methods in summaries["examples"].items():
            for method, values in methods.items():
                for seed, compliance, calls, runtime in zip(
                        values["seeds"], values["per_seed"],
                        values["fem_calls"], values["wall_time_s"]):
                    seed_rows.append({
                        "example": example, "method": method, "seed": seed,
                        "compliance": compliance, "fem_calls": calls,
                        "wall_time_s": runtime,
                    })
        temporary = seed_path.with_name(
            f".{seed_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
        temporary.replace(seed_path)
        return [json_path, csv_path, seed_path]

    def write_editable_table_bundle(self) -> list[Path]:
        """Extract all manuscript tables verbatim for a separate upload.

        The paper keeps the tables beside the discussion for readability.  The
        extracted LaTeX fragment satisfies portals that additionally request
        tables as a separate editable-text file.  No table is rendered as an
        image and the source order, captions and labels are preserved.
        """
        candidates = (
            ROOT.parent / "submission" / "PISCO_en.tex",
            ROOT.parent / "PISCO_en.tex",
        )
        manuscript = next((path for path in candidates if path.is_file()), None)
        if manuscript is None:
            print("No manuscript source found; editable table export skipped.")
            return []
        source = manuscript.read_text(encoding="utf-8")
        begin_token = r"\begin{table}"
        end_token = r"\end{table}"
        blocks: list[str] = []
        cursor = 0
        while True:
            begin = source.find(begin_token, cursor)
            if begin < 0:
                break
            end = source.find(end_token, begin)
            if end < 0:
                raise RuntimeError(
                    f"Unclosed table environment in {manuscript}")
            end += len(end_token)
            blocks.append(source[begin:end].strip())
            cursor = end
        if len(blocks) != 11:
            raise RuntimeError(
                f"Expected 11 manuscript tables, found {len(blocks)}")
        content = (
            "% Editable table upload for the PISCO revision.\n"
            "% Extracted verbatim from PISCO_en.tex; not supplementary "
            "material.\n"
            "% Requires the same LaTeX preamble as the main manuscript.\n\n"
            + "\n\n".join(blocks) + "\n")
        artifact = self.table_dir / "PISCO_tables.tex"
        _atomic_text(artifact, content)
        if manuscript.parent.name == "submission":
            _atomic_text(manuscript.parent / "PISCO_tables.tex", content)
        print(f"Wrote editable upload with {len(blocks)} tables: {artifact}")
        return [artifact]

    def write_submission_assets_manifest(self) -> list[Path]:
        """Validate private journal-upload assets and record their hashes."""
        submission_root = ROOT.parent / "submission"
        manuscript = submission_root / "PISCO_en.tex"
        figure_root = submission_root
        table_path = submission_root / "PISCO_tables.tex"
        if not manuscript.is_file():
            print("No private submission tree found; asset audit skipped.")
            return []
        expected_figures = [
            f"{stem}{suffix}"
            for stem, suffix in self.JOURNAL_FIGURE_SELECTION.items()]
        mapped_figures = [
            target for _, target in self.SUBMISSION_FIGURE_MAP]
        if mapped_figures != expected_figures:
            raise RuntimeError(
                "The generated journal-figure map does not match the "
                "one-format-per-figure selection")
        missing = [
            name for name in expected_figures
            if not (figure_root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing submission figures: " + ", ".join(missing))
        if not table_path.is_file():
            raise FileNotFoundError(f"Missing editable table file: {table_path}")

        journal_extensions = (".eps", ".tiff", ".tif", ".jpeg", ".jpg")
        for stem, selected_suffix in self.JOURNAL_FIGURE_SELECTION.items():
            selected = figure_root / f"{stem}{selected_suffix}"
            alternatives = [
                figure_root / f"{stem}{suffix}"
                for suffix in journal_extensions
                if (figure_root / f"{stem}{suffix}").is_file()]
            if alternatives != [selected]:
                names = ", ".join(path.name for path in alternatives)
                raise RuntimeError(
                    f"Expected one journal file for {stem} "
                    f"({selected.name}), found: {names or 'none'}")

        from PIL import Image

        records = []
        for name in expected_figures:
            path = figure_root / name
            record: dict[str, Any] = {
                "path": name,
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if path.suffix.lower() in (".tiff", ".tif", ".jpeg", ".jpg"):
                with Image.open(path) as image:
                    dpi = image.info.get("dpi", (0.0, 0.0))
                    record["pixels"] = [image.width, image.height]
                    record["dpi"] = [float(dpi[0]), float(dpi[1])]
                if min(record["dpi"]) < 650.0 - 1e-6:
                    raise RuntimeError(
                        f"{name} has only {record['dpi']} dpi")
            else:
                header = path.read_bytes()[:64]
                if not header.startswith(b"%!PS-Adobe") or b"EPSF" not in header:
                    raise RuntimeError(f"{name} is not a valid EPS file")
                record["vector"] = True
            records.append(record)

        manuscript_text = manuscript.read_text(encoding="utf-8")
        table_text = table_path.read_text(encoding="utf-8")
        table_count = table_text.count(r"\begin{table}")
        if table_count != 11:
            raise RuntimeError(
                f"Expected 11 editable tables, found {table_count}")
        if r"\corref" not in manuscript_text or r"\ead{" not in manuscript_text:
            raise RuntimeError(
                "Corresponding-author marker or contact email is missing")
        for stem in tuple(self.JOURNAL_FIGURE_SELECTION)[:-1]:
            include = rf"\includegraphics"
            filename = f"{stem}.jpeg"
            if include not in manuscript_text or filename not in manuscript_text:
                raise RuntimeError(
                    f"The manuscript does not directly include {filename}")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "journal_figure_requirement": (
                "Separate TIFF, EPS or JPEG files at >=650 dpi"),
            "manuscript_uses_uploaded_figure_files": True,
            "corresponding_author_marked": True,
            "one_journal_file_per_figure": True,
            "journal_upload_selection": {
                stem: f"{stem}{suffix}"
                for stem, suffix in self.JOURNAL_FIGURE_SELECTION.items()
            },
            "editable_table_count": table_count,
            "editable_tables": {
                "path": table_path.name,
                "size_bytes": table_path.stat().st_size,
                "sha256": _sha256(table_path),
            },
            "figures": records,
        }
        manifest_path = submission_root / "submission_assets_manifest.json"
        _atomic_text(manifest_path, json.dumps(
            manifest, indent=2, sort_keys=True) + "\n")
        print(
            "Validated nine numbered figures and one graphical abstract "
            "with one journal format each, plus "
            f"{table_count} editable tables: {manifest_path}")
        return [manifest_path]

    @staticmethod
    def _best(runs: list[SolverResult]) -> SolverResult:
        return min(runs, key=lambda result: result.best_compliance)

    @staticmethod
    def _trajectory_matrix(runs: list[SolverResult]
                           ) -> tuple[np.ndarray, np.ndarray]:
        maximum = max(run.fem_calls_total for run in runs)
        calls = np.arange(1, maximum + 1)
        rows = []
        for run in runs:
            source_x = np.asarray([point[0] for point in run.trajectory])
            reference = float(run.extra.get("solid_compliance") or 1.0)
            source_y = np.asarray([
                point[1] / reference for point in run.trajectory])
            rows.append(np.interp(
                calls, source_x, source_y,
                left=source_y[0], right=source_y[-1]))
        return calls, np.asarray(rows)

    @staticmethod
    def _save_figure(figure: plt.Figure, path: Path) -> Path:
        figure.savefig(path, dpi=700, bbox_inches="tight")
        if path.suffix.lower() == ".png":
            figure.savefig(
                path.with_suffix(".jpeg"), format="jpeg", dpi=700,
                bbox_inches="tight",
                facecolor="white",
                pil_kwargs={
                    "quality": 100,
                    "subsampling": 0,
                    "optimize": True,
                })
        plt.close(figure)
        return path

    @staticmethod
    def _displacement_magnitude(nelx: int, nely: int,
                                displacement: np.ndarray) -> np.ndarray:
        ux = np.asarray(displacement)[0::2].reshape(
            (nely + 1, nelx + 1), order="F")
        uy = np.asarray(displacement)[1::2].reshape(
            (nely + 1, nelx + 1), order="F")
        return np.sqrt(ux * ux + uy * uy)

    def _ellipse_solutions(self, results: dict[str, Any]
                           ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        runs = results.get("example1", {}).get("pisco", [])
        if not runs:
            raise RuntimeError("Example-1 PISCO results are required")
        problem = solvers.make_publication_ellipse_problem()
        best = self._best(runs)
        solid = float(best.extra["solid_compliance"])
        optimum = solvers._eval_fem(
            problem, np.asarray(best.best_design, dtype=float))
        reference = solvers._eval_fem(problem, np.asarray([
            0.5 * problem.Lx_mm, 0.5 * problem.Ly_mm,
            problem.target_hole_radius_mm,
            problem.target_hole_radius_mm, 0.0]))
        for solution in (reference, optimum):
            solution["normalised_compliance"] = (
                float(solution["compliance"]) / solid)
        return problem, reference, optimum

    def plot_physics_link(self, results: dict[str, Any]) -> Path:
        problem, _reference, optimum = self._ellipse_solutions(results)
        displacement = self._displacement_magnitude(
            problem.nelx, problem.nely, optimum["U"])
        extent = (0, problem.Lx_mm, 0, problem.Ly_mm)
        figure, axes = plt.subplots(1, 4, figsize=(14.5, 3.3))
        axes[0].imshow(optimum["rho"], origin="lower", cmap="gray_r",
                       vmin=0, vmax=1, extent=extent)
        axes[0].set_title(r"Design field $\rho$")
        axes[1].imshow(optimum["von_mises"], origin="lower", cmap="magma",
                       extent=extent)
        axes[1].set_title("Stress field")
        axes[2].imshow(displacement, origin="lower", cmap="viridis",
                       extent=extent)
        axes[2].set_title(r"Finite-element response $|u|$")
        axes[3].axis("off")
        axes[3].text(
            0.5, 0.58, r"$J=F^{T}u$", ha="center", va="center",
            fontsize=22)
        axes[3].text(
            0.5, 0.36, "Finite-element evaluated\ncompliance",
            ha="center", va="center", fontsize=12)
        for left, right in zip(axes[:-1], axes[1:]):
            figure.text(
                0.5 * (left.get_position().x1 + right.get_position().x0),
                0.52, r"$\rightarrow$", ha="center", va="center",
                fontsize=22)
        for axis in axes[:3]:
            axis.set_xlabel(r"$x$ [mm]")
            axis.set_ylabel(r"$y$ [mm]")
        figure.suptitle("From the design field to the physical response")
        return self._save_figure(
            figure, self.figure_dir / "fig_02_designfeld_zu_physik_en.png")

    def plot_ellipse_results(self, results: dict[str, Any]) -> Path:
        problem, reference, optimum = self._ellipse_solutions(results)
        extent = (0, problem.Lx_mm, 0, problem.Ly_mm)
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        annotations = []
        for row, solution, label in (
                (0, reference, "Centred circular design"),
                (1, optimum, "PISCO optimum (best seed)")):
            annotations.append((row, solution, label))
            displacement = self._displacement_magnitude(
                problem.nelx, problem.nely, solution["U"])
            panels = (
                (solution["rho"], "gray_r", "Design field"),
                (displacement, "viridis", "Displacement magnitude"),
                (solution["von_mises"], "magma", "von Mises stress"),
            )
            for column, (field, colour_map, title) in enumerate(panels):
                axes[row, column].imshow(
                    field, origin="lower", cmap=colour_map, extent=extent,
                    vmin=0 if column == 0 else None,
                    vmax=1 if column == 0 else None)
                if row == 0:
                    axes[row, column].set_title(title)
                axes[row, column].set_xlabel(r"$x$ [mm]")
                axes[row, column].set_ylabel(r"$y$ [mm]")
        figure.suptitle(
            "Example 1: comparison design and PISCO result", fontsize=14)
        figure.tight_layout(rect=[0.085, 0.01, 1, 0.93], h_pad=3.2)
        for row, solution, label in annotations:
            box = axes[row, 0].get_position()
            figure.text(
                0.018, 0.5 * (box.y0 + box.y1),
                f"{label}\n"
                rf"$\widetilde{{J}}={solution['normalised_compliance']:.4f}$",
                rotation=90, ha="center", va="center", fontsize=11)
        return self._save_figure(
            figure, self.figure_dir / "fig_08_ellipse_resultate_en.png")

    def plot_ellipse_trajectories(self, results: dict[str, Any]) -> Path:
        methods = results.get("example1", {})
        styles = {
            "de": ("Differential evolution", "#0072B2", "-", "o"),
            "bo": ("Bayesian optimisation", "#009E73", "--", "s"),
            "cmaes": ("CMA-ES", "#D55E00", "-.", "^"),
            "pisco": ("PISCO", "#CC79A7", (0, (1.5, 1.3)), "D"),
        }
        figure = plt.figure(figsize=(10.2, 6.8))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.18, 1.0), hspace=0.42, wspace=0.34)
        full_axis = figure.add_subplot(grid[0, :])
        zoom_axis = figure.add_subplot(grid[1, 0])
        endpoint_axis = figure.add_subplot(grid[1, 1])
        initial_calls = int(
            solvers.make_publication_pisco_e1_config().n_init)

        for method, (label, colour, line_style, marker) in styles.items():
            runs = methods.get(method, [])
            if not runs:
                continue
            calls, matrix = self._trajectory_matrix(runs)
            if method == "pisco":
                keep = calls >= initial_calls
                calls, matrix = calls[keep], matrix[:, keep]
            mean = matrix.mean(axis=0)
            mark_every = max(1, len(calls) // 9)
            for axis in (full_axis, zoom_axis):
                axis.step(
                    calls, mean, where="post", color=colour,
                    linestyle=line_style, linewidth=2.2, marker=marker,
                    markevery=mark_every, markersize=4.6,
                    markerfacecolor="white", markeredgewidth=1.0,
                    label=label)

        for axis in (full_axis, zoom_axis):
            axis.axvline(initial_calls, color="0.35", linestyle=":",
                         linewidth=1.4, zorder=0)
            axis.grid(True, color="0.88", linewidth=0.7)
            axis.set_xlabel("Cumulative FEM calls", fontsize=11)
            axis.set_ylabel("Normalised compliance", fontsize=11)
        full_axis.set_title(
            "(a) Complete FEM-verified mean trajectories", loc="left",
            fontsize=12.2)
        full_axis.set_xlim(0, 365)
        full_axis.set_ylim(1.0475, 1.101)
        full_axis.legend(
            frameon=True, framealpha=0.96, edgecolor="0.8", ncol=2,
            fontsize=10.0, loc="upper right")
        full_axis.annotate(
            f"PISCO search starts\nafter {initial_calls} initial calls",
            xy=(initial_calls, 1.095), xytext=(initial_calls + 16, 1.098),
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0},
            fontsize=9.8, color="0.25", ha="left", va="top")
        zoom_axis.set_title(
            "(b) Decision range after 50 FEM calls", loc="left",
            fontsize=12.2)
        zoom_axis.set_xlim(50, 160)
        zoom_axis.set_ylim(1.0485, 1.060)

        endpoint_order = ("cmaes", "de", "pisco", "bo")
        endpoint_labels = []
        for position, method in enumerate(endpoint_order):
            label, colour, _line_style, marker = styles[method]
            runs = methods[method]
            values = self._values(runs)
            calls = np.asarray([
                result.fem_calls_total for result in runs], dtype=float)
            mean = float(values.mean())
            deviation = float(values.std(ddof=1))
            offsets = np.linspace(-0.14, 0.14, len(values))
            endpoint_axis.scatter(
                values, position + offsets, s=32, facecolors="white",
                edgecolors=colour, linewidths=1.0, marker=marker, zorder=2)
            endpoint_axis.errorbar(
                mean, position, xerr=deviation, fmt=marker, color=colour,
                markerfacecolor=colour, markeredgecolor="white",
                markeredgewidth=0.8, markersize=7.6, linewidth=2.3,
                capsize=4.0, zorder=3)
            endpoint_labels.append(
                f"{label}\n{calls.mean():.0f} $\\pm$ "
                f"{calls.std(ddof=1):.0f} calls")
        endpoint_axis.set_title(
            "(c) FEM-verified result at native stopping", loc="left",
            fontsize=12.2)
        endpoint_axis.set_yticks(
            np.arange(len(endpoint_order)), endpoint_labels)
        endpoint_axis.invert_yaxis()
        endpoint_axis.set_xlim(1.0485, 1.060)
        endpoint_axis.set_xlabel(
            "Final normalised compliance\n(lower is better)", fontsize=11)
        endpoint_axis.grid(True, axis="x", color="0.88", linewidth=0.7)
        endpoint_axis.grid(False, axis="y")
        for axis in (full_axis, zoom_axis, endpoint_axis):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=9.8)
        figure.subplots_adjust(
            left=0.13, right=0.985, bottom=0.12, top=0.96)
        return self._save_figure(
            figure, self.figure_dir / "fig_09_ellipse_verlauf_en.png")

    def plot_lbracket_results(self, results: dict[str, Any]) -> Path:
        methods = results.get("example2", {})
        if not methods.get("pisco_binary") or not methods.get("simp"):
            raise RuntimeError(
                "Example-2 binary PISCO and SIMP results are required")
        problem = solvers.make_publication_lbracket_problem()
        mask = solvers._l_bracket_mask(problem.nelx, problem.nely)
        pisco = self._best(methods["pisco_binary"])
        simp = methods["simp"][0]
        raw_simp = np.asarray(
            simp.extra.get("raw_design", simp.best_design), dtype=float)
        common_simp = solvers.common_filtered_design(raw_simp, problem, mask)
        binary_simp = solvers.volume_preserving_binary(
            common_simp, mask, problem.volfrac)
        force, fixed = solvers._l_bracket_load(problem.nelx, problem.nely)
        _, simp_compliance, _, _ = solvers._fea(
            problem.nelx, problem.nely, binary_simp, fixed, force,
            penal=problem.penal, E0=1.0, Emin=1e-9, nu=0.30)

        figure = plt.figure(figsize=(12.5, 8.2))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.0, 1.05), hspace=0.3)
        for cell, density, compliance, fem_calls, title in (
                (grid[0, 0], binary_simp, simp_compliance,
                 simp.fem_calls_total, "SIMP, common binary evaluation"),
                (grid[0, 1], pisco.best_design, pisco.best_compliance,
                 pisco.fem_calls_total,
                 "PISCO binary objective (best seed)")):
            axis = figure.add_subplot(cell)
            field = np.asarray(density, dtype=float).reshape(
                problem.nely, problem.nelx).copy()
            field[~mask] = np.nan
            axis.imshow(
                field, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            axis.set_title(
                f"{title}\n$J={compliance:.3f}$, FEM={fem_calls}")
            axis.set_xlabel(r"Element $x$")
            axis.set_ylabel(r"Element $y$")

        active_runs = []
        anchor_calls = []
        for result in methods["pisco_binary"]:
            start = result.extra.get("binary_objective_start_fem")
            if start is None:
                first_cycle = result.extra.get("cycle_metrics", [{}])[0]
                first_cycle_fem = first_cycle.get("fem_calls")
                if first_cycle_fem is None:
                    raise RuntimeError(
                        "Binary-objective result lacks its phase boundary")
                start = (first_cycle_fem
                         - solvers.make_publication_pisco_e2_config().n_top)
            start = int(start)
            anchor_calls.append(start)
            trajectory = [
                (call - start, value)
                for call, value in result.trajectory if call > start]
            active_runs.append(SimpleNamespace(
                trajectory=trajectory,
                fem_calls_total=trajectory[-1][0], extra={}))
        calls, matrix = self._trajectory_matrix(active_runs)
        mean = matrix.mean(axis=0)
        deviation = matrix.std(axis=0, ddof=1)
        axis = figure.add_subplot(grid[1, :])
        axis.plot(
            calls, mean, color="C1", linewidth=2,
            label="PISCO binary-objective mean")
        axis.fill_between(
            calls, mean - deviation, mean + deviation,
            color="C1", alpha=0.2,
            label=r"$\pm 1$ standard deviation")
        axis.axhline(
            simp_compliance, color="C0", linewidth=2, linestyle="--",
            label=(f"SIMP binary reference "
                   f"({simp.fem_calls_total} FEM calls)"))
        total_calls = [
            result.fem_calls_total for result in methods["pisco_binary"]]
        axis.text(
            0.99, 0.97,
            f"PISCO anchor generation: {min(anchor_calls)}--"
            f"{max(anchor_calls)} FEM\n"
            f"PISCO total: {min(total_calls)}--{max(total_calls)} FEM",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.9})
        axis.set_xlabel(
            "Binary-objective FEM validations after anchor generation")
        axis.set_ylabel(r"Best-so-far compliance $J$")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.suptitle(
            "Example 2: L-bracket topology optimisation", fontsize=14)
        return self._save_figure(
            figure, self.figure_dir / "fig_11_lbracket_resultate_en.png")

    @staticmethod
    def _fin_background(axis: plt.Axes, problem: Any) -> None:
        domain, skin, design = problem.geometry()
        image = np.ones((*domain.shape, 3), dtype=float)
        image[design] = np.asarray([0.82, 0.88, 0.95])
        image[skin] = np.asarray([0.34, 0.39, 0.45])
        axis.imshow(
            image, origin="lower", interpolation="nearest",
            extent=(0, problem.length, 0, problem.height))
        hx, hy = problem.length / problem.nelx, problem.height / problem.nely
        axis.set_xlim(-1.5 * hx, problem.length + 1.5 * hx)
        axis.set_ylim(-3.0 * hy, problem.height + 3.0 * hy)
        axis.set_aspect("equal")
        axis.axis("off")

    @staticmethod
    def _fin_support_and_load(axis: plt.Axes, problem: Any,
                              *, annotate: bool = False) -> None:
        _domain, skin, _design = problem.geometry()
        hx, hy = problem.length / problem.nelx, problem.height / problem.nely
        pivot_x, pivot_y = 0.13 * problem.length, 0.50 * problem.height
        radius = 0.09 * problem.length
        axis.add_patch(Circle(
            (pivot_x, pivot_y), radius, facecolor="0.88",
            edgecolor="0.08", hatch="///", linewidth=1.6, zorder=7))
        load_x = 0.55 * problem.length
        axis.plot(
            [load_x, load_x], [2.0 * hy, problem.height - hy],
            color="#c62828", linewidth=1.1, linestyle="--",
            alpha=0.85, zorder=6)
        for offset in (-2.4, -0.8, 0.8, 2.4):
            column = int(np.clip(
                round(load_x / hx + offset - 0.5), 0, problem.nelx - 1))
            rows = np.flatnonzero(skin[:, column])
            if not rows.size:
                continue
            top, bottom = float(rows.max() + 1) * hy, float(rows.min()) * hy
            # All forces in FinProblem.load_and_support are downward, on
            # both skins. The arrows illustrate that common direction.
            for start, end in ((top + 2.4 * hy, top - 1.6 * hy),
                               (bottom + 1.6 * hy, bottom - 2.4 * hy)):
                axis.annotate(
                    "", xy=(load_x + offset * hx, end),
                    xytext=(load_x + offset * hx, start),
                    arrowprops={
                        "arrowstyle": "-|>", "color": "#c62828",
                        "lw": 2.2, "mutation_scale": 14}, zorder=8)
        if annotate:
            axis.annotate(
                "fixed pivot bearing\n(both displacements constrained)",
                xy=(pivot_x, pivot_y - radius),
                xytext=(0.48 * problem.length, 0.13 * problem.height),
                ha="left", va="top", fontsize=11, color="0.08",
                arrowprops={"arrowstyle": "-", "color": "0.15",
                            "lw": 1.0})
            axis.annotate(
                "downward unit load\n" + r"near $x=0.55L_x$ from root",
                xy=(load_x, problem.height - 4.0 * hy),
                xytext=(0.65 * problem.length, problem.height - 5.0 * hy),
                ha="left", va="bottom", fontsize=13, color="#a61b1b",
                arrowprops={"arrowstyle": "-", "color": "#c62828",
                            "lw": 1.2})

    def plot_fin_setup(self) -> Path:
        problem = solvers.FinProblem()
        figure = plt.figure(figsize=(12.0, 8.2))
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.0))
        axes = (
            figure.add_subplot(grid[0, :]),
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]))
        self._fin_background(axes[0], problem)
        self._fin_support_and_load(axes[0], problem, annotate=True)
        axes[0].set_title(
            "(a) Structural domain, support, and loading",
            fontsize=16, pad=10)
        for axis, dimension, title in (
                (axes[1], 35, r"(b) Coarse parameterisation ($d=35$)"),
                (axes[2], 243, r"(c) Fine parameterisation ($d=243$)")):
            self._fin_background(axis, problem)
            self._fin_support_and_load(axis, problem)
            controls = solvers.GaussianFinParameterization(
                problem, dimension).controls
            axis.scatter(
                (controls[:, 0] + 0.5) * problem.length / problem.nelx,
                (controls[:, 1] + 0.5) * problem.height / problem.nely,
                s=24 if dimension == 35 else 9, color="#0b559f",
                edgecolors="white", linewidths=0.25, zorder=9)
            axis.set_title(title, fontsize=16, pad=10)
        legend = [
            Patch(facecolor="0.34", edgecolor="0.34",
                  label=r"passive solid skin ($\rho=1$)"),
            Patch(facecolor=(0.82, 0.88, 0.95), edgecolor="0.45",
                  label=r"optimisable interior ($0\leq\rho\leq1$)"),
            Patch(facecolor="0.88", edgecolor="0.08", hatch="///",
                  label="fixed pivot bearing"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#0b559f", markeredgecolor="white",
                   markersize=7, label="Gaussian design controls"),
            Line2D([0], [0], color="#c62828", linewidth=2.2,
                   marker=r"$\blacktriangledown$", markersize=8,
                   label=r"downward load near $x=0.55L_x$"),
        ]
        figure.legend(
            handles=legend, loc="lower center", ncol=3, frameon=False,
            fontsize=14, bbox_to_anchor=(0.5, 0.005), handlelength=2.2,
            columnspacing=1.8)
        figure.tight_layout(
            rect=(0.0, 0.13, 1.0, 1.0), h_pad=1.2, w_pad=1.8)
        return self._save_figure(
            figure, self.figure_dir / "fig_13_fin_setup_en.png")

    def plot_fin_results(self, results: dict[str, Any]) -> Path:
        methods = results.get("example3", {})
        required = [
            f"{prefix}_d{dimension}"
            for prefix in ("pisco", "ard_bo", "cmaes", "de", "random")
            for dimension in (35, 62, 135, 243)]
        missing = [key for key in required if key not in methods]
        if missing:
            raise RuntimeError(
                "Incomplete Example-3 result matrix: " + ", ".join(missing))

        figure = plt.figure(figsize=(10.8, 8.4))
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.0, 0.9), width_ratios=(1.12, 1.0),
            hspace=0.46, wspace=0.27)
        dimension_axis = figure.add_subplot(grid[0, :])
        ablation_axis = figure.add_subplot(grid[1, 0])
        design_axis = figure.add_subplot(grid[1, 1])
        dimensions = np.asarray([35, 62, 135, 243], dtype=float)
        styles = {
            "pisco": ("PISCO", "#087bb5", "o", "-"),
            "ard_bo": ("GP-BO (ARD)", "#d98c00", "s", "--"),
            "cmaes": ("CMA-ES", "#00896b", "^", "-."),
            "de": ("DE", "0.30", "D", (0, (5, 2))),
            "random": ("Random", "0.62", "X", ":"),
        }
        for prefix, (label, colour, marker, line_style) in styles.items():
            arrays = [
                self._values(methods[f"{prefix}_d{dimension}"])
                for dimension in dimensions.astype(int)]
            means = [float(values.mean()) for values in arrays]
            deviations = [float(values.std(ddof=1)) if len(values)>1 else 0.0 for values in arrays]
            dimension_axis.errorbar(
                dimensions, means, yerr=deviations, linewidth=2.0,
                capsize=3.8, markersize=7.0, markeredgecolor="0.12",
                markeredgewidth=0.6, label=label, color=colour,
                marker=marker, linestyle=line_style)
        for key, label, colour, line_style in (
                ("pisco_progressive", "Progressive PISCO", "#087bb5",
                 (0, (7, 2, 1.5, 2))),
                ("ard_bo_progressive", "Progressive ARD-BO", "#b66a00",
                 (0, (1, 1.5)))):
            if key not in methods:
                continue
            values = self._values(methods[key])
            mean = float(values.mean())
            deviation = float(values.std(ddof=1))
            dimension_axis.axhline(
                mean, color=colour, linestyle=line_style, linewidth=2.0,
                label=f"{label} ({mean:.1f}$\\pm${deviation:.1f})")
        dimension_axis.set_xticks(dimensions)
        dimension_axis.set_xlabel(
            r"number of design variables $d$ (150 FEM calls)")
        dimension_axis.set_ylabel(
            r"final compliance $J$ (mean $\pm$ 1 std, nine seeds)")
        dimension_axis.set_title("(a) dimension sweep")
        dimension_axis.grid(axis="y", alpha=0.25)
        dimension_axis.legend(
            loc="upper left", frameon=True, facecolor="white",
            edgecolor="none", framealpha=0.90, fontsize=8.2, ncol=4,
            columnspacing=1.0, handlelength=2.2, borderpad=0.25)

        ablations = (
            ("PISCO\nfield +\nequilibrium", "pisco_d35", "#087bb5", "o"),
            ("PISCO\nfield without\nequilibrium", "pisco_no_physics_d35",
             "#7da9cb", "s"),
            ("PISCO\nscalar\noutput", "scalar_surrogate_d35", "0.68", "^"),
            ("CMA-ES", "cmaes_d35", "#00896b", "D"),
            ("GP-BO", "ard_bo_d35", "#d98c00", "P"),
        )
        samples = [
            self._values(methods[key]) for _label, key, _colour, _marker
            in ablations]
        positions = np.arange(len(ablations))
        offsets = np.linspace(-0.14, 0.14, 9)
        for position, sample, (_label, _key, colour, marker) in zip(
                positions, samples, ablations):
            ablation_axis.scatter(
                position + offsets, sample, s=20, color=colour,
                marker=marker, edgecolor="0.15", linewidth=0.35,
                alpha=0.9, zorder=3)
        means = [float(sample.mean()) for sample in samples]
        deviations = [float(sample.std(ddof=1)) for sample in samples]
        ablation_axis.errorbar(
            positions, means, yerr=deviations, fmt="D", markersize=5.5,
            color="0.08", ecolor="0.08", elinewidth=1.4, capsize=4.0,
            label=r"mean $\pm$ 1 std", zorder=4)
        ablation_axis.set_xticks(
            positions, [label for label, *_ in ablations])
        ablation_axis.set_ylabel(
            r"final compliance $J$ at $d=35$ (nine seeds)")
        ablation_axis.set_title(
            r"(b) $d=35$: PISCO variants and baselines")
        ablation_axis.tick_params(axis="x", labelsize=7.6)
        ablation_axis.grid(axis="y", alpha=0.25)
        ablation_axis.set_axisbelow(True)
        ablation_axis.legend(
            loc="upper left", frameon=True, facecolor="white",
            edgecolor="none", framealpha=0.90, fontsize=8.0,
            borderpad=0.25, handlelength=1.5)
        for position, mean, deviation in zip(
                positions, means, deviations):
            ablation_axis.text(
                position, mean + deviation + 0.45,
                f"{mean:.1f} ± {deviation:.1f}",
                ha="center", va="bottom", fontsize=9.2,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.75, "pad": 0.8})

        ordered = sorted(
            methods["pisco_d35"],
            key=lambda result: (result.best_compliance, result.seed))
        representative = ordered[len(ordered) // 2]
        problem = solvers.FinProblem()
        _domain, _skin, design = problem.geometry()
        density = np.asarray(representative.best_design, dtype=float)
        displayed = np.full_like(density, np.nan)
        displayed[design] = density[design]
        self._fin_background(design_axis, problem)
        colour_map = plt.get_cmap("Greys").copy()
        colour_map.set_bad((1.0, 1.0, 1.0, 0.0))
        density_image = design_axis.imshow(
            displayed, origin="lower", interpolation="nearest",
            extent=(0, problem.length, 0, problem.height), cmap=colour_map,
            vmin=0.0, vmax=1.0)
        contour_density = np.ma.masked_where(~design, density)
        design_axis.contour(
            contour_density, levels=[0.20, 0.40, 0.60], origin="lower",
            extent=(0, problem.length, 0, problem.height),
            colors=["0.12", "0.38", "0.78"],
            linestyles=[":", "-", "--"], linewidths=[1.1, 1.3, 1.1])
        contour_legend = [
            Line2D([0], [0], color="0.12", linestyle=":", linewidth=1.3,
                   label=r"$\rho=0.20$"),
            Line2D([0], [0], color="0.38", linestyle="-", linewidth=1.5,
                   label=r"$\rho=0.40$"),
            Line2D([0], [0], color="0.78", linestyle="--", linewidth=1.3,
                   label=r"$\rho=0.60$"),
        ]
        self._fin_support_and_load(design_axis, problem)
        design_axis.legend(
            handles=contour_legend, title="density contours",
            loc="lower left", fontsize=8.2, title_fontsize=8.5, ncol=3,
            frameon=True, facecolor="white", edgecolor="0.75",
            framealpha=0.9, handlelength=2.2, columnspacing=0.8,
            borderpad=0.35)
        design_axis.set_title(
            "(c) Density in the FEM-verified end design\n"
            f"median seed {representative.seed}, $d=35$, "
            f"$J={representative.best_compliance:.2f}$, $V_f=0.40$",
            fontsize=10.5)
        colour_bar = figure.colorbar(
            density_image, ax=design_axis, orientation="horizontal",
            fraction=0.055, pad=0.04, shrink=0.72)
        colour_bar.set_label(
            r"density $\rho$ (white: void; black: solid)")
        colour_bar.set_ticks([0.0, 0.40, 1.0])
        figure.subplots_adjust(
            left=0.075, right=0.985, bottom=0.08, top=0.93)
        return self._save_figure(
            figure, self.figure_dir / "fig_14_fin_results_en.png")

    @staticmethod
    def _boxes(axis, boxes: list[tuple[float, str]], *, title: str) -> None:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        width = 0.17
        for index, (center, label) in enumerate(boxes):
            axis.add_patch(Rectangle(
                (center - width / 2, 0.39), width, 0.22,
                facecolor="white", edgecolor="black", linewidth=1.2))
            axis.text(center, 0.50, label, ha="center", va="center", fontsize=9)
            if index:
                previous = boxes[index - 1][0]
                axis.annotate(
                    "", xy=(center - width / 2, 0.50),
                    xytext=(previous + width / 2, 0.50),
                    arrowprops=dict(arrowstyle="->", linewidth=1.1))
        axis.set_title(title, fontsize=11)

    def plot_workflow(self) -> Path:
        return self._render_typst_figure(
            "wrap_workflow_detail_en.typ", "schema_workflow_detail_en.pdf")

    def plot_wiring(self) -> Path:
        return self._render_typst_figure(
            "wrap_verschaltung_en.typ", "schema_verschaltung_en.pdf")

    def plot_ellipse_setup(self) -> Path:
        return self._render_typst_figure(
            "wrap_ellipse_setup_en.typ", "ellipse_setup_en.pdf")

    def plot_graphical_abstract(self) -> Path:
        return self._render_typst_figure(
            "wrap_workflow_kompakt_en.typ", "graphical_abstract.pdf")

    def _render_typst_figure(self, wrapper_name: str,
                             output_name: str) -> Path:
        """Render an approved vector diagram from its editable Typst source."""
        support_root = ROOT / "paper_support"
        source = support_root / "schemas" / wrapper_name
        output = self.figure_dir / output_name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing editable figure source: {source}")
        subprocess.run(
            ["typst", "compile", "--root", str(support_root),
             str(source), str(output)],
            check=True)
        jpeg_path = output.with_suffix(".jpeg")
        temporary_root = output.with_suffix("")
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "700",
             "-jpeg", "-jpegopt",
             "quality=100,optimize=y,progressive=n",
             str(output), str(temporary_root)],
            check=True)
        generated_jpeg = temporary_root.with_suffix(".jpg")
        _atomic_bytes(jpeg_path, generated_jpeg.read_bytes())
        generated_jpeg.unlink()
        return output

    def render_figures(self, results: dict[str, Any]) -> list[Path]:
        for legacy_name in self.LEGACY_GENERATED_FIGURES:
            legacy_path = self.figure_dir / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()
        primary = [
            self.plot_physics_link(results),
            self.plot_ellipse_results(results),
            self.plot_ellipse_trajectories(results),
            self.plot_lbracket_results(results),
            self.plot_fin_setup(),
            self.plot_fin_results(results),
            self.plot_workflow(),
            self.plot_wiring(),
            self.plot_ellipse_setup(),
            self.plot_graphical_abstract(),
        ]
        companions = []
        for path in primary:
            companions.append(path.with_suffix(".jpeg"))
        submission_dir = self.figure_dir / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        for legacy_name in self.LEGACY_SUBMISSION_FIGURES:
            legacy_path = submission_dir / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()
        submission_paths = []
        for source_name, target_name in self.SUBMISSION_FIGURE_MAP:
            source = self.figure_dir / source_name
            target = submission_dir / target_name
            _atomic_bytes(target, source.read_bytes())
            submission_paths.append(target)
        return [*primary, *companions, *submission_paths]

    def install_private_manuscript_assets(self) -> list[Path]:
        """Install rebuilt data figures when run inside the private revision."""
        submission_root = ROOT.parent / "submission"
        if not (submission_root / "PISCO_en.tex").is_file():
            return []
        generated_root = ROOT.parent / "generated" / "paper_artifacts"
        upload_root = submission_root
        installed: list[Path] = []
        for name in self.DATA_FIGURE_NAMES:
            source = self.figure_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing rebuilt figure: {source}")
            target = generated_root / name
            _atomic_bytes(target, source.read_bytes())
            installed.append(target)
        for legacy_name in self.LEGACY_SUBMISSION_FIGURES:
            legacy_path = upload_root / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()
        for _, upload_name in self.SUBMISSION_FIGURE_MAP:
            source = self.figure_dir / "submission" / upload_name
            target = upload_root / upload_name
            _atomic_bytes(target, source.read_bytes())
            installed.append(target)
        self.write_submission_assets_manifest()
        print(
            "Installed rebuilt manuscript and journal-upload figures in the "
            "private submission directory.")
        return installed

    def write_manifest(self, paths: list[Path]) -> Path:
        manifest_path = self.output_root / "reproduction_manifest.json"
        records = []
        for path in sorted(set(paths)):
            records.append({
                "path": path.relative_to(self.output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        document = {
            "protocol": solvers.PUBLICATION_PROTOCOL,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "solvers_sha256": _sha256(ROOT / "solvers.py"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "artifacts": records,
        }
        _atomic_text(manifest_path, json.dumps(
            document, indent=2, sort_keys=True) + "\n")
        return manifest_path

    def build_all(self, results: dict[str, Any] | None = None) -> list[Path]:
        if results is None:
            results = self.load_results()
        summaries = self.summaries(results)
        paths = self.write_tables(summaries)
        paths.extend(self.write_editable_table_bundle())
        paths.extend(self.render_figures(results))
        paths.append(self.bundle_path)
        manifest = self.write_manifest(paths)
        # Installing into the manuscript is an explicit editorial step.
        return [*paths, manifest]

    def verify(self) -> int:
        failures = _verify_frozen_claim_rows()
        comparison_failures = _verify_frozen_comparisons()
        failures.extend(comparison_failures)
        if not failures:
            print(
                "OK  frozen publication record: 94 claim rows and "
                "22 statistical comparisons recomputed")
        table_path = self.table_dir / "publication_results.json"
        manifest_path = self.output_root / "reproduction_manifest.json"
        has_local_build = any((
            self.bundle_path.exists(), table_path.exists(),
            manifest_path.exists()))
        if not has_local_build:
            if failures:
                print("\nVerification failed:")
                for failure in failures:
                    print(f"  - {failure}")
                return 1
            print(
                "No local solver build is present; run --all to add the "
                "solver-to-reference and artifact-hash checks.")
            return 0
        if not table_path.is_file():
            failures.append(f"missing {table_path}")
        else:
            document = json.loads(table_path.read_text(encoding="utf-8"))
            for key, (reference, tolerance) in REFERENCE_MEANS.items():
                example, method = key.split("/", 1)
                entry = document.get("examples", {}).get(example, {}).get(method)
                if entry is None:
                    # Partial/example-specific reproductions are valid builds;
                    # only values present in them can be checked.
                    continue
                expected_runs = (
                    3 if method in THREE_SEED_REFERENCE_METHODS else 9)
                if int(entry.get("n", 0)) != expected_runs:
                    print(
                        f"PARTIAL  {key}: {entry.get('n')} run(s); the frozen "
                        f"mean uses {expected_runs} seeds")
                    continue
                actual = float(entry["mean"])
                if abs(actual - reference) > tolerance:
                    failures.append(
                        f"{key}: {actual:.8g}, expected "
                        f"{reference:.8g} ± {tolerance:g}")
                else:
                    print(f"OK  {key}: {actual:.8g}")
        if self.bundle_path.is_file():
            saved_results = self.load_results()
            projection_runs = saved_results.get("example2", {}).get(
                "pca_projection_study", [])
            if projection_runs:
                projection = projection_runs[0].extra
                projection_checks = {
                    "example2/SIMP direct binary": (
                        projection.get("simp_direct_binary_compliance"),
                        69.32671948818714, 0.01),
                    "example2/PCA anchors40 iters6": (
                        projection.get("projections", {}).get(
                            "anchors40_iters6", {}).get(
                                "projected_simp_binary_compliance"),
                        70.49374095552662, 0.01),
                }
                for label, (actual, reference, tolerance) in (
                        projection_checks.items()):
                    if actual is None or abs(actual - reference) > tolerance:
                        failures.append(
                            f"{label}: {actual!r}, expected "
                            f"{reference:.8g} ± {tolerance:g}")
                    else:
                        print(f"OK  {label}: {actual:.8g}")
        if not manifest_path.is_file():
            failures.append(f"missing {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("artifacts", []):
                path = self.output_root / record["path"]
                if not path.is_file():
                    failures.append(f"missing artifact {path}")
                elif path.stat().st_size != record["size_bytes"]:
                    failures.append(f"size mismatch {path}")
                elif _sha256(path) != record["sha256"]:
                    failures.append(f"hash mismatch {path}")
            present = {
                Path(record["path"]).name
                for record in manifest.get("artifacts", [])}
            missing_figures = set(self.FIGURE_NAMES) - present
            if missing_figures:
                failures.append(
                    "figures absent from manifest: "
                    + ", ".join(sorted(missing_figures)))
        if failures:
            print("\nVerification failed:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nVerification successful: values and artifact hashes agree.")
        return 0


def _parse_seeds(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    unique = list(dict.fromkeys(int(value) for value in values))
    if not unique:
        raise ValueError("At least one seed is required")
    return unique


def scientific_outputs(builder, results):
    """Manuscript evidence computed from this run, including all 94 R1 mappings.

    Historical rows supply identifiers and comparison values only. No frozen
    value is substituted for a missing current result. This also runs on a
    partial/smoke bundle, marking unavailable populations explicitly.
    """
    from scipy.stats import spearmanr, t as student_t
    def summary(values):
        values = np.asarray(values, float)
        n = len(values)
        mean = float(values.mean()) if n else None
        sd = float(values.std(ddof=1)) if n > 1 else None
        half = float(student_t.ppf(.975, n-1)*sd/math.sqrt(n)) if n > 1 else None
        return {"n":n, "values":values.tolist(), "mean":mean, "std":sd,
                "ci95_mean":None if half is None else [mean-half, mean+half]}
    def runs(family, method):
        return sorted(results.get(family, {}).get(method, []), key=lambda run:run.seed)
    def source(family, method, seed):
        return f"{family}/{method}/seed_{seed}/run.json"
    claims = {}
    def put(identifier, values, seeds, sources, definition, status="available"):
        claims[identifier] = {"id":identifier, **summary(values), "seeds":seeds,
                              "sources":sources, "definition":definition, "status":status}
    aliases = {
        "e1_pisco_lam0":"pisco_lambda_0", "e1_pisco_lam0_001":"pisco_lambda_0.001",
        "e1_pisco_lam0_1":"pisco_lambda_0.1", "e3_sklearn_bo_d35":"ard_bo_d35",
        "e3_scalar":"scalar_surrogate_d35", "e3_no_physics":"pisco_no_physics_d35",
        "e3_nested_pisco":"pisco_progressive", "e3_nested_bo":"ard_bo_progressive"}
    e2_aliases = {"pisco_pca":"pisco_dynamic_pca", "pisco_pca_static":"pisco_static_pca",
        "pisco_pca_fixed_budget":"pisco_dynamic_pca_fixed_cycles",
        "pisco_pca_static_fixed_budget":"pisco_static_pca_fixed_cycles",
        "pisco_vae":"pisco_vae", "pisco_ae":"pisco_ae", "pisco_vae_nophys":"pisco_vae_lambda_0",
        "pisco_vae_strong":"pisco_vae_lambda_1", "pisco_vae_purephys":"pisco_vae_data_lambda_0",
        "scalar_surrogate":"scalar_surrogate", "no_reeval":"no_periodic_verification",
        "latent_fem":"latent_fem", "simp_free_init":"pisco_random_basis",
        "pisco_binary_objective":"pisco_binary", "pisco_strong_anchors":"pisco_strong_anchors",
        "latent_cma_es":"latent_cmaes", "latent_ard_bo":"latent_ard_bo"}
    for old in FROZEN_CLAIM_ROWS:
        identifier = old["id"]; family = "example"+identifier[1]
        method = aliases.get(identifier)
        if identifier.startswith("e1_fixed156_"):
            method = identifier.removeprefix("e1_fixed156_")+"_fixed_156"
        elif identifier.startswith("e1_") and method is None:
            method = identifier[3:]
        elif identifier.startswith("e3_") and method is None:
            method = identifier[3:].replace("bo_ard_", "ard_bo_")
        elif identifier.startswith("e2_"):
            method = e2_aliases.get(identifier[3:])
        population = runs(family, method)
        if population:
            put(identifier, builder._values(population), [r.seed for r in population],
                [source(family, method, r.seed) for r in population],
                "FEM endpoint; E1 normalised by its current solid reference; other examples absolute compliance")

    # Full measured costs are retained; no runtime is invented for a truncated
    # budget endpoint or for repetitions of the deterministic SIMP reference.
    costs = []
    for family, methods in results.items():
        for method, population in methods.items():
            for run in population:
                audit = run.extra.get("r2_evidence", {})
                costs.append({"family":family, "method":method, "seed":run.seed,
                    "wall_time_s":run.wall_time_s, "logical_search_calls":run.fem_calls_total,
                    "recorded_fem_calls":len(audit.get("fem_events", [])),
                    "exclusive_seconds":audit.get("exclusive_seconds", {}),
                    "construction":audit.get("construction"),
                    "raw_run_fem_calls":run.extra.get("raw_run_fem_calls"),
                    "source":source(family, method, run.seed)})
    for identifier, method, metric in (
        ("e2_pisco_wall_time", "pisco_dynamic_pca", "wall_time_s"),
        ("e2_simp_wall_time", "simp", "wall_time_s"),
        ("e2_surrogate_training_s", "pisco_dynamic_pca", "surrogate_training_s"),
        ("e2_latent_optimisation_s", "pisco_dynamic_pca", "latent_optimisation_s"),
        ("e2_active_fem_s", "pisco_dynamic_pca", "active_fem_s")):
        population = runs("example2", method)
        if population:
            put(identifier, [r.wall_time_s if metric=="wall_time_s" else r.extra["timing_breakdown_s"][metric]
                              for r in population], [r.seed for r in population],
                [source("example2", method, r.seed) for r in population],
                "Measured seconds; component stopwatches may include nested work; additive costs use exclusive_seconds. Single runs have no SD.")

    # Re-evaluation is accounted for separately from optimisation budgets.
    audit = solvers.RunAudit(); token = solvers._ACTIVE_AUDIT.set(audit)
    designs = []
    try:
        with solvers.audit_scope("manuscript_postprocessing"):
            problem = solvers.make_publication_lbracket_problem()
            mask = solvers._l_bracket_mask(problem.nelx, problem.nely)
            force, fixed = solvers._l_bracket_load(problem.nelx, problem.nely)
            for method in ("pisco_dynamic_pca", "simp"):
                for run in runs("example2", method):
                    rho = solvers.common_filtered_design(np.asarray(run.extra["raw_design"]), problem, mask)
                    binary = solvers.volume_preserving_binary(rho, mask, problem.volfrac)
                    jc = solvers._fea(problem.nelx, problem.nely, rho, fixed, force, penal=problem.penal)[1]
                    jb = solvers._fea(problem.nelx, problem.nely, binary, fixed, force, penal=problem.penal)[1]
                    designs.append({"method":method, "seed":run.seed,
                        "volume_fraction":float(rho[mask].mean()), "grayness":float(4*np.mean(rho[mask]*(1-rho[mask]))),
                        "common_filtered_compliance":jc, "volume_preserving_binary_compliance":jb,
                        "binary_volume_fraction":float(binary[mask].mean()), "source":source("example2", method, run.seed)})
    finally:
        solvers._ACTIVE_AUDIT.reset(token)
    for prefix, method in (("e2_design_", "pisco_dynamic_pca"), ("e2_simp_design_", "simp")):
        rows = [r for r in designs if r["method"]==method]
        if rows:
            for metric in ("volume_fraction", "grayness", "common_filtered_compliance", "volume_preserving_binary_compliance"):
                put(prefix+metric, [r[metric] for r in rows], [r["seed"] for r in rows],
                    [r["source"] for r in rows], "One common filter and volume correction; continuous and rank-binary metrics are distinct")
        put(prefix+"min_feature_size_elements", [], [], [],
            "Removed: the historical global erosion-survival statistic is not minimum strut thickness", "removed_invalid_interpretation")
    for method in ("pisco_binary", "simp"):
        population = runs("mesh_refinement", method)
        if population:
            put("e2_fine_mesh_"+("pisco" if method=="pisco_binary" else "simp"),
                [r.extra["mesh_refinement"][1]["compliance"] for r in population],
                [r.seed for r in population], [source("mesh_refinement", method, r.seed) for r in population],
                "Replaces the historical continuous interpolation with geometry-preserving binary refinement", "replaced_by_binary_metric")
    decoder = [(r, r.extra.get("decoder_diagnostics_all_candidates", {})) for r in runs("example2", "pisco_dynamic_pca")]
    decoder = [(r, d) for r, d in decoder if d.get("n", 0)]
    if decoder:
        for suffix, key, reduce in (("raw_rho_min_global", "raw_min", min), ("raw_rho_max_global", "raw_max", max),
            ("raw_clipped_fraction_mean", "clipped_fraction_mean", np.mean),
            ("projected_volume_max_abs_error", "projected_volume_max_abs_error", max)):
            put("e2_decoder_"+suffix, [float(reduce([d[key] for r,d in decoder]))],
                [r.seed for r,d in decoder], [source("example2", "pisco_dynamic_pca", r.seed) for r,d in decoder],
                "All generated candidates; clipping is the unweighted mean of per-run means; n=1 is a global aggregate")
    for run in runs("example2", "pca_projection_study"):
        for key, values in run.extra["projections"].items():
            put("e2_projected_simp_"+key, [values["projected_simp_binary_compliance"]], [run.seed],
                [source("example2", "pca_projection_study", run.seed)], "Frozen representation diagnostic; not a performance bound")

    learning = []
    fixed_runs = runs("example2", "pisco_dynamic_pca_fixed_cycles")
    for run in fixed_runs:
        for cycle in run.extra.get("cycle_metrics", []):
            if cycle["cycle"] not in (1, 12) or "holdout_true_compliance" not in cycle:
                continue
            true, pred = np.asarray(cycle["holdout_true_compliance"]), np.asarray(cycle["holdout_predicted_compliance"])
            reps = np.asarray(cycle["holdout_representations"])
            for representation in ("continuous", "binary", "pooled"):
                keep = np.ones(len(true), bool) if representation=="pooled" else reps==representation
                a, b = true[keep], pred[keep]
                if not len(a): continue
                learning.append({"seed":run.seed, "cycle":cycle["cycle"], "representation":representation,
                    "n_designs":len(a), "rel_error_mean":float(np.mean(abs(a-b)/np.maximum(abs(a), 1e-12))),
                    "spearman":float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None,
                    "top_k_overlap":solvers.top_k_overlap(a,b,cycle["holdout_top_k"]),
                    "equilibrium_residual_mean":cycle.get("holdout_equilibrium_residual_mean") if representation=="pooled" else None,
                    "source":source("example2", "pisco_dynamic_pca_fixed_cycles", run.seed)})
    for metric in ("rel_error_mean", "spearman", "top_k_overlap", "equilibrium_residual_mean"):
        representation = "pooled" if metric=="equilibrium_residual_mean" else "continuous"
        valid = set.intersection(*[{r["seed"] for r in learning if r["cycle"]==cycle
                   and r["representation"]==representation and r[metric] is not None} for cycle in (1,12)])
        for cycle in (1,12):
            selected = [r for r in learning if r["cycle"]==cycle and r["representation"]==representation and r["seed"] in valid]
            if fixed_runs:
                key = f"e2_cycle{cycle}_holdout_{metric}"
                reached = all(any(c["cycle"]==12 for c in run.extra.get("cycle_metrics", [])) for run in fixed_runs)
                put(key, [r[metric] for r in selected], [r["seed"] for r in selected], [r["source"] for r in selected],
                    "Same seeds with defined values at cycles 1 and 12; continuous holdout except pooled equilibrium residual",
                    "available" if selected else "undefined_metric" if reached else "missing_cycle12")
                claims[key]["excluded_seeds"] = sorted(set(r.seed for r in fixed_runs)-valid)
                claims[key]["cohort_n"] = len(fixed_runs)

    # Reuse the pre-existing R1 comparison definitions, with CURRENT vectors.
    e3_ids = [r["id"] for r in FROZEN_CLAIM_ROWS if r["id"].startswith("e3_")]
    e3 = []
    try:
        vectors = {key:claims[key]["values"] for key in e3_ids if key in claims and claims[key]["n"]>=2}
        e3 = _recompute_e3_comparisons(vectors)
    except KeyError:
        pass  # partial/smoke bundle: never replace a missing current vector by R1
    tests = {"example1":[], "example2_exploratory":[], "example3":e3,
             "source_vectors_e3":{key:{k:claims[key][k] for k in ("values","seeds","sources")}
                                  for key in e3_ids if key in claims}}
    def contrast(family, left, right, group, paired, output):
        a, b = runs(family,left), runs(family,right)
        if min(len(a),len(b)) < 2: return
        if paired and [r.seed for r in a] != [r.seed for r in b]: return
        av, bv = builder._values(a), builder._values(b)
        function = _paired_comparison if paired else _welch_comparison
        row = function(av,bv,label=left+" vs "+right,family=group)
        row.update(left=left,right=right,seeds_a=[r.seed for r in a],seeds_b=[r.seed for r in b],
                   sources=[source(family, method, r.seed) for method, population in ((left,a),(right,b)) for r in population])
        if paired:
            def anchors(run):
                events=run.extra.get("r2_evidence",{}).get("fem_events",[])
                return [(v["design_sha256"],v["compliance"]) for v in events if v["role"]=="initial_data"]
            row["matching_initial_data_pairs"] = sum(anchors(x)==anchors(y) for x,y in zip(a,b))
        tests[output].append(row)
    for suffix, group in (("", "native"), ("_fixed_156", "fixed156")):
        for other in ("de", "bo", "cmaes"):
            contrast("example1", "pisco"+suffix, other+suffix, group, False, "example1")
    for other in ("pisco_lambda_0", "pisco_lambda_0.001", "pisco_lambda_0.1"):
        contrast("example1", "pisco", other, "physics_weight", True, "example1")
    for left,right,group in (
        ("pisco_dynamic_pca","pisco_static_pca","pca_representation"),
        ("pisco_dynamic_pca_fixed_cycles","pisco_static_pca_fixed_cycles","pca_representation"),
        ("pisco_dynamic_pca_fixed_cycles","no_periodic_verification","matched_controls"),
        ("pisco_dynamic_pca_fixed_cycles","field_no_physics_matched","matched_controls"),
        ("pisco_dynamic_pca_fixed_cycles","scalar_surrogate","matched_controls"),
        ("pisco_binary","latent_cmaes","binary_baselines"), ("pisco_binary","latent_ard_bo","binary_baselines"),
        ("pisco_vae","pisco_vae_lambda_0","vae_loss"), ("pisco_vae","pisco_vae_lambda_1","vae_loss")):
        contrast("example2",left,right,group,True,"example2_exploratory")
    for output in ("example1", "example2_exploratory"):
        _holm_adjust(tests[output])
    tests["interpretation"] = "E3 retains the 22 historical test definitions/families; E1 cross-method tests use Welch; E2 contrasts are exploratory. Holm corrections are within the stated families. Historical numerical targets are not inputs."
    table = []
    for old in FROZEN_CLAIM_ROWS:
        row = claims.get(old["id"], {"id":old["id"], **summary([]), "status":"missing_current_evidence", "seeds":[], "sources":[]})
        row = {**row, "historical_mean":old["mean"], "historical_n":old["n"]}
        table.append(row)
    binary_table = [{"method":method, "seed":run.seed, "coarse_compliance":run.extra["mesh_refinement"][0]["compliance"],
                    "fine_compliance":run.extra["mesh_refinement"][1]["compliance"],
                    "source":source("mesh_refinement",method,run.seed)}
                   for method in ("pisco_binary", "simp") for run in runs("mesh_refinement", method)]
    return {"claims":table, "statistical_tests":tests, "measured_costs":costs,
            "e2_design_metrics":designs, "e2_binary_main_comparison":binary_table,
            "e2_learning_by_seed":learning, "postprocessing_evidence":audit.snapshot(),
            "coverage":{"mapped_rows":len(table), "missing":[r["id"] for r in table if r["status"].startswith("missing")],
                        "undefined_metrics":[r["id"] for r in table if r["status"]=="undefined_metric"],
                        "e3_comparisons":len(e3)}}


def write_r2_documentation(builder, results):
    """Self-contained evidence; historical manuscript values are never inputs."""
    from scipy.stats import spearmanr, kendalltau, ttest_rel
    root = builder.output_root / "documentation"
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    def emit(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(path, json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n")
        paths.append(path)
    plan = solvers.expected_publication_experiments()
    present = {f"{family}/{method}": [int(r.seed) for r in runs]
               for family, methods in results.items() for method, runs in methods.items()}
    missing = {key: sorted(set(seeds)-set(present.get(key, []))) for key, seeds in plan.items()}
    missing = {key: seeds for key, seeds in missing.items() if seeds}
    errors = [f"{key}: unexpected publication seeds {sorted(set(seeds)-set(plan[key]))}"
              for key,seeds in present.items() if key in plan and set(seeds)-set(plan[key])]
    figure_failure=builder.output_root / "figure_generation_error.json"
    if figure_failure.is_file():
        errors.append("Publication figure generation failed: " + figure_failure.read_text())
    rows = []
    comparisons = []
    expected_source = hashlib.sha256(Path(solvers.__file__).read_bytes()).hexdigest()
    for family, methods in results.items():
        for method, runs in methods.items():
            if len({r.seed for r in runs}) != len(runs):
                errors.append(f"{family}/{method}: duplicate seeds")
            for run in runs:
                key = f"{family}/{method}/seed_{run.seed}"
                audit = run.extra.get("r2_evidence", {})
                if not audit or audit.get("configuration", {}).get("source_sha256") != expected_source:
                    errors.append(key + ": missing or stale provenance")
                configuration=audit.get("configuration",{}).get("config",{})
                if family == "example1" and method.startswith("pisco"):
                    expected={"n_init":60,"T_cycles":12,"n_top":8,"surrogate_steps":400}
                elif family == "example2" and "n_init" in configuration:
                    expected={"n_init":40,"T_cycles":12,"n_top":6}
                elif family == "example3":
                    expected={"budget":150,"n_init":20}
                else:
                    expected={}
                for parameter,value in expected.items():
                    if configuration.get(parameter)!=value:
                        errors.append(f"{key}: non-publication {parameter}={configuration.get(parameter)} (expected {value})")
                if not math.isfinite(run.best_compliance):
                    errors.append(key + ": nonfinite objective")
                if run.extra.get("binary_objective_candidates"):
                    density=np.asarray(run.best_design)
                    if not np.all((density==0.) | (density==1.)):
                        errors.append(key + ": binary objective returned a nonbinary design")
                relative = key + "/run.json"
                emit(relative, {"family":family, "method":method, **asdict(run)})
                design_path = root / key / "best_design.npy"
                if run.best_design is not None:
                    np.save(design_path, run.best_design, allow_pickle=False)
                    paths.append(design_path)
                predictions = audit.get("predictions", [])
                metrics = {"n":len(predictions), "population":"selected candidates before training inclusion; selection-biased, not independent test"}
                if predictions:
                    pred = np.asarray([v["predicted_before_fem"] for v in predictions])
                    true = np.asarray([v["fem_compliance"] for v in predictions])
                    metrics.update(mae=float(np.mean(abs(pred-true))), rmse=float(np.sqrt(np.mean((pred-true)**2))),
                                   mean_relative_error=float(np.mean(abs(pred-true)/np.maximum(abs(true),1e-12))))
                    # Rank only within a model update; pooling different models confounds ranking.
                    metrics["rank_by_update"] = []
                    for cycle in sorted({v["cycle"] for v in predictions}):
                        subset=[v for v in predictions if v["cycle"] == cycle]
                        a=np.asarray([v["predicted_before_fem"] for v in subset]); b=np.asarray([v["fem_compliance"] for v in subset])
                        valid=len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0
                        metrics["rank_by_update"].append({"update":cycle,"n":len(a),
                            "spearman":float(spearmanr(a,b).statistic) if valid else None,
                            "kendall":float(kendalltau(a,b).statistic) if valid else None})
                emit(key + "/accuracy.json", metrics)
                independent = list(audit.get("test_predictions", []))
                cycles=run.extra.get("cycle_metrics", [])
                if cycles and cycles[-1].get("holdout_true_compliance"):
                    last=cycles[-1]
                    independent.extend({"predicted":p,"actual":a,"representation":r}
                        for p,a,r in zip(last["holdout_predicted_compliance"],last["holdout_true_compliance"],
                                         last.get("holdout_representations",["continuous"]*len(last["holdout_true_compliance"]))))
                independent_metrics = {"n":len(independent), "raw":independent}
                if independent:
                    actual=np.asarray([v["actual"] for v in independent]); predicted=np.asarray([v["predicted"] for v in independent])
                    independent_metrics.update(mae=float(np.mean(abs(predicted-actual))),
                        rmse=float(np.sqrt(np.mean((predicted-actual)**2))),
                        mean_relative_error=float(np.mean(abs(predicted-actual)/np.maximum(abs(actual),1e-12))))
                independent_metrics["by_representation"]={}
                for representation in sorted({v["representation"] for v in independent}):
                    subset=[v for v in independent if v["representation"]==representation]
                    a=np.asarray([v["actual"] for v in subset]); p=np.asarray([v["predicted"] for v in subset])
                    valid=len(a)>1 and np.ptp(a)>0 and np.ptp(p)>0
                    k=min(5,len(a))
                    independent_metrics["by_representation"][representation]={"n":len(a),
                        "mae":float(np.mean(abs(p-a))),"rmse":float(np.sqrt(np.mean((p-a)**2))),
                        "mean_relative_error":float(np.mean(abs(p-a)/np.maximum(abs(a),1e-12))),
                        "spearman":float(spearmanr(a,p).statistic) if valid else None,
                        "kendall":float(kendalltau(a,p).statistic) if valid else None,
                        "top_k":k,"top_k_overlap":solvers.top_k_overlap(a,p,k),
                        "tie_policy":"undefined if a tie crosses either top-k cutoff"}
                emit(key + "/independent_test.json", independent_metrics)
                trace = []
                for call, value in run.trajectory:
                    matches=[item for item in audit.get("incumbents", [])
                             if item["logical_call"]==call and np.isclose(item["best_compliance"],value,rtol=1e-10)]
                    trace.append({"logical_call":int(call), "best_compliance":float(value),
                                  "recorded_elapsed_s":min(item["elapsed_s"] for item in matches) if matches else None})
                emit(key + "/trajectory.json", trace)
                emit(key + "/training.json", audit.get("training", []))
                emit(key + "/fem_events.json", audit.get("fem_events", []))
                # One browsable diagnostic figure per run; raw arrays remain authoritative.
                fig,axes=plt.subplots(1,3,figsize=(13,3.5))
                if trace and family != "mesh_refinement":
                    axes[0].plot([v["logical_call"] for v in trace],[v["best_compliance"] for v in trace])
                axes[0].set(xlabel="Logical FEM calls",ylabel="Best compliance")
                phases=audit.get("exclusive_seconds",{})
                axes[1].barh(list(phases),list(phases.values())); axes[1].set_xlabel("Exclusive seconds")
                if independent:
                    axes[2].scatter(actual,predicted,s=12,label="Independent final test")
                elif predictions:
                    axes[2].scatter(true,pred,s=12,label="Selected candidates")
                axes[2].set(xlabel="FEM compliance",ylabel="Predicted compliance")
                fig.suptitle(f"{family}/{method}, seed {run.seed}"); fig.tight_layout()
                plot=root / key / "diagnostics.png"; fig.savefig(plot,dpi=130); plt.close(fig); paths.append(plot)
                events = audit.get("fem_events", [])
                counts = {}
                for event in events:
                    role=event.get("role", "unspecified")
                    counts[role]=counts.get(role,0)+1
                row={"family":family,"method":method,"seed":run.seed,"compliance":float(run.best_compliance),
                     "normalized_compliance":run.best_compliance_normalized,
                     "logical_fem_calls":run.fem_calls_total,"recorded_fem_calls":len(events),
                     "fem_calls_by_role":counts,"duplicate_evaluations":sum(e.get("duplicate_design",False) for e in events),
                     "exclusive_seconds":audit.get("exclusive_seconds",{}),
                     "solver_wall_time_s":run.wall_time_s,"audit_elapsed_s":audit.get("elapsed_s"),
                     "evidence":relative,"accuracy":key + "/accuracy.json"}
                rows.append(row)
        if family != "mesh_refinement":
            for left,right in combinations(methods,2):
                a={r.seed:float(builder._values([r])[0]) for r in methods[left]}
                b={r.seed:float(builder._values([r])[0]) for r in methods[right]}
                seeds=sorted(a.keys() & b.keys())
                diff=np.asarray([a[k]-b[k] for k in seeds])
                valid=len(diff)>1 and np.std(diff)>0
                comparisons.append({"family":family,"left":left,"right":right,"paired_seeds":seeds,
                    "mean_difference":float(diff.mean()) if len(diff) else None,
                    "paired_t_pvalue":float(ttest_rel([a[k] for k in seeds],[b[k] for k in seeds]).pvalue) if valid else None,
                    "interpretation":"exploratory, unadjusted; confirmatory contrasts must be prespecified"})
    scientific=scientific_outputs(builder,results)
    for name,value in scientific.items():
        emit("manuscript/"+name+".json",value)
    for name in ("claims", "e2_design_metrics", "e2_binary_main_comparison", "e2_learning_by_seed"):
        data=scientific[name]
        if data:
            path=root / "manuscript" / (name+".csv")
            with path.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in data for k in row)))
                writer.writeheader(); writer.writerows(_json_value(data))
            paths.append(path)
    if not missing and (scientific["coverage"]["missing"] or scientific["coverage"]["e3_comparisons"]!=22):
        errors.append("Manuscript evidence incomplete: "+json.dumps(scientific["coverage"]))
    completeness={"full_publication_complete":not missing and not errors,"missing":missing,"errors":errors,
                  "expected":plan,"present":present}
    emit("completeness.json",completeness)
    emit("run_index.json",rows)
    emit("comparisons.json",comparisons)
    summaries=builder.summaries(results)
    for methods in summaries["examples"].values():
        for row in methods.values():
            if row["n"]<2: row["std"]=None; row["ci95_mean"]=None
    # Flat claim table for manuscript use, with paths back to the contributing runs.
    table=[]
    for family, methods in summaries["examples"].items():
        for method, values in methods.items():
            table.append({"family":family,"method":method,"n":values["n"],"mean":values["mean"],
                "std":values["std"],"ci95":values["ci95_mean"],"seeds":values["seeds"],
                "source_runs":[r["evidence"] for r in rows if r["family"]==family and r["method"]==method]})
    emit("claim_provenance.json",table)
    csv_path=root / "publication_results.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["family","method","n","mean","std","ci95","seeds","source_runs"])
        writer.writeheader(); writer.writerows(table)
    paths.append(csv_path)
    tex_path=root / "publication_results.tex"
    latex=[r"\begin{tabular}{lllrr}",r"Example & Method & $n$ & Mean & SD \\"]
    for row in table:
        name=row["method"].replace("_",r"\_")
        family=row["family"].replace("_",r"\_")
        deviation="--" if row["std"] is None else f"{row['std']:.6g}"
        latex.append(f"{family} & {name} & {row['n']} & {row['mean']:.8g} & {deviation}" + r" \\")
    latex.append(r"\end{tabular}")
    _atomic_text(tex_path,"\n".join(latex)+"\n"); paths.append(tex_path)
    emit("statistics.json",summaries)
    emit("mesh_refinement.json",{m:[{"seed":r.seed, **r.extra} for r in runs]
        for m,runs in results.get("mesh_refinement",{}).items()})
    for artifact in sorted(builder.figure_dir.rglob("*")):
        if artifact.is_file():
            target=root / "publication_figures" / artifact.relative_to(builder.figure_dir)
            target.parent.mkdir(parents=True,exist_ok=True)
            _atomic_bytes(target,artifact.read_bytes()); paths.append(target)
    plan_file=builder.output_root / "experiment_plan.json"
    if plan_file.is_file():
        emit("experiment_plan.json",json.loads(plan_file.read_text()))
    environment_file = builder.output_root / "reproduction_environment.json"
    if environment_file.is_file():
        emit("reproduction_environment.json", json.loads(environment_file.read_text()))
    postprocessing=builder.output_root / "figure_generation_evidence.json"
    if postprocessing.is_file():
        emit("figure_generation_evidence.json",json.loads(postprocessing.read_text()))
    for source in (Path(solvers.__file__), Path(__file__)):
        target=root / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        _atomic_bytes(target, source.read_bytes()); paths.append(target)
    import importlib.metadata
    emit("environment.json",{name:importlib.metadata.version(name) for name in
                             ("numpy","scipy","torch","matplotlib","scikit-learn","Pillow","threadpoolctl")})
    lines=["# PISCO Revision 2 – Ergebnisdokumentation", "",
        "Status: " + ("vollständige Publikationsmatrix" if completeness["full_publication_complete"] else "TEILLAUF / Prüfung offen"),
        "", "Alle Zahlen stammen aus den gespeicherten Läufen. R1-Referenzen sind keine R2-Ergebnisse.",
        "", "- [Vollständigkeit und fehlende Seeds](completeness.json)",
        "- [Aktuelle Manuskriptstatistik und Holm-Testfamilien](manuscript/statistical_tests.json)",
        "- [Alle 94 historischen Aussagen: aktuelle Werte, Ersatz oder Streichung](manuscript/claims.csv)",
        "- [Binärer E2-Hauptvergleich](manuscript/e2_binary_main_comparison.csv)",
        "- [Kontinuierliche und binäre E2-Designmetriken](manuscript/e2_design_metrics.csv)",
        "- [Lernkurven mit gültigen Fallzahlen](manuscript/e2_learning_by_seed.json)",
        "- [Kosten der zusätzlichen Manuskriptauswertung](manuscript/postprocessing_evidence.json)",
        "- [Einzelläufe, Kosten und Belegpfade](run_index.json)",
        "- [Statistik aus tatsächlichen Seeds](statistics.json)",
        "- [Explorative gepaarte Vergleiche](comparisons.json)",
        "- [Binäre Netzverfeinerung](mesh_refinement.json)",
        "", "Pro Lauf enthält run.json Konfiguration, Hardware, Trainingskurven, FEM-Ereignisse, Vorhersagen, Abbruchgrund und Trajektorie. best_design.npy enthält das Design; reine Budgetausschnitte enthalten bewusst kein nachträglich ausgewähltes Design. independent_test.json trennt kontinuierliche und binäre Testdaten.",
        "", "Zeitkonten sind exklusiv innerhalb solve(); Konstruktor-, Import- und Berichtskosten sind darin nicht enthalten. Konstruktor-Messungen stehen separat unter construction, zusätzliche FEM-Auswertungen für Abbildungen in figure_generation_evidence.json. FEM-Rolle other ist nicht automatisch reine Suchzeit. Checkpoint-Fortsetzungen bewahren die protokollierte Rechenzeit. Laufzeit eines FEM-Budgetausschnitts bleibt die des vollständigen Ursprungsruns.",
        "", "Vorhersagefehler ausgewählter Kandidaten sind selektionsbedingt; sie ersetzen keinen unabhängigen Test. Validierung zur Trainingssteuerung ist ebenfalls kein Test. Fehlende Rangmetriken oder Konfidenzintervalle bleiben null.",
        "", "E1-Suchzufall: SeedSequence trennt Laufseed, Zyklus und Zweck (Suchstarts, Explorationspool, unabhängiger Test). Die Suchseeds stehen je Zyklus in search_random_streams; Workerzahlen gehen nicht in die Ableitung ein. Q4-Knoten sind gegen den Uhrzeigersinn im physikalischen x-rechts/y-oben-System geordnet; Finnengrafiken verwenden physikalische Längen und die tatsächliche abwärts gerichtete Last.",
        "", "no_periodic_verification bezeichnet den eingefrorenen Surrogat-Kontrolllauf; Kandidaten werden weiterhin FEM-bewertet. Progressive BO behält historische Labels nach approximativem Koordinatentransfer und ist explorativ.",
        "", "Netzverfeinerung überträgt binäre Zellen ohne erneute Optimierung. Die Punktlast begrenzt Aussagen über Kontinuumskonvergenz.",
        "", "Die Diagramme diagnostics.png in jedem Laufverzeichnis zeigen Konvergenz, exklusive Zeitanteile und Vorhersagevergleich. trajectory.json enthält zusätzlich protokollierte Zeitpunkte; null bedeutet nicht instrumentiert. Rekonstruierte Ankertrajektorien tragen den Zeitpunkt ihrer Protokollierung, nicht rückwirkend geschätzte Zeiten.",
        "", "[Publikationstabelle CSV](publication_results.csv) · [Belegzuordnung](claim_provenance.json)",
        "", "## Laufbelege", ""]
    lines += [f"- [{r['family']}/{r['method']}, Seed {r['seed']}]({r['evidence']}): J={r['compliance']:.9g}" for r in rows]
    report=root / "REPORT.md"; _atomic_text(report,"\n".join(lines)+"\n"); paths.append(report)
    import html
    overview=root / "REPORT.html"
    links="".join(f'<li><a href="{r["evidence"]}">{html.escape(r["family"]+"/"+r["method"])} Seed {r["seed"]}</a> <a href="{r["evidence"].replace("run.json","diagnostics.png")}">Diagramm</a></li>' for r in rows)
    _atomic_text(overview,'<meta charset="utf-8"><title>PISCO R2</title><h1>PISCO R2</h1><pre>'+html.escape("\n".join(lines[:25]))+"</pre><ul>"+links+"</ul>")
    paths.append(overview)
    # Hash every documented artifact; this index is intentionally not self-hashed.
    emit("artifact_manifest.json",{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    print(f"R2 documentation: {report}; complete={completeness['full_publication_complete']}")
    return completeness


def verify_r2(builder):
    root=builder.output_root / "documentation"
    try:
        status=json.loads((root / "completeness.json").read_text())
        manifest=json.loads((root / "artifact_manifest.json").read_text())
        failures=[name for name,digest in manifest.items() if not (root/name).is_file()
                  or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
        if failures or not status["full_publication_complete"]:
            print("R2 verification FAILED: incomplete experiments, provenance errors, or changed artifacts.")
            print(json.dumps({"missing":status["missing"],"errors":status["errors"],"hash_failures":failures},indent=2))
            return 1
        for source in (Path(solvers.__file__),Path(__file__)):
            if source.read_bytes() != (root / "source" / source.name).read_bytes():
                print("R2 verification FAILED: source changed since report generation")
                return 1
        print("R2 verification passed: complete experiment matrix and artifact hashes")
        return 0
    except (OSError,KeyError,ValueError) as error:
        print(f"R2 verification FAILED: {error}")
        return 1


def verify_smoke(builder):
    """Check the small run's evidence without relaxing publication verification."""
    root=builder.output_root / "documentation"
    manifest=json.loads((root / "artifact_manifest.json").read_text())
    failures=[name for name,digest in manifest.items() if not (root/name).is_file()
        or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    rows=json.loads((root / "run_index.json").read_text())
    expected={"example1/pisco":{7,8},"example2/pisco_binary":{7,8},
        "example2/field_no_physics_matched":{7,8},"example2/scalar_surrogate":{7,8},
        "example2/simp":{0},"example3/pisco_d35":{0,1},
        "example3/scalar_surrogate_d35":{0,1},"mesh_refinement/pisco_binary":{7,8},"mesh_refinement/simp":{0}}
    observed={}
    for row in rows:
        key=row["family"]+"/"+row["method"]
        observed.setdefault(key,set()).add(row["seed"])
        run=json.loads((root / row["evidence"]).read_text())
        audit=run["extra"]["r2_evidence"]
        if not math.isfinite(row["compliance"]) or not audit["fem_events"] or not audit["exclusive_seconds"]:
            failures.append(row["evidence"]+": missing numerical evidence")
        if row["family"] != "mesh_refinement" and row["method"] != "simp":
            if not audit["training"] or not audit["predictions"]:
                failures.append(row["evidence"]+": missing training/predictions")
            independent=json.loads((root / row["evidence"]).with_name("independent_test.json").read_text())
            if independent["n"]<4:
                failures.append(row["evidence"]+": missing independent test")
        if row["family"] == "mesh_refinement":
            mesh=run["extra"]["mesh_refinement"]
            if [v["factor"] for v in mesh]!=[1,2] or not all(v["binary"] for v in mesh):
                failures.append(row["evidence"]+": invalid mesh study")
            if row["method"] == "pisco_binary":
                source=json.loads((root/f"example2/pisco_binary/seed_{row['seed']}/run.json").read_text())
                density=np.asarray(source["best_design"])
                if (not np.all((density==0.) | (density==1.))
                    or not np.array_equal(density,np.asarray(run["best_design"]))
                    or not np.isclose(source["best_compliance"],mesh[0]["compliance"],rtol=1e-9,atol=1e-9)):
                    failures.append(row["evidence"]+": main and mesh study do not evaluate the same binary design")
    if observed != expected:
        failures.append("Smoke experiment matrix differs from expected matrix")
    status=json.loads((root / "completeness.json").read_text())
    if status["full_publication_complete"]:
        failures.append("Smoke run incorrectly declared publication complete")
    report={"mode":"SMOKE_TEST_NOT_PUBLICATION","passed":not failures,"runs":len(rows),
        "hashed_artifacts":len(manifest),"failures":failures,
        "publication_complete":False,"scope":"Real reduced-budget E1/E2/Fin solves, scalar/field controls, mesh refinement and documentation; not all publication algorithms"}
    _atomic_text(builder.output_root / "smoke_verification.json",json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return int(bool(failures))


def reproduction_environment():
    """Record the execution boundary; a probe match is not a portability proof."""
    import importlib.metadata
    import torch
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=1):
        # Fixed inputs, no consumption of experiment RNG streams.
        x = torch.arange(63, dtype=torch.float64).reshape(9, 7) / 63
        w = torch.linspace(-.7, .9, 35, dtype=torch.float64).reshape(7, 5)
        w.requires_grad_(True)
        y = torch.tanh(x @ w)
        gradient, = torch.autograd.grad(y.square().sum(), w)
        a = np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 3.]])
        solution = spsolve(csc_matrix(a), np.array([1., 2., 3.]))
        probe = {"forward": solvers._tensor_digest([y]),
                 "gradient": solvers._tensor_digest([gradient]),
                 "sparse_solve": hashlib.sha256(solution.tobytes()).hexdigest()}
        libraries = []
        for item in threadpool_info():
            row = {k: item.get(k) for k in
                   ("internal_api", "prefix", "version", "architecture", "threading_layer", "num_threads")}
            path = Path(item["filepath"])
            row["binary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            libraries.append(row)
    cpu = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().split("\n\n", 1)[0].splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("vendor_id", "model name", "cpu family", "model", "stepping", "flags", "microcode"):
                cpu[key.strip()] = value.strip()
    return {"schema": "pisco-environment-v1", "execution": solvers.execution_metadata(),
            "platform": platform.platform(), "cpu": cpu,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "torch", "matplotlib", "scikit-learn", "Pillow", "threadpoolctl")},
            "native_libraries": sorted(libraries, key=lambda r: (r["prefix"] or "", r["binary_sha256"])),
            "probe": probe,
            "scope": "Environment guard and small numerical probe; no cross-platform equality guarantee."}


def guard_reproduction_environment(output, current):
    """Reject continuation under changed conditions before modifying evidence."""
    path = Path(output) / "reproduction_environment.json"
    if path.is_file():
        if json.loads(path.read_text()) != current:
            raise RuntimeError("Reproduction environment or numerical probe differs; use a new --output directory")
    else:
        _atomic_text(path, json.dumps(current, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent fixed work packets (default: 4); native kernels use one thread per packet")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="Heartbeat interval in seconds (default: 30)")
    parser.add_argument("--smoke", action="store_true", help="Quick real E1/E2/E3 runs and documentation; not publication results")
    parser.add_argument(
        "--example", choices=("example1", "example2", "example3"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--without-controls", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-frozen", action="store_true")
    parser.add_argument("--training-policy", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--fresh", action="store_true", help="Require a new empty output directory")
    parser.add_argument("--legacy-figures", action="store_true", help="Also render manuscript-layout figures")
    parser.add_argument("--submission-assets", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    solvers.register_source_guard(Path(__file__))
    if args.workers < 1:
        parser.error("--workers must be positive")
    solvers.configure_execution(args.workers)
    if not math.isfinite(args.progress_interval) or args.progress_interval <= 0:
        parser.error("--progress-interval must be a finite positive number")

    if args.list or not any((
            args.all, args.smoke, args.example, args.build, args.verify, args.verify_frozen,
            args.submission_assets)):
        print("Available benchmark families:")
        print("  example1  Parametric ellipse: PISCO, DE, BO and CMA-ES")
        print("  example2  L-bracket: revised PISCO variants, controls and SIMP")
        print("  example3  Fin: dimensions 35/62/135/243 and progressive search")
        print("\nAll completed seeds are stored atomically and reused on restart.")
        return 0
    if args.smoke and any((args.all,args.example,args.seeds,args.without_controls)):
        parser.error("--smoke uses its own fixed small experiment matrix")
    if args.smoke and args.output == DEFAULT_OUTPUT:
        args.output = ROOT / "smoke_output_r2"
    if args.all and args.example:
        parser.error("choose --all or --example, not both")
    if args.seeds and not (args.all or args.example):
        parser.error("--seeds requires --all or --example")

    if args.fresh and args.output.exists() and any(args.output.iterdir()):
        parser.error("--fresh requires an empty/new --output directory; existing evidence is preserved")
    builder = PublicationBuilder(args.output)
    if (args.all or args.example or args.smoke) and builder.bundle_path.is_file():
        try:
            existing = builder.load_results()
            for methods in existing.values():
                for runs in methods.values():
                    for result in runs:
                        configuration = result.extra.get("r2_evidence", {}).get("configuration", {})
                        if (configuration.get("source_sha256") != hashlib.sha256(Path(solvers.__file__).read_bytes()).hexdigest()
                                or configuration.get("numerical_execution") != solvers.execution_metadata()):
                            raise RuntimeError("Source or numerical execution settings differ")
        except RuntimeError as error:
            parser.error(f"{error}; use a new --output directory to preserve existing evidence")
    if args.all or args.example or args.smoke:
        try:
            guard_reproduction_environment(args.output, reproduction_environment())
        except RuntimeError as error:
            parser.error(str(error))
    if args.verify_frozen:
        failures = _verify_frozen_claim_rows() + _verify_frozen_comparisons()
        print("Frozen R1 reference:", failures or "verified (not R2 reproduction)")
        if failures: return 1
    if args.submission_assets:
        builder.write_editable_table_bundle()
        builder.write_submission_assets_manifest()
    numerical_results = None
    if args.smoke:
        policy=solvers.TrainingPolicy(independent_test_size=4)
        if args.training_policy == "adaptive":
            policy=solvers.TrainingPolicy(independent_test_size=4,retain_optimizer=True,
                validation_fraction=.25,normalize_compliance=True,update_steps=2,min_steps=1,patience=2)
        _atomic_text(args.output / "experiment_plan.json",json.dumps({"mode":"SMOKE_TEST_NOT_PUBLICATION",
            "command":sys.argv,"numerical_execution":solvers.execution_metadata(),"training_policy":asdict(policy),"description":"Two seeds for E1/E2/Fin surrogates, short SIMP, binary refinement; reduced budgets"},indent=2))
        smoke_suite=PublicationExperimentSuite(args.output,training_policy=policy)
        smoke_suite.store.progress_interval=args.progress_interval
        numerical_results=smoke_suite.smoke()
        builder.save_results(numerical_results)
        numerical_results=builder.load_results()
    if args.all or args.example:
        policy = (solvers.TrainingPolicy(retain_optimizer=True, validation_fraction=0.2,
                    normalize_compliance=True, update_steps=100)
                  if args.training_policy == "adaptive" else solvers.TrainingPolicy())
        selected_examples=["example1","example2","example3"] if args.all else [args.example]
        _atomic_text(args.output / "experiment_plan.json",json.dumps({
            "protocol":solvers.PUBLICATION_PROTOCOL,"numerical_execution":solvers.execution_metadata(),"training_policy":asdict(policy),
            "source_sha256":hashlib.sha256(Path(solvers.__file__).read_bytes()).hexdigest(),
            "command":sys.argv,"selected":solvers.expected_publication_experiments(
                seeds=_parse_seeds(args.seeds),include_controls=not args.without_controls,examples=selected_examples),
            "full_publication":solvers.expected_publication_experiments()},indent=2))
        suite = PublicationExperimentSuite(args.output, training_policy=policy)
        suite.store.progress_interval=args.progress_interval
        selected = "all" if args.all else args.example
        progress = ProgressDisplay(3 if selected == "all" else 1)
        if selected == "all":
            numerical_results = {}
            for example in ("example1", "example2", "example3"):
                subset = progress.run(
                    f"Numerical benchmark: {example}",
                    lambda example=example: suite.run(
                        example, seeds=_parse_seeds(args.seeds),
                        include_controls=not args.without_controls))
                numerical_results.update(subset)
                builder.save_results(subset)
        else:
            numerical_results = progress.run(
                f"Numerical benchmark: {selected}",
                lambda: suite.run(
                    selected, seeds=_parse_seeds(args.seeds),
                    include_controls=not args.without_controls))
        builder.save_results(numerical_results)
        numerical_results = builder.load_results()

    if args.build or numerical_results is not None:
        if numerical_results is None:
            numerical_results = builder.load_results()
        available={f"{family}/{method}" for family,methods in numerical_results.items() for method in methods}
        if args.legacy_figures or set(solvers.expected_publication_experiments()).issubset(available):
            figure_audit=solvers.RunAudit()
            token=solvers._ACTIVE_AUDIT.set(figure_audit)
            failure_path=args.output / "figure_generation_error.json"
            try:
                with solvers.audit_scope("publication_figures"):
                    builder.render_figures(numerical_results)
                failure_path.unlink(missing_ok=True)
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
                _atomic_text(failure_path,json.dumps({"error":str(error)},indent=2))
                print(f"Figure generation failed; numerical evidence will still be documented: {error}")
            finally:
                solvers._ACTIVE_AUDIT.reset(token)
            _atomic_text(args.output / "figure_generation_evidence.json",json.dumps(_json_value(figure_audit.snapshot()),indent=2))
        progress = ProgressDisplay(1)
        progress.run(
            "Statistics, tables, plots and manifest",
            lambda: write_r2_documentation(builder, numerical_results if numerical_results is not None else builder.load_results()))

    if args.smoke:
        return verify_smoke(builder)
    if args.verify or args.all:
        return verify_r2(builder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
