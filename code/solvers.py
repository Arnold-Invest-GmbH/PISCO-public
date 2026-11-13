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

"""
solvers.py — numerical API for the revised PISCO paper (self-contained)
=================================================================

This is the single numerical source file of the public reproduction.  It
contains the finite-element core, all benchmark definitions, the original
and revised PISCO variants, the comparison methods and durable result
storage.  Plotting, statistics and publication-output formatting deliberately
live in :mod:`reproduce_all`, where ``PublicationBuilder`` keeps that work
separate from the numerical algorithms.

Aufbau (in dieser Reihenfolge im File)
--------------------------------------
1. **Problem-Dataclasses** — ``EllipseProblem``, ``EllipseBox``,
   ``LBracketProblem``.
2. **FEM-Kern** — Steifigkeitsmatrix, Mesh, BC, Solve, Spannungen
   (``_lk``, ``_prepare_fem``, ``_fea``, ``_assemble_K``, …).
3. **Ellipsen-Geometrie** — Levelset, Feasibility, Sampling,
   Lastvektor, Volldesign-Referenz (Beispiel 1).
4. **L-Bracket-Geometrie** — Maske + Lastvektor (Beispiel 2).
5. **SIMP-Internals** — Sensitivitätsfilter und OC-Iteration.
6. **PISCO-Internals** — physikinformiertes MLP-Surrogat,
   Trainings-Loop, Latent-Multi-Start, Top-k-Selektion.
7. **Solver-Contract** — ``SolverResult`` und ``BaseSolver`` mit
   automatischem Datei-Cache.
8. **Solver** — ``PISCOSolver`` (mit Generator-Hierarchie),
   ``SIMPSolver``, ``DESolver``.

Bezug zum Paper
---------------
* Tabelle 2 (Beispiel 1): ``DESolver`` als Black-Box-Baseline,
  ``PISCOSolver`` als Hauptverfahren mit
  ``PISCOConfig(T_cycles=12, lam_phys=1e-3)``.
* Tabelle 3 (Beispiel 2): ``SIMPSolver(n_starts=1)`` und
  ``SIMPSolver(n_starts=4)`` als Vergleichsgrößen.

Designentscheidungen
--------------------
* Solver besitzen ihr Problem und ihre Konfiguration (Dataclass) als
  unveränderliche Attribute; ``solve(*, seed)`` ist die einzige
  Pflichtschnittstelle.
* Reproduzierbarkeit: Der Seed wird *erst beim Aufruf* übergeben,
  nicht beim Konstruktor — derselbe Solver-Instance lässt sich mit
  verschiedenen Seeds vermessen.
* Result-Caching: Identische Aufrufe (Solver-Klasse + Config + Problem
  + Seed) treffen den Cache und kehren in Millisekunden zurück.

Beispielnutzung
---------------
>>> from solvers import (
...     EllipseProblem, EllipseBox, LBracketProblem,
...     PISCOSolver, PISCOConfig, AnalyticEllipseGenerator,
...     SIMPSolver, SIMPConfig, DESolver, DEConfig)
>>>
>>> problem = EllipseProblem()
>>> generator = AnalyticEllipseGenerator(problem, EllipseBox())
>>> r_pisco = PISCOSolver(problem, PISCOConfig(), generator=generator).solve(seed=7)
>>> r_de    = DESolver(problem, DEConfig()).solve(seed=7)
>>>
>>> lbracket = LBracketProblem()
>>> r_simp_s = SIMPSolver(lbracket, SIMPConfig(n_starts=1)).solve(seed=0)
>>> r_simp_m = SIMPSolver(lbracket, SIMPConfig(n_starts=4)).solve(seed=0)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# A long-running process must never stamp already imported code with the hash
# of a later edit on disk. Register the driver too when invoked through the CLI.
_SOURCE_GUARDS = {Path(__file__).resolve(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def register_source_guard(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path in _SOURCE_GUARDS and _SOURCE_GUARDS[path] != digest:
        raise RuntimeError("Source changed since process start; use a new run with consistent sources")
    _SOURCE_GUARDS[path] = digest


def _checked_source_digest():
    for path, expected in _SOURCE_GUARDS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during execution: {path}; stop and restart with consistent sources")
    return _SOURCE_GUARDS[Path(__file__).resolve()]

# Native kernels are fixed; --workers controls independent work packets.
import os
for _native_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_native_variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import numpy as np
import scipy
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

# PyTorch warnt seit ~2.x bei jedem ``sparse_coo_tensor``-Aufruf, dass die
# Invarianten-Prüfung "implizit deaktiviert" sei und man sich explizit
# entscheiden möge. Da unsere Sparse-Inputs ausnahmslos aus
# ``scipy.sparse``-Matrizen kommen (per Konstruktion gültige Indizes),
# wählen wir globales Disable — eine Modulebene-Entscheidung, kein
# Kontextwechsel pro Aufruf.
torch.sparse.check_sparse_tensor_invariants.disable()

# Standard-Cache-Pfad: ``solver_cache/`` neben dieser Datei. So findet
# das Reproduktions-Notebook denselben Cache unabhängig vom aktuellen
# Arbeitsverzeichnis. Wer einen anderen Ort braucht, gibt
# ``cache_dir=...`` beim Solver-Konstruktor an.
DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent / "solver_cache"


# =====================================================================
# 1. PROBLEM-DEFINITIONEN — Mesh, Material, Lasten
# =====================================================================
#
# Beide Probleme sind reine Daten-Container (Dataclasses), damit das
# Result-Caching ihre Felder hashen kann. Methoden gehören in die
# Solver-Klassen, die die Probleme konsumieren.
# =====================================================================

@dataclass
class EllipseProblem:
    """Beispiel 1 — rechteckige Kragarmplatte mit elliptischer Aussparung.

    Default-Werte entsprechen Abschn.\\ 5.1 / Tab.\\ 1 im Paper:
    120 × 60 mm Domäne, 36 × 18 Elemente, Stahl-Modul ($E_0 = 210$ GPa),
    Linienlast $q = 100$ N/mm an der rechten Kante zwischen
    $y \\in [25, 35]$ mm.
    """
    nelx: int = 36
    nely: int = 18
    Lx_mm: float = 120.0
    Ly_mm: float = 60.0
    E0: float = 210000.0
    nu: float = 0.30
    penal: float = 3.0
    q_line: float = 100.0
    thickness_mm: float = 5.0
    load_y1: float = 25.0
    load_y2: float = 35.0
    target_hole_radius_mm: float = 10.0
    beta: float = 8.0
    rho_void: float = 1e-3
    fixed_area: bool = True

    @property
    def Emin(self) -> float:
        return self.E0 * 1e-6

    @property
    def target_hole_area(self) -> float:
        return math.pi * self.target_hole_radius_mm ** 2

    def ry_from_rx(self, rx: float) -> float:
        return self.target_hole_area / (math.pi * rx)


@dataclass
class EllipseBox:
    """Box-Constraints für den Ellipse-Designparametervektor
    $(x_c, y_c, r_x, r_y, \\theta)$. Werte aus Abschn.\\ 5.1 im Paper."""
    xc: tuple = (25.0, 90.0)
    yc: tuple = (15.0, 45.0)
    rx: tuple = (7.15, 14.0)
    ry: tuple = (6.0, 14.0)
    th: tuple = (0.0, math.pi)


@dataclass
class LBracketProblem:
    """Beispiel 2 — L-förmige 2D-Domäne mit Volumenrestriktion.

    60 × 36 Elemente, $V^* = 0.45$, Penalty $p=3$, Filterradius
    $r_\\min = 2.4$ — Standardparameter aus Abschn.\\ 5.3.1."""
    nelx: int = 60
    nely: int = 36
    volfrac: float = 0.45
    penal: float = 3.0
    rmin: float = 2.4


# =====================================================================
# 2. FEM-KERN — Steifigkeitsmatrix, Mesh, Solve
# =====================================================================
#
# Implementation orientiert sich an top88 (Andreassen et al., 2011);
# die Funktionen sind state-frei und werden von SIMP, DE und PISCO
# gemeinsam verwendet.
# =====================================================================

def _lk(E: float = 1.0, nu: float = 0.3, *, hx=1.0, hy=1.0,
        thickness=1.0) -> np.ndarray:
    """Element-Steifigkeitsmatrix eines bilinearen Q4-Elements unter
    ebenem Spannungszustand, einschließlich Zellmaßen und Dicke.

    Die top88-Form gilt nur für quadratische Zellen. Rechteckige Zellen
    werden mit exakter 2x2-Gaußquadratur integriert (gleiche lokale DOF-Folge).
    """
    if not all(math.isfinite(v) and v > 0 for v in (hx, hy, thickness, E)):
        raise ValueError("Element lengths, thickness and modulus must be finite and positive")
    if not -1.0 < nu < 0.5:
        raise ValueError("Poisson ratio must lie between -1 and 0.5")
    if hx != hy:
        D = E / (1.0 - nu**2) * np.array(
            [[1., nu, 0.], [nu, 1., 0.], [0., 0., (1. - nu) / 2.]])
        matrix = np.zeros((8, 8))
        for xi in (-1. / math.sqrt(3.), 1. / math.sqrt(3.)):
            for eta in (-1. / math.sqrt(3.), 1. / math.sqrt(3.)):
                dx = np.array([eta-1., 1.-eta, 1.+eta, -1.-eta]) / (2.*hx)
                dy = np.array([xi-1., -1.-xi, 1.+xi, 1.-xi]) / (2.*hy)
                B = np.zeros((3, 8))
                B[0, 0::2], B[1, 1::2] = dx, dy
                B[2, 0::2], B[2, 1::2] = dy, dx
                matrix += (B.T @ D @ B) * (hx * hy * thickness / 4.)
        return matrix
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0],
                    [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6],
                    [0, -3, -6, 3], [3, -6, 3, -6]], dtype=float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4],
                    [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2],
                    [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=float)
    return E * thickness / (1.0 - nu ** 2) / 24.0 * (
        np.block([[A11, A12], [A12.T, A11]])
        + nu * np.block([[B11, B12], [B12.T, B11]])
    )


def _prepare_fem(nelx: int, nely: int):
    """Mesh-Konnektivität: Knotennummerierung (column-major), Element-
    DOF-Tabelle und (iK, jK) für die spätere Sparse-Assemblierung.

    Physical y increases with the array row, as in _structured_grid and
    _node_coords. Local nodes are counter-clockwise: lower left, lower right,
    upper right, upper left. This is the convention of _lk and B.
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape((nely + 1, nelx + 1), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 2).reshape(nelx * nely, order="F")
    edofMat = np.tile(edofVec[:, None], (1, 8)) + np.array(
        [-2, -1, 2 * nely, 2 * nely + 1, 2 * nely + 2, 2 * nely + 3, 0, 1])
    iK = np.kron(edofMat, np.ones((8, 1))).ravel().astype(int)
    jK = np.kron(edofMat, np.ones((1, 8))).ravel().astype(int)
    return ndof, nodenrs, edofMat.astype(int), iK, jK


def _structured_grid(nelx: int, nely: int, Lx: float = 1.0, Ly: float = 1.0):
    """Element-Zentren-Koordinaten auf einem strukturierten 2D-Mesh."""
    xs = np.linspace(0.0, Lx, nelx + 1)
    ys = np.linspace(0.0, Ly, nely + 1)
    Xc, Yc = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]))
    return xs, ys, Xc, Yc


def _fea_impl(nelx, nely, xPhys, fixed_dofs, F, penal=3.0, E0=1.0, Emin=1e-9, nu=0.3,
              *, hx=1.0, hy=1.0, thickness=1.0):
    """Vollständige FEM-Lösung für Dichteverteilung ``xPhys``.

    Liefert Verschiebungsfeld $U$, skalare Compliance $c = U^T F$, die
    Dehnungsenergie pro Element ($c_e = u_e^T K_e u_e$), und die
    elementweise Steifigkeit ``Eelem``."""
    KE = _lk(1.0, nu, hx=hx, hy=hy, thickness=thickness)
    ndof, _, edofMat, iK, jK = _prepare_fem(nelx, nely)
    xPhys = np.asarray(xPhys).reshape(nely, nelx, order="F")
    Eelem = Emin + np.power(xPhys.ravel(order="F"), penal) * (E0 - Emin)
    sK = ((KE.ravel()[None]).T * Eelem).T.ravel()
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = (K + K.T) * 0.5
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    U = np.zeros(ndof)
    U[free] = spla.spsolve(K[free, :][:, free], F[free])
    ce = np.sum((U[edofMat] @ KE) * U[edofMat], axis=1)
    compliance = float(np.sum(Eelem * ce))
    return U, compliance, ce.reshape((nely, nelx), order="F"), Eelem.reshape((nely, nelx), order="F")


def _fea(nelx, nely, xPhys, fixed_dofs, F, penal=3.0, E0=1.0, Emin=1e-9, nu=0.3,
         *, hx=1.0, hy=1.0, thickness=1.0):
    audit = _ACTIVE_AUDIT.get()
    role = audit.stage if audit is not None else "unscoped"
    with audit_scope("fem_" + role):
        result = _fea_impl(nelx, nely, xPhys, fixed_dofs, F, penal, E0, Emin, nu,
                           hx=hx, hy=hy, thickness=thickness)
    if audit is not None:
        density = np.asarray(xPhys, dtype=np.float64)
        digest = hashlib.sha256(density.tobytes()).hexdigest()
        audit.events.append({"call": len(audit.events)+1, "elapsed_s": audit.elapsed(),
            "role": role, "compliance": float(result[1]), "design_sha256": digest,
            "duplicate_design": digest in audit.seen, "mesh": [nelx, nely],
            "element_geometry": {"hx": hx, "hy": hy, "thickness": thickness},
            "binary": bool(np.all((density == 0) | (density == 1)))})
        audit.seen.add(digest)
    return result


def _assemble_K(nelx, nely, rho, *, penal=3.0, E0=1.0, Emin=1e-9, nu=0.3,
                 hx=1.0, hy=1.0, thickness=1.0) -> sp.csr_matrix:
    """Assembliere globale Steifigkeitsmatrix $K(\\rho)$ — exakt analog
    zur Assemblierung in ``_fea`` (für PISCO-Physikverlust)."""
    KE = _lk(1.0, nu, hx=hx, hy=hy, thickness=thickness)
    ndof, _, edofMat, iK, jK = _prepare_fem(nelx, nely)
    rho_flat = np.asarray(rho).reshape(nely, nelx, order="F").ravel(order="F")
    Eelem = Emin + np.power(rho_flat, penal) * (E0 - Emin)
    sK = ((KE.ravel()[None]).T * Eelem).T.ravel()
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
    return 0.5 * (K + K.T)


def _csr_to_torch(K: sp.csr_matrix) -> torch.Tensor:
    """Konvertiere scipy-Sparse-Matrix in einen ``torch.sparse_coo_tensor``
    (Eingabe für die Physikverlust-Berechnung im PISCO-Surrogat).

    ``check_invariants=False``: PyTorch warnt sonst bei jedem Aufruf, dass
    die Invarianten-Prüfung explizit aktiviert/deaktiviert werden müsse.
    Wir deaktivieren explizit pro Aufruf — die Eingabe-Indizes kommen aus
    ``scipy.sparse.coo_matrix`` und sind per Konstruktion gültig."""
    coo = K.tocoo()
    idx = np.stack([coo.row, coo.col])
    return torch.sparse_coo_tensor(
        torch.tensor(idx, dtype=torch.long),
        torch.tensor(coo.data, dtype=torch.float32),
        size=K.shape, check_invariants=False).coalesce()


def _element_B_center(hx: float, hy: float) -> np.ndarray:
    """Strain-Displacement-Matrix $B$ am Element-Mittelpunkt — für die
    Spannungsauswertung (von Mises) im Postprocessing."""
    dN_dx = np.array([-0.25, 0.25, 0.25, -0.25]) * 2.0 / hx
    dN_dy = np.array([-0.25, -0.25, 0.25, 0.25]) * 2.0 / hy
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dN_dx[i]
        B[1, 2 * i + 1] = dN_dy[i]
        B[2, 2 * i] = dN_dy[i]
        B[2, 2 * i + 1] = dN_dx[i]
    return B


def _von_mises(nelx, nely, U, Eelem, nu=0.3, Lx=1.0, Ly=1.0) -> np.ndarray:
    """Elementweise von-Mises-Spannung am Mittelpunkt, multipliziert mit
    der elementweisen Steifigkeit. Reine Auswertung — wird im Loop nicht
    differenziert, dient nur dem Postprocessing der besten Designs."""
    B = _element_B_center(Lx / nelx, Ly / nely)
    D0 = 1.0 / (1.0 - nu ** 2) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    _, _, edofMat, _, _ = _prepare_fem(nelx, nely)
    Eflat = Eelem.ravel(order="F")
    vm = np.zeros(edofMat.shape[0])
    for e, edofs in enumerate(edofMat):
        sx, sy, txy = Eflat[e] * (D0 @ (B @ U[edofs]))
        vm[e] = np.sqrt(max(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2, 0.0))
    return vm.reshape((nely, nelx), order="F")


def _build_filter(nelx: int, nely: int, rmin: float):
    """Linearer Sensitivitätsfilter (cone filter) für SIMP. Verhindert
    Schachbrettmuster und macht den Optimierer mesh-unabhängiger."""
    iH, jH, sH = [], [], []
    radius = int(np.floor(rmin))
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(i - radius, 0), min(i + radius + 1, nelx)):
                for ll in range(max(j - radius, 0), min(j + radius + 1, nely)):
                    w = rmin - np.sqrt((i - k) ** 2 + (j - ll) ** 2)
                    if w > 0.0:
                        iH.append(row); jH.append(k * nely + ll); sH.append(w)
    H = sp.coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsr()
    return H, np.asarray(H.sum(axis=1)).ravel()


# =====================================================================
# 3. ELLIPSEN-GEOMETRIE & BEWERTUNG (Beispiel 1)
# =====================================================================
#
# Ellipse als Levelset auf Element-Zentren plus tanh-Glättung →
# Dichtefeld; dann Kragarm-Linienlast und FEM-Solve. Diese Funktionen
# werden von DE (jeder Schritt = Solve) und PISCO (Initialdatensatz +
# Revalidierung) gemeinsam verwendet.
# =====================================================================

def _ellipse_phi(X, Y, params):
    """Levelset-Funktion einer rotierten Ellipse:
    $\\phi(x) = (x'/r_x)^2 + (y'/r_y)^2 - 1$ in lokalen Koordinaten."""
    xc, yc, rx, ry, theta = params
    ct, st = np.cos(theta), np.sin(theta)
    xt = ct * (X - xc) + st * (Y - yc)
    yt = -st * (X - xc) + ct * (Y - yc)
    return (xt / rx) ** 2 + (yt / ry) ** 2 - 1.0


def _ellipse_extents(rx, ry, theta):
    """Achsenparallele Bounding-Box der rotierten Ellipse (für
    Feasibility-Checks)."""
    ct, st = np.cos(theta), np.sin(theta)
    return float(np.sqrt((rx * ct) ** 2 + (ry * st) ** 2)), \
           float(np.sqrt((rx * st) ** 2 + (ry * ct) ** 2))


def _ellipse_feasible(params, *, Lx, Ly, load_y1, load_y2,
                       margin=3.0, load_clearance=5.0, support_clearance=6.0):
    """Geometrische Zulässigkeit: Ellipse muss vollständig im Designgebiet
    liegen, vom Lastbereich freigehalten werden und einen Mindestabstand
    zum eingespannten Rand wahren."""
    xc, yc, rx, ry, theta = params
    if rx <= 0.0 or ry <= 0.0:
        return False
    ext_x, ext_y = _ellipse_extents(rx, ry, theta)
    if not ((xc - ext_x >= margin) and (xc + ext_x <= Lx - margin)
            and (yc - ext_y >= margin) and (yc + ext_y <= Ly - margin)):
        return False
    if (xc + ext_x > Lx - load_clearance) and (yc + ext_y > load_y1) and (yc - ext_y < load_y2):
        return False
    if xc - ext_x < support_clearance:
        return False
    return True


def _ellipse_density(nelx, nely, params, *, Lx, Ly, beta=8.0, rho_void=1e-3):
    """Dichtefeld via tanh-Glättung des Levelsets:
    $\\rho = \\rho_\\text{void} + (1 - \\rho_\\text{void}) \\cdot
    \\tfrac12 (1 + \\tanh(\\beta \\phi))$.
    $\\beta$ steuert die Schärfe der Aussparungskante; $\\rho_\\text{void}$
    verhindert eine singuläre Steifigkeit innerhalb der Ellipse."""
    _, _, Xc, Yc = _structured_grid(nelx, nely, Lx, Ly)
    phi = _ellipse_phi(Xc, Yc, params)
    return rho_void + (1.0 - rho_void) * 0.5 * (1.0 + np.tanh(beta * phi))


def _cantilever_load(nelx, nely, *, Lx, Ly, load_y1, load_y2, q_line, thickness):
    """Kragarmplatte: linke Kante eingespannt (Dirichlet), rechte Kante
    trägt eine vertikale Linienlast $q$ über das Intervall
    $[y_1, y_2]$. Knoten an den Intervallenden bekommen jeweils
    nur die Hälfte der Last (Knoten-Konsistente Diskretisierung)."""
    ndof, nodenrs, *_ = _prepare_fem(nelx, nely)
    F = np.zeros(ndof)
    ys = np.linspace(0.0, Ly, nely + 1)
    right = nodenrs[:, -1]
    if not 0.0 <= load_y1 < load_y2 <= Ly or thickness <= 0:
        raise ValueError("Invalid load interval or plate thickness")
    # q_line is N/mm, already integrated through the thickness. Integrate the
    # edge shape functions on the actual loaded interval, including partial edges.
    for j, (bottom, top) in enumerate(zip(ys[:-1], ys[1:])):
        lower, upper = max(bottom, load_y1), min(top, load_y2)
        if upper <= lower:
            continue
        high_weight = ((upper-bottom)**2 - (lower-bottom)**2) / (2.*(top-bottom))
        low_weight = upper-lower-high_weight
        F[2*right[j]+1] -= q_line*low_weight
        F[2*right[j+1]+1] -= q_line*high_weight
    fixed = np.union1d(2 * nodenrs[:, 0], 2 * nodenrs[:, 0] + 1)
    return F, fixed


def _solve_ellipse(problem: EllipseProblem, params) -> dict:
    """Vollständige FEM-Auswertung eines Ellipsen-Designvektors —
    Dichtefeld bauen, FEM lösen, von-Mises-Spannungen rechnen."""
    F, fixed = _cantilever_load(
        problem.nelx, problem.nely, Lx=problem.Lx_mm, Ly=problem.Ly_mm,
        load_y1=problem.load_y1, load_y2=problem.load_y2,
        q_line=problem.q_line, thickness=problem.thickness_mm)
    rho = _ellipse_density(problem.nelx, problem.nely, params,
                           Lx=problem.Lx_mm, Ly=problem.Ly_mm,
                           beta=problem.beta, rho_void=problem.rho_void)
    U, comp, ce, Eelem = _fea(problem.nelx, problem.nely, rho, fixed, F,
                               penal=problem.penal, E0=problem.E0,
                               Emin=problem.Emin, nu=problem.nu,
                               hx=problem.Lx_mm/problem.nelx, hy=problem.Ly_mm/problem.nely,
                               thickness=problem.thickness_mm)
    vm = _von_mises(problem.nelx, problem.nely, U, Eelem,
                    nu=problem.nu, Lx=problem.Lx_mm, Ly=problem.Ly_mm)
    return {"rho": rho, "U": U, "compliance": comp, "strain_energy": ce,
            "Eelem": Eelem, "von_mises": vm, "F": F, "fixed_dofs": fixed}


def _solid_case(problem: EllipseProblem) -> dict:
    """Vergleichslösung der Vollplatte ($\\rho \\equiv 1$). Dient als
    Normierungsreferenz $J_\\text{solid}$ für die normierte Compliance
    $\\tilde J = J/J_\\text{solid}$ in Tab.\\ 2."""
    F, fixed = _cantilever_load(
        problem.nelx, problem.nely, Lx=problem.Lx_mm, Ly=problem.Ly_mm,
        load_y1=problem.load_y1, load_y2=problem.load_y2,
        q_line=problem.q_line, thickness=problem.thickness_mm)
    U, comp, ce, Eelem = _fea(problem.nelx, problem.nely,
                               np.ones((problem.nely, problem.nelx)), fixed, F,
                               penal=problem.penal, E0=problem.E0,
                               Emin=problem.Emin, nu=problem.nu,
                               hx=problem.Lx_mm/problem.nelx, hy=problem.Ly_mm/problem.nely,
                               thickness=problem.thickness_mm)
    vm = _von_mises(problem.nelx, problem.nely, U, Eelem,
                    nu=problem.nu, Lx=problem.Lx_mm, Ly=problem.Ly_mm)
    return {"rho": np.ones((problem.nely, problem.nelx)), "U": U,
            "compliance": comp, "strain_energy": ce, "Eelem": Eelem,
            "von_mises": vm, "F": F, "fixed_dofs": fixed}


def _design_to_input(a) -> np.ndarray:
    """Surrogat-Eingabekodierung: 5-dim Parametervektor →
    5-dim ``(xc, yc, rx, sin θ, cos θ)``. ``ry`` entfällt, da bei
    fixierter Lochfläche ``ry = A*/(π rx)`` vollständig durch ``rx``
    bestimmt ist."""
    xc, yc, rx, ry, theta = a
    return np.array([xc, yc, rx, math.sin(theta), math.cos(theta)],
                    dtype=np.float32)


def _area_ok(a, problem: EllipseProblem, tol: Optional[float] = None) -> bool:
    """Prüfe den Flächenconstraint unabhängig vom Erzeugungspfad.

    Auch bei konstruktiv fixierter Fläche wird das vollständige, an die FEM
    übergebene Design noch einmal geprüft. So können fehlerhafte oder extern
    erzeugte 5D-Designvektoren den Constraint nicht stillschweigend umgehen.
    """
    rx, ry = a[2], a[3]
    rel = abs(math.pi * rx * ry - problem.target_hole_area) \
        / problem.target_hole_area
    # 1e-6 ist eng genug für einen konstruktiv exakten Constraint und zugleich
    # robust gegenüber float32-Rundung in Generator- und Optimierungspfaden.
    effective_tol = 1e-6 if problem.fixed_area else (0.05 if tol is None else tol)
    return bool(rel <= effective_tol)


def _design_feasible_full(a, problem: EllipseProblem) -> bool:
    """Vollständige Feasibility: Geometrie *und* Flächen-Constraint."""
    gm = 2.0 * (problem.Lx_mm / problem.nelx)
    return (_ellipse_feasible(a, Lx=problem.Lx_mm, Ly=problem.Ly_mm,
                               load_y1=problem.load_y1, load_y2=problem.load_y2,
                               margin=gm, load_clearance=gm)
            and _area_ok(a, problem))


def _sample_design(rng: np.random.Generator, box: EllipseBox,
                    problem: EllipseProblem, max_tries: int = 400):
    """Ziehe ein zulässiges Zufallsdesign. Bei ``fixed_area=True`` wird
    ``ry = A*/(π rx)`` deterministisch berechnet (4 freie Parameter);
    andernfalls wird ``ry`` aus einem schmalen Intervall um den Sollwert
    gesampelt (5 freie Parameter)."""
    for _ in range(max_tries):
        rx = rng.uniform(*box.rx)
        if problem.fixed_area:
            ry = problem.ry_from_rx(rx)
        else:
            ry_target = problem.target_hole_area / (math.pi * rx)
            ry_lo = max(box.ry[0], ry_target * (1 - 0.05) ** 0.5)
            ry_hi = min(box.ry[1], ry_target * (1 + 0.05) ** 0.5)
            if ry_hi <= ry_lo:
                continue
            ry = rng.uniform(ry_lo, ry_hi)
        a = (rng.uniform(*box.xc), rng.uniform(*box.yc), rx, ry,
             rng.uniform(*box.th))
        if _design_feasible_full(a, problem):
            return a
    return None


def _eval_fem(problem: EllipseProblem, a) -> dict:
    """Eine FEM-Auswertung eines Ellipsen-Designs — kompakte Repräsentation
    für den PISCO-Datensatz (Designvektor, Verschiebungsfeld, Compliance)."""
    sol = _solve_ellipse(problem, a)
    return {"a": np.asarray(a, dtype=np.float64),
            "U": sol["U"].astype(np.float32),
            "rho": sol["rho"].astype(np.float32),
            "compliance": float(sol["compliance"]),
            "von_mises": sol["von_mises"].astype(np.float32),
            "Eelem": sol["Eelem"].astype(np.float32)}


# =====================================================================
# 4. L-BRACKET-GEOMETRIE (Beispiel 2)
# =====================================================================

def _l_bracket_mask(nelx: int, nely: int, cutoff: float = 0.58) -> np.ndarray:
    """Boolesche Maske der L-Bracket-Domäne: alle Elemente außerhalb der
    rechten oberen Quadrant-Ausnahme (oberhalb von ``cutoff`` in beiden
    Achsen ausgeschlossen)."""
    _, _, Xc, Yc = _structured_grid(nelx, nely, 1.0, 1.0)
    return ~((Xc > cutoff) & (Yc > cutoff))


def _l_bracket_load(nelx: int, nely: int):
    """Lastvektor + Dirichlet-Indizes für die L-Bracket-Domäne:
    linke Kante eingespannt, vertikale Einzellast an der rechten unteren
    Ecke ($y = 0$)."""
    ndof, nodenrs, *_ = _prepare_fem(nelx, nely)
    F = np.zeros(ndof)
    fixed = np.union1d(2 * nodenrs[:, 0], 2 * nodenrs[:, 0] + 1)
    F[2 * nodenrs[0, -1] + 1] = -1.0
    return F, fixed


# =====================================================================
# 5. SIMP-INTERNALS — Optimalitätskriterium-Iteration
# =====================================================================

def _density_filter(raw, mask, H, Hs):
    """Physical densities: masked row-normalised cone filter of design variables."""
    variables = np.where(mask, raw, 0.).ravel(order="F")
    physical = ((H @ variables) / Hs).reshape(raw.shape, order="F")
    return np.where(mask, physical, 0.)


def _density_filter_pullback(gradient, mask, H, Hs):
    """Exact transpose of the masked forward filter, including boundary weights."""
    physical_gradient = np.where(mask, gradient, 0.).ravel(order="F")
    result = (H.T @ (physical_gradient / Hs)).reshape(gradient.shape, order="F")
    return np.where(mask, result, 0.)


def _initial_filtered_volume(raw, mask, H, Hs, volume):
    """Make the initial OC design feasible in physical, not raw, volume."""
    if not 0. < volume <= _density_filter(np.ones_like(raw), mask, H, Hs)[mask].mean():
        raise ValueError("Requested physical volume is infeasible for this masked filter")
    lower, upper = -float(np.max(raw)), 1. - float(np.min(raw))
    for _ in range(60):
        middle = (lower + upper) * .5
        candidate = np.where(mask, np.clip(raw + middle, 0., 1.), 0.)
        if _density_filter(candidate, mask, H, Hs)[mask].mean() < volume:
            lower = middle
        else:
            upper = middle
    return np.where(mask, np.clip(raw + (lower + upper)*.5, 0., 1.), 0.)


def _density_oc_update(raw, physical_dc, mask, H, Hs, volume, move=.2):
    """Density-filter SIMP OC step with exact compliance and volume gradients.

    The multiplier is chosen using the actual filtered material volume. No
    detached post-projection or sensitivity-filter heuristic is used in SIMP.
    """
    dc = _density_filter_pullback(physical_dc, mask, H, Hs)
    dv = _density_filter_pullback(np.ones_like(raw), mask, H, Hs)
    ratio = np.zeros_like(raw)
    ratio[mask] = np.maximum(-dc[mask] / dv[mask], 0.)
    lower, upper = 0., max(1., float(ratio[mask].max()))
    def candidate(multiplier):
        trial = raw * np.sqrt(ratio / max(multiplier, np.finfo(float).tiny))
        return np.where(mask, np.clip(trial, np.maximum(0., raw-move),
                                      np.minimum(1., raw+move)), 0.)
    while _density_filter(candidate(upper), mask, H, Hs)[mask].mean() > volume:
        upper *= 2.
        if not math.isfinite(upper):
            raise FloatingPointError("Could not bracket the OC volume multiplier")
    for _ in range(60):
        middle = (lower + upper) * .5
        updated = candidate(middle)
        if _density_filter(updated, mask, H, Hs)[mask].mean() > volume:
            lower = middle
        else:
            upper = middle
    return candidate(upper)


def _simp(nelx, nely, mask, F, fixed, *, volfrac=0.45, penal=3.0, rmin=2.5,
          E0=1.0, Emin=1e-9, nu=0.30, maxiter=80, tol=1e-3, verbose=False,
          x_init=None):
    """SIMP-Optimalitätskriterium-Iteration (top88-Stil).

    Ein Lauf: Filter aufbauen, in jeder Iteration FEM lösen,
    Sensitivitäten berechnen, OC-Update mit Move-Limit 0.2, bis
    ``max|Δρ| < tol`` oder ``maxiter`` erreicht. ``x_init`` erlaubt
    perturbierte Startfelder (für Multi-Start)."""
    mask = np.asarray(mask, dtype=bool)
    if x_init is None:
        x = np.zeros((nely, nelx)); x[mask] = volfrac
    else:
        x = np.array(x_init, dtype=float)
        x[~mask] = 0.0
    H, Hs = _build_filter(nelx, nely, rmin)
    x = _initial_filtered_volume(x, mask, H, Hs, volfrac)
    obj_hist = []
    for it in range(1, maxiter + 1):
        xPhys = _density_filter(x, mask, H, Hs)
        U, comp, ce, _ = _fea(nelx, nely, xPhys, fixed, F,
                               penal=penal, E0=E0, Emin=Emin, nu=nu)
        dc = -penal * (E0 - Emin) * np.power(np.maximum(xPhys, 1e-9), penal - 1.0) * ce
        xnew = _density_oc_update(x, dc, mask, H, Hs, volfrac)
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        obj_hist.append(float(comp))
        if verbose and (it == 1 or it % 10 == 0 or change < tol):
            print(f"  it={it:3d}  c={comp:10.4f}  chg={change:.4f}")
        if change < tol:
            break
    xPhys = _density_filter(x, mask, H, Hs)
    U, comp, ce, Eelem = _fea(nelx, nely, xPhys, fixed, F,
                               penal=penal, E0=E0, Emin=Emin, nu=nu)
    return {"x": x, "xPhys": xPhys, "U": U, "compliance": float(comp),
            "objective_history": np.array(obj_hist), "mask": mask,
            "strain_energy": ce, "Eelem": Eelem,
            "fem_calls": int(len(obj_hist) + 1)}


def _simp_multistart(nelx, nely, mask, F, fixed, *, n_starts=4, seed=0,
                     volfrac=0.45, penal=3.0, rmin=2.5,
                     E0=1.0, Emin=1e-9, nu=0.30, maxiter=80, tol=1e-3,
                     verbose=False):
    """Multi-Start-SIMP: ``n_starts`` unabhängige Läufe mit unterschiedlichen
    Startdichten; Rückgabe der besten Lösung plus Statistik aller Läufe.

    Lauf 0 verwendet die klassische gleichverteilte Startdichte
    $\\rho_0 \\equiv V^*$; die übrigen Läufe ziehen perturbierte Felder
    $\\rho_0 = \\text{clip}(V^* + \\sigma \\eta, 0, 1)$ mit
    $\\eta \\sim \\mathcal{N}(0, 1)$ und $\\sigma = 0.25$, anschließend
    leicht geglättet plus Volumenkorrektur."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, dtype=bool)
    runs = []
    fem_calls_total = 0
    best = None
    for k in range(n_starts):
        if k == 0:
            x_init = None  # klassischer Start
        else:
            eta = rng.standard_normal((nely, nelx))
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0
            from scipy.signal import convolve2d
            eta_smooth = convolve2d(eta, kernel, mode="same", boundary="symm")
            x_init = np.clip(volfrac + 0.25 * eta_smooth, 0.0, 1.0)
            scale = volfrac / max(x_init[mask].mean(), 1e-6)
            x_init = np.clip(x_init * scale, 0.0, 1.0)
            x_init[~mask] = 0.0
        res = _simp(nelx, nely, mask, F, fixed, volfrac=volfrac,
                     penal=penal, rmin=rmin, E0=E0, Emin=Emin, nu=nu,
                     maxiter=maxiter, tol=tol, verbose=False, x_init=x_init)
        fem_calls_total += res["fem_calls"]
        runs.append({"start_idx": k,
                     "compliance": float(res["compliance"]),
                     "fem_calls": int(res["fem_calls"])})
        if best is None or res["compliance"] < best["compliance"]:
            best = res
        if verbose:
            print(f"  multistart_simp[{k}]  c={res['compliance']:.4f}  "
                  f"fem={res['fem_calls']}")
    Js = np.array([r["compliance"] for r in runs])
    summary = {
        "n_starts": n_starts,
        "fem_calls_total": fem_calls_total,
        "compliance_best": float(Js.min()),
        "compliance_mean": float(Js.mean()),
        "compliance_std": float(Js.std(ddof=1) if n_starts > 1 else 0.0),
        "compliance_worst": float(Js.max()),
        "runs": runs,
    }
    return best, summary


# =====================================================================
# 6. PISCO-INTERNALS — Surrogat, Training, Latent-Suche
# =====================================================================

def _make_field_surrogate(d_in, d_out, hidden, depth, a_mean, a_std, u_scale,
                            fixed_dofs=None) -> nn.Module:
    """Konstruiere das physikinformierte MLP-Surrogat:
    ``d_in``-dim Designvektor → Verschiebungsfeld
    $\\hat U \\in \\mathbb{R}^{n_\\text{dof}}$ mit harter Dirichlet-Maske
    auf den fixierten DOFs (Output dort exakt 0)."""
    free_mask_np = np.ones(d_out, dtype=np.float32)
    if fixed_dofs is not None:
        free_mask_np[np.asarray(fixed_dofs, dtype=np.int64)] = 0.0

    class _FieldSurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("a_mean", a_mean)
            self.register_buffer("a_std", a_std)
            self.register_buffer("free_mask",
                                  torch.tensor(free_mask_np, dtype=torch.float32))
            self.u_scale = float(u_scale)
            layers = [nn.Linear(d_in, hidden), nn.Tanh()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            layers += [nn.Linear(hidden, d_out)]
            self.net = nn.Sequential(*layers)

        def forward(self, a):
            an = (a - self.a_mean) / self.a_std
            U_raw = self.u_scale * self.net(an)
            return U_raw * self.free_mask

    return _FieldSurrogate()


def _train_field_surrogate(model, dataset, F_torch, K_caches, *, steps=400, lr=2e-3,
                              lam_data=1.0, lam_phys=1e-2, free_dofs=None, clip=1.0) -> dict:
    """Surrogat-Update auf dem aktuellen Datensatz.

    Verlust = $\\lambda_d \\cdot \\Vert \\hat J - J \\Vert^2 +
    \\lambda_p \\cdot \\Vert K \\hat U - F \\Vert^2 / \\Vert F \\Vert^2$,
    Letzteres restringiert auf freie DOFs. Adam mit globalem Gradient-Clip
    auf Norm 1."""
    inputs = torch.tensor(np.stack([_design_to_input(item["a"]) for item in dataset]), dtype=torch.float32)
    targets = torch.tensor([item["compliance"] for item in dataset], dtype=torch.float32)
    free = torch.arange(F_torch.numel()) if free_dofs is None else torch.tensor(free_dofs, dtype=torch.long)
    return _fit_field(model, inputs, targets, F_torch, K_caches, free,
                      float(torch.linalg.norm(F_torch)), steps=steps, lr=lr,
                      lam_data=lam_data, lam_phys=lam_phys, clip=clip)


def _surrogate_compliance(model, a_batch, F_torch):
    """Surrogat-Compliance: $\\hat J(a) = \\hat U(a)^T F$."""
    return model(a_batch) @ F_torch


def _ellipse_penalty(a_batch, problem: EllipseProblem, box: EllipseBox,
                      lam_area: float = 120.0, lam_feas: float = 100.0):
    """Differenzierbare Soft-Penalty in der Latent-Suche. Bei
    ``fixed_area=True`` entfällt der Flächenterm (Fläche ist per
    Konstruktion exakt) und die Box-Penalty wirkt nur auf die 4
    unabhängigen Parameter ``(xc, yc, rx, θ)``."""
    if problem.fixed_area:
        bounds = torch.tensor([[box.xc[0], box.yc[0], box.rx[0], box.th[0]],
                                [box.xc[1], box.yc[1], box.rx[1], box.th[1]]],
                              dtype=a_batch.dtype, device=a_batch.device)
        a4 = torch.stack([a_batch[..., 0], a_batch[..., 1],
                          a_batch[..., 2], a_batch[..., -1]], dim=-1)
        over = torch.relu(a4 - bounds[1]) ** 2 + torch.relu(bounds[0] - a4) ** 2
        return lam_feas * over.sum(dim=-1)
    rx = a_batch[..., 2]; ry = a_batch[..., 3]
    target = problem.target_hole_area
    pen_area = lam_area * ((math.pi * rx * ry - target) / target) ** 2
    bounds = torch.tensor([[box.xc[0], box.yc[0], box.rx[0], box.ry[0], box.th[0]],
                            [box.xc[1], box.yc[1], box.rx[1], box.ry[1], box.th[1]]],
                          dtype=a_batch.dtype, device=a_batch.device)
    over = torch.relu(a_batch - bounds[1]) ** 2 + torch.relu(bounds[0] - a_batch) ** 2
    return pen_area + lam_feas * over.sum(dim=-1)


def _project_box_e1(a_batch, box: EllipseBox, *, fixed_area: bool = True):
    """Projektion in die Box-Constraints. Bei ``fixed_area=True`` wird
    nur auf die 4 unabhängigen Parameter ``(xc, yc, rx, θ)`` projiziert;
    ``ry`` wird anschließend aus ``rx`` berechnet."""
    if fixed_area:
        bounds = torch.tensor([[box.xc[0], box.yc[0], box.rx[0], box.th[0]],
                                [box.xc[1], box.yc[1], box.rx[1], box.th[1]]],
                              dtype=a_batch.dtype, device=a_batch.device)
        return torch.clamp(a_batch, bounds[0], bounds[1])
    bounds = torch.tensor([[box.xc[0], box.yc[0], box.rx[0], box.ry[0], box.th[0]],
                            [box.xc[1], box.yc[1], box.rx[1], box.ry[1], box.th[1]]],
                          dtype=a_batch.dtype, device=a_batch.device)
    return torch.clamp(a_batch, bounds[0], bounds[1])


def _e1_search_seed(run_seed: int, cycle: int, purpose: int) -> int:
    """Stable 128-bit seed for one E1 run/cycle/purpose, without seed+t reuse.

    SeedSequence mixes the run seed and a fixed namespace with separate cycle
    and purpose identifiers. Purpose 0: search starts; 1: exploration;
    2: independent post-search test (cycle 0). No worker ID enters this key.
    """
    if any(not isinstance(v, (int, np.integer)) or v < 0
           for v in (run_seed, cycle, purpose)):
        raise ValueError("Random-stream identifiers must be nonnegative integers")
    sequence = np.random.SeedSequence(int(run_seed),
        spawn_key=(0x50495343, int(cycle), int(purpose)))
    return sum(int(word) << (32*i) for i, word in enumerate(sequence.generate_state(4)))


def _e1_stream_record(run_seed: int, cycle: int) -> dict:
    return {"policy": "seedsequence_run_cycle_purpose_v1", "run_seed": int(run_seed),
            "cycle": int(cycle), "starts_seed_hex": hex(_e1_search_seed(run_seed, cycle, 0)),
            "exploration_seed_hex": hex(_e1_search_seed(run_seed, cycle, 1))}


def _multistart_optimize(model, F_torch, problem: EllipseProblem,
                           box: EllipseBox, seed_designs,
                           *, M=16, jitter_frac=8, steps=200, lr=0.4,
                           lam_area=120.0, c_solid=1.0, seed=0,
                           jitter_scale=(3.0, 1.5, 0.5, 0.5, 0.1)):
    """Multi-Start-Latent-Optimierung am eingefrorenen Surrogat.

    Bei ``fixed_area=True``: Optimierung über 4D ``(xc, yc, rx, θ)``;
    Sinus/Kosinus werden erst für die Surrogatabfrage gebildet.
    Rückgabe: 5D-Designvektoren ``(xc, yc, rx, ry, θ)``."""
    rng = np.random.default_rng(seed)
    jitter_scale = np.asarray(jitter_scale, dtype=float)
    if jitter_scale.shape != (5,) or not np.all(np.isfinite(jitter_scale)) or np.any(jitter_scale < 0):
        raise ValueError("jitter_scale must contain five finite nonnegative values")
    starts = []
    if problem.fixed_area:
        if seed_designs:
            best = seed_designs[0]
            for _ in range(min(M, jitter_frac)):
                jitter = rng.normal(scale=jitter_scale[[0, 1, 2, 4]], size=4)
                clipped = np.clip(
                    np.array([best[0], best[1], best[2], best[4]]) + jitter,
                    [box.xc[0], box.yc[0], box.rx[0], box.th[0]],
                    [box.xc[1], box.yc[1], box.rx[1], box.th[1]])
                ry = problem.ry_from_rx(clipped[2])
                starts.append(np.array([clipped[0], clipped[1], clipped[2],
                                        ry, clipped[3]], dtype=np.float64))
        while len(starts) < M:
            a = _sample_design(rng, box, problem)
            if a is not None:
                starts.append(np.array(a, dtype=np.float64))
        # Optimise physical coordinates, encoding the angle inside the graph.
        # Every surrogate query therefore remains on sin²(theta)+cos²(theta)=1.
        A = torch.tensor(np.stack(starts[:M])[:, [0, 1, 2, 4]], dtype=torch.float32)
        def objective(a_4):
            encoded = torch.cat([a_4[:, :3], torch.sin(a_4[:, 3:4]),
                                  torch.cos(a_4[:, 3:4])], dim=1)
            return (_surrogate_compliance(model, encoded, F_torch)/c_solid
                    + _ellipse_penalty(a_4, problem, box))
        a_4, _ = _packet_adam(A, objective, steps=steps, lr=lr,
            project=lambda values: _project_box_e1(values, box, fixed_area=True))
        target = problem.target_hole_area
        ry = target / (math.pi * a_4[:, 2])
        return torch.stack([a_4[:, 0], a_4[:, 1], a_4[:, 2],
                            ry, a_4[:, 3]], dim=-1)
    # Legacy 5D path (fixed_area=False)
    if seed_designs:
        best = seed_designs[0]
        for _ in range(min(M, jitter_frac)):
            jitter = rng.normal(scale=jitter_scale, size=5)
            starts.append(np.clip(best + jitter,
                                   [box.xc[0], box.yc[0], box.rx[0], box.ry[0], box.th[0]],
                                   [box.xc[1], box.yc[1], box.rx[1], box.ry[1], box.th[1]]))
    while len(starts) < M:
        a = _sample_design(rng, box, problem)
        if a is not None:
            starts.append(np.array(a, dtype=np.float64))
    A = torch.tensor(np.stack([_design_to_input(s) for s in starts]),
                     dtype=torch.float32, requires_grad=True)
    def objective(A):
        a_5 = torch.stack([A[:, 0], A[:, 1], A[:, 2], A[:, 3], torch.atan2(A[:, 4], A[:, 5])], dim=-1)
        return _surrogate_compliance(model, A, F_torch)/c_solid + _ellipse_penalty(a_5, problem, box, lam_area=lam_area)
    A, _ = _packet_adam(A, objective, steps=steps, lr=lr)
    a5 = torch.stack([A[:, 0], A[:, 1], A[:, 2], A[:, 3],
                      torch.atan2(A[:, 4], A[:, 5])], dim=-1)
    return _project_box_e1(a5.detach(), box, fixed_area=False)


def _select_topk(model, F_torch, candidates, problem: EllipseProblem,
                  box: EllipseBox, k: int, c_solid: float = 1.0):
    """Wähle die ``k`` zulässigen Kandidaten mit dem niedrigsten
    Surrogat-Score (Compliance + Feasibility-Penalty)."""
    model.eval()
    with torch.no_grad():
        a_in = torch.tensor(np.stack([_design_to_input(c.numpy()) for c in candidates]),
                            dtype=torch.float32)
        J_hat = _surrogate_compliance(model, a_in, F_torch) / c_solid
        pen = _ellipse_penalty(candidates, problem, box)
    score = (J_hat + pen).numpy()
    feas = []
    for i, c in enumerate(candidates.numpy()):
        if _design_feasible_full(c, problem):
            feas.append((score[i], c))
    feas.sort(key=lambda t: t[0])
    return [c for _, c in feas[:k]]


# =====================================================================
# 7. SOLVER-CONTRACT — Ergebnis-Schema und abstrakte Basis
# =====================================================================

@dataclass
class SolverResult:
    """Vereinheitlichtes Ergebnis aller Solver.

    Die Felder sind so gewählt, dass sie für SIMP, DE und PISCO sinnvoll
    besetzt werden können; Solver-spezifische Zusatzinformation geht
    in das frei strukturierbare ``extra``-Dict.

    Attributes
    ----------
    best_compliance:
        Compliance des besten gefundenen Designs (FEM-validiert).
    best_design:
        Design-Repräsentation. SIMP — Dichtefeld ``(nely, nelx)``;
        DE — Parametervektor; PISCO — Parametervektor (Fall 1).
    fem_calls_total:
        Anzahl der Hochfidelitäts-Auswertungen (Effizienzmetrik).
    trajectory:
        Liste von ``(kumulative_FEM, best_so_far_compliance)`` —
        eine Zeile pro FEM-Auswertung; Eingabe für Verlauf-Plots.
    seed:
        Random-Seed des Laufs.
    wall_time_s:
        Wandzeit des ``solve``-Aufrufs in Sekunden.
    extra:
        Solver-spezifische Zusatzinformation (Surrogatfehler je Zyklus,
        SIMP-Multi-Start-Statistiken, DE-Konvergenznachricht, …).
    """
    best_compliance: float
    best_design: Any
    fem_calls_total: int
    trajectory: list[tuple[int, float]]
    seed: int
    wall_time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best_compliance_normalized(self) -> Optional[float]:
        """Normierte Compliance, falls ``extra['solid_compliance']`` gesetzt ist."""
        ref = self.extra.get("solid_compliance")
        return self.best_compliance / ref if ref else None

    def to_summary_dict(self) -> dict[str, Any]:
        """Kompakte Repräsentation für JSON-Serialisierung — nur die für
        Tabellen relevanten Skalare."""
        return {
            "best_compliance": float(self.best_compliance),
            "best_compliance_normalized": self.best_compliance_normalized,
            "fem_calls_total": int(self.fem_calls_total),
            "seed": int(self.seed),
            "wall_time_s": float(self.wall_time_s),
        }


# R2 evidence instrumentation. This records measured execution, not inferred costs.
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
import functools
import inspect
import platform
from datetime import datetime, timezone

# The numerical partition is independent of the number of concurrent workers.
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits, threadpool_info

NUMERICAL_PROTOCOL = "fixed_packets_ccw_seedsequence_v2"
TRAINING_PACKET_SIZE = 4
_EXECUTION_WORKERS = 4
_PACKET_EXECUTOR = ContextVar("pisco_packet_executor", default=None)


def configure_execution(workers: int = 4) -> None:
    """Configure CPU packet concurrency before constructing/running solvers.

    One native thread computes each fixed packet. All workers use identical
    kernels; only the scheduling of independent packets changes. Generator
    training and non-partitioned baseline operations stay on the coordinator.
    Concurrent solver calls in the same Python process are not supported.
    """
    global _EXECUTION_WORKERS
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if _PACKET_EXECUTOR.get() is not None:
        raise RuntimeError("Cannot change workers during a solver run")
    _EXECUTION_WORKERS = workers
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def _initialize_packet_worker():
    # Explicit per-thread initialization also resets stale local MKL limits.
    torch.init_num_threads()
    if torch.get_num_threads() != 1:
        raise RuntimeError("Packet kernels must use exactly one native thread")


@contextmanager
def _packet_execution():
    if _PACKET_EXECUTOR.get() is not None:
        yield
        return
    configure_execution(_EXECUTION_WORKERS)
    with threadpool_limits(limits=1), ThreadPoolExecutor(
            max_workers=_EXECUTION_WORKERS, initializer=_initialize_packet_worker) as executor:
        token = _PACKET_EXECUTOR.set(executor)
        try:
            yield
        finally:
            _PACKET_EXECUTOR.reset(token)


def _ordered_packets(function, items):
    """Barrier plus input-order results, never completion-order accumulation."""
    with _packet_execution():
        return list(_PACKET_EXECUTOR.get().map(function, items))


def _tensor_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _packet_adam(starts, objective, *, steps, lr, project=None):
    """Independent Adam search packets; no shared parameter.grad writes."""
    _progress_step("candidate search", 0, steps)
    def optimize(start):
        coordinates = start.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([coordinates], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(coordinates).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite candidate objective")
            coordinates.grad, = torch.autograd.grad(loss, coordinates)
            optimizer.step()
            if project is not None:
                with torch.no_grad():
                    coordinates.copy_(project(coordinates))
        with torch.no_grad():
            score = objective(coordinates).detach()
        return coordinates.detach(), score
    results = _ordered_packets(optimize, starts.split(1))
    _progress_step("candidate search", steps, steps)
    return torch.cat([item[0] for item in results]), torch.cat([item[1].reshape(-1) for item in results])


def execution_metadata():
    return {"protocol": NUMERICAL_PROTOCOL, "workers": _EXECUTION_WORKERS,
            "fem_coordinates": "x right, y up; Q4 nodes LL/LR/UR/UL counter-clockwise",
            "e1_random_streams": "SeedSequence(run_seed, spawn_key=(namespace, cycle, purpose))",
            "native_threads_per_packet": 1, "training_packet_size": TRAINING_PACKET_SIZE,
            "gradient_reduction": "fixed packet order after barrier",
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
            "machine": platform.machine(), "torch_build_sha256": hashlib.sha256(
                torch.__config__.show().encode()).hexdigest()}


R2_EVIDENCE_SCHEMA = "pisco-r2-evidence-v1"
_ACTIVE_AUDIT = ContextVar("pisco_audit", default=None)


_LIVE_PROGRESS = ContextVar("pisco_live_progress", default=None)


def _progress_cycle(current, total):
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.cycle_progress = {"current":int(current), "maximum":int(total)}


def _progress_step(label, completed, total):
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.progress = {"stage": audit.stage, "label": label,
                          "completed": int(completed), "total": int(total)}


class LiveProgress:
    """Read-only heartbeat: no RNG changes, FEM calls or tensor synchronisation.

    ETA applies only to the remaining seeds of this method, after two fresh
    completed seeds. Cached retrieval times never enter the estimate.
    """
    def __init__(self, root, group, method, seed, *, batch=None, durations=(), interval=30):
        import threading
        self.root = Path(root)
        self.group, self.method, self.seed = group, method, int(seed)
        self.batch, self.durations = batch, list(durations)
        self.interval = interval
        self.audit = None
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self.emit("running")

    def emit(self, state, *, error=None, cached=False):
        with self.lock:
            audit = self.audit
            elapsed = time.perf_counter() - self.started
            phase = audit.stage if audit is not None else "cache_lookup/setup"
            detail = dict(getattr(audit, "progress", {}))
            if detail.get("stage") != phase:
                detail = {}
            eta = None
            if self.batch and len(self.durations) >= 2:
                remaining = self.batch[1] - self.batch[0]
                mean = sum(self.durations) / len(self.durations)
                eta = remaining * mean + (max(0., mean-elapsed) if state == "running" else 0.)
            row = {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                   "state":state,"example":self.group,"method":self.method,"seed":self.seed,
                   "batch":self.batch,"phase":phase,"progress":detail,"elapsed_s":elapsed,
                   "cycle":dict(getattr(audit,"cycle_progress",{})),
                   "fem_calls_recorded":len(audit.events) if audit else 0,
                   "training_updates_completed":len(audit.training) if audit else 0,
                   "resumed":bool(audit.resumed) if audit else False,
                   "remaining_method_eta_s":eta,"eta_basis_fresh_seeds":len(self.durations),
                   "cached":cached,"error":error}
            step = ""
            if detail:
                fraction = detail["completed"] / max(1,detail["total"])
                filled = min(20,round(20*fraction))
                step = f" | {detail['label']} [{'#'*filled}{'.'*(20-filled)}] {detail['completed']}/{detail['total']}"
            batch = f" | seed {self.batch[0]}/{self.batch[1]}" if self.batch else ""
            cycle = row["cycle"]
            cycle_text = f" | cycle {cycle['current']}/{cycle['maximum']} max" if cycle else ""
            estimate = f" | method ETA ~{eta/60:.1f} min" if eta is not None else ""
            print(f"[{state.upper()}] {self.group}/{self.method} seed={self.seed}{batch}"
                  f" | {phase}{step}{cycle_text} | elapsed {elapsed/60:.1f} min"
                  f" | FEM {row['fem_calls_recorded']} | updates {row['training_updates_completed']}"
                  f"{estimate}" + (f" | {error}" if error else ""), flush=True)
            self.root.mkdir(parents=True,exist_ok=True)
            target = self.root / "progress.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row,indent=2)+"\n")
            temporary.replace(target)
            with (self.root / "progress.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(row)+"\n")

    def __enter__(self):
        self.token = _LIVE_PROGRESS.set(self)
        self.emit("running")
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        try:
            if kind is not None:
                self.emit("interrupted" if issubclass(kind,KeyboardInterrupt) else "failed",error=str(error))
        finally:
            _LIVE_PROGRESS.reset(self.token)


class RunAudit:
    """Exclusive wall-clock accounting and ordered FEM events for one solve.

    Nested scopes charge time only to the innermost scope. CPU execution is
    used by the current solvers; GPU extensions must synchronize their timers.
    Checkpoint snapshots retain the completed evidence across process restarts.
    """
    def __init__(self):
        self.started = self.last = time.perf_counter()
        self.offset = 0.0
        self.stage = "other"
        self.seconds = {}
        self.events = []
        self.training = []
        self.predictions = []
        self.test_predictions = []
        self.incumbents = []
        self.seen = set()
        self.resumed = False
        self.progress = {}

    def charge(self):
        now = time.perf_counter()
        self.seconds[self.stage] = self.seconds.get(self.stage, 0.0) + now - self.last
        self.last = now

    @contextmanager
    def scope(self, stage):
        self.charge()
        previous = self.stage
        self.stage = stage
        try:
            yield
        finally:
            self.charge()
            self.stage = previous

    def elapsed(self):
        return self.offset + time.perf_counter() - self.started

    def snapshot(self):
        self.charge()
        return {"schema": R2_EVIDENCE_SCHEMA, "exclusive_seconds": dict(self.seconds),
                "elapsed_s": self.elapsed(), "fem_events": list(self.events),
                "training": list(self.training), "predictions": list(self.predictions),
                "incumbents":list(self.incumbents), "test_predictions":list(self.test_predictions), "resumed": self.resumed}

    def restore(self, state):
        if not state:
            return
        self.offset = float(state["elapsed_s"])
        self.seconds = dict(state["exclusive_seconds"])
        self.events = list(state["fem_events"])
        self.training = list(state["training"])
        self.predictions = list(state["predictions"])
        self.test_predictions = list(state.get("test_predictions", []))
        self.incumbents = list(state.get("incumbents", []))
        self.seen = {event["design_sha256"] for event in self.events}
        self.started = self.last = time.perf_counter()
        self.resumed = True


@contextmanager
def audit_scope(stage):
    audit = _ACTIVE_AUDIT.get()
    if audit is None:
        yield
    else:
        with audit.scope(stage):
            yield


def _audited_method(stage, operation):
    @functools.wraps(operation)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(operation).bind(*args, **kwargs)
        model = next((bound.arguments.get(name) for name in ("model", "surrogate")
                      if isinstance(bound.arguments.get(name), nn.Module)), None)
        frozen = []
        if stage == "candidate_search" and model is not None:
            frozen = [(p, p.requires_grad) for p in model.parameters()]
            for p, _ in frozen:
                p.requires_grad_(False)
        try:
            with audit_scope(stage):
                value = operation(*args, **kwargs)
            audit = _ACTIVE_AUDIT.get()
            if stage == "surrogate_training" and audit is not None and isinstance(value, dict):
                audit.training.append({"update": len(audit.training) + 1,
                                       "elapsed_s": audit.elapsed(), **value})
            return value
        finally:
            for p, required in frozen:
                p.requires_grad_(required)
    return wrapped


def _append_trajectory(trajectory, point):
    trajectory.append(point)
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.incumbents.append({"logical_call":int(point[0]),"best_compliance":float(point[1]),
                                "elapsed_s":audit.elapsed(),"stage":audit.stage})


def _record_prediction(predicted, actual, cycle, density):
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.predictions.append({"cycle": int(cycle), "predicted_before_fem": float(predicted),
            "fem_compliance": float(actual), "relative_error": float(abs(predicted-actual)/max(abs(actual),1e-12)),
            "binary": bool(np.all((density == 0) | (density == 1))),
            "design_sha256": hashlib.sha256(np.asarray(density, dtype=np.float64).tobytes()).hexdigest(),
            "evaluation_role": "selected_candidate_before_training_inclusion"})


# Bounded, per-run stiffness cache. No FEM result is substituted by this cache.
_STIFFNESS_CACHE = OrderedDict()
_STIFFNESS_CACHE_BYTES = 0
_STIFFNESS_CACHE_LIMIT = 128 * 1024 * 1024


def _training_stiffness(nelx, nely, rho, penal=3.0, *, hx=1.0, hy=1.0,
                         thickness=1.0):
    global _STIFFNESS_CACHE_BYTES
    # Cache the original assembly, including the input precision: promoting
    # float32 densities changes rho**penal before K is converted to torch.
    # Even small changes here can alter the subsequent optimisation path.
    array = np.asarray(rho)
    key = (nelx, nely, penal, hx, hy, thickness, array.dtype.str,
           hashlib.sha256(array.tobytes()).hexdigest())
    if key in _STIFFNESS_CACHE:
        _STIFFNESS_CACHE.move_to_end(key)
        return _STIFFNESS_CACHE[key][0]
    with audit_scope("training_preparation"):
        matrix = _csr_to_torch(_assemble_K(nelx, nely, array, penal=penal,
                                          hx=hx, hy=hy, thickness=thickness))
    size = matrix._nnz() * (2 * 8 + 4)
    while _STIFFNESS_CACHE and _STIFFNESS_CACHE_BYTES + size > _STIFFNESS_CACHE_LIMIT:
        _, (_, removed) = _STIFFNESS_CACHE.popitem(last=False)
        _STIFFNESS_CACHE_BYTES -= removed
    if size <= _STIFFNESS_CACHE_LIMIT:
        _STIFFNESS_CACHE[key] = (matrix, size)
        _STIFFNESS_CACHE_BYTES += size
    return matrix


def _batched_equilibrium_loss(matrices, displacement, force, free, force_norm):
    """One block-diagonal sparse matmul; exactly the same mean residual loss.

    The block operator is prepared separately and can be reused for all steps.
    ``matrices`` may be the already prepared block operator.
    """
    block = matrices if isinstance(matrices, torch.Tensor) else _block_stiffness(matrices)
    residual = torch.sparse.mm(block, displacement.reshape(-1, 1)).reshape_as(displacement) - force
    return residual[:, free].square().sum(dim=1).mean() / (force_norm ** 2 + 1e-12)


def _block_stiffness(matrices):
    n = matrices[0].shape[0]
    indices = torch.cat([m.indices() + i*n for i, m in enumerate(matrices)], dim=1)
    values = torch.cat([m.values() for m in matrices])
    return torch.sparse_coo_tensor(indices, values, (len(matrices)*n,)*2).coalesce()


@dataclass(frozen=True)
class TrainingPolicy:
    """Explicit R2 experiment settings; defaults retain fixed-step training.

    Validation is reserved from the initial evaluated designs and never used
    in the loss. It is training-control data, NOT an independent test set.
    Adaptive settings are opt-in and have their own cache fingerprint.
    """
    retain_optimizer: bool = False
    validation_fraction: float = 0.0
    min_steps: int = 25
    patience: int = 40
    relative_min_delta: float = 1e-4
    normalize_compliance: bool = False
    update_steps: int | None = None
    batch_physics: bool = True
    independent_test_size: int = 16

    def __post_init__(self):
        if not 0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in [0, 0.5)")
        if self.independent_test_size < 0:
            raise ValueError("independent_test_size must be nonnegative")
        if self.min_steps < 1 or self.patience < 1:
            raise ValueError("minimum steps and patience must be positive")
        if self.update_steps is not None and self.update_steps < 1:
            raise ValueError("update_steps must be positive")


def _fit_field(model, inputs, targets, force, matrices, free, force_norm,
               *, steps, lr, lam_data, lam_phys, clip=1.0, physics_batch=None, scalar=False):
    """Shared field trainer for all three benchmarks, with auditable curves."""
    if steps < 1:
        raise ValueError("Training requires at least one step")
    audit = _ACTIVE_AUDIT.get()
    policy = getattr(audit, "training_policy", TrainingPolicy())
    def score(value):
        return value if scalar else value @ force
    updates = int(getattr(model, "_r2_updates", 0))
    if updates and policy.update_steps is not None:
        steps = min(steps, policy.update_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if policy.retain_optimizer and hasattr(model, "_r2_optimizer"):
        optimizer.load_state_dict(model._r2_optimizer)
    if not hasattr(model, "_r2_validation"):
        count = int(len(targets) * policy.validation_fraction)
        model._r2_validation = list(range(count)) if len(targets)-count >= 2 else []
    val = model._r2_validation
    train = [i for i in range(len(targets)) if i not in val]
    scale = max(float(targets[train].abs().mean()), 1e-12) if policy.normalize_compliance else 1.0
    # Fin uses a reproducible physics subset; E1/E2 use every training item.
    sampled_physics = physics_batch is not None and physics_batch < len(train)
    groups = [train[i:i+TRAINING_PACKET_SIZE] for i in range(0, len(train), TRAINING_PACKET_SIZE)]
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    fixed_blocks = {}
    if lam_phys and policy.batch_physics and not sampled_physics:
        with audit_scope("training_preparation"):
            fixed_blocks = {tuple(group): _block_stiffness([matrices[i] for i in group]) for group in groups}
    def gradient_packet(job):
        group, positions, block, nphysics = job
        prediction = model(inputs[group])
        data = ((score(prediction)-targets[group])/scale).square().sum()/len(train)
        physics = prediction.new_zeros(())
        if lam_phys and positions:
            if policy.batch_physics:
                physics = _batched_equilibrium_loss(block, prediction[positions], force, free, force_norm) * (len(positions)/nphysics)
            else:
                physics = torch.stack([
                    (torch.sparse.mm(matrices[group[j]], prediction[j:j+1].T).squeeze(-1)-force)[free]
                    .square().sum()/(force_norm**2+1e-12) for j in positions]).sum()/nphysics
        loss = lam_data*data + lam_phys*physics
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite field training loss")
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return float(data.detach()), float(physics.detach()), gradients
    rng = np.random.default_rng(0)
    best_val = math.inf
    stale = 0
    curve = []
    stop = "step_limit"
    for step in range(1, steps+1):
        _progress_step(f"training update {updates+1}", step-1, steps)
        optimizer.zero_grad(set_to_none=True)
        selected = ((rng.choice(train, physics_batch, replace=False).tolist()
                     if sampled_physics else train) if lam_phys else [])
        selected_set = set(selected)
        jobs = []
        with audit_scope("training_preparation"):
            for group in groups:
                chosen = [i for i in group if i in selected_set]
                positions = [group.index(i) for i in chosen]
                block = None
                if chosen and policy.batch_physics:
                    block = (fixed_blocks[tuple(group)] if not sampled_physics
                             else _block_stiffness([matrices[i] for i in chosen]))
                jobs.append((group, positions, block, len(selected)))
        pieces = _ordered_packets(gradient_packet, jobs)
        data_loss = physics_loss = 0.0
        for data, physics, gradients in pieces:
            data_loss += data
            physics_loss += physics
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    if parameter.grad is None:
                        parameter.grad = gradient.clone()
                    else:
                        parameter.grad.add_(gradient)
        gradient_sha256 = _tensor_digest(p.grad for p in parameters if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        model_sha256 = _tensor_digest(model.state_dict().values())
        del pieces
        validation = None
        if val:
            with torch.no_grad():
                validation = float(((score(model(inputs[val]))-targets[val])/scale).square().mean())
            if validation < best_val - policy.relative_min_delta*max(abs(best_val) if math.isfinite(best_val) else 1.0, 1e-12):
                best_val, stale = validation, 0
            else:
                stale += 1
            if stale and stale % max(1, policy.patience//2) == 0:
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr']*0.5, lr*1e-3)
        _progress_step(f"training update {updates+1}", step, steps)
        curve.append({"step":step, "data_loss":float(data_loss),
                      "physics_loss":float(physics_loss), "validation_loss":validation,
                      "learning_rate":optimizer.param_groups[0]['lr'],
                      "gradient_sha256":gradient_sha256, "model_sha256":model_sha256})
        if val and step >= policy.min_steps and stale >= policy.patience:
            stop = "validation_patience"
            break
    # The final iterate is retained; early stopping does not silently rewind Adam.
    model._r2_optimizer = optimizer.state_dict()
    model._r2_updates = updates+1
    with torch.no_grad():
        error = ((score(model(inputs[train]))-targets[train]).abs()/targets[train].abs().clamp_min(1e-12)).mean()
    report = {"final_data":curve[-1]['data_loss'], "final_phys":curve[-1]['physics_loss'],
        "surrogate_rel_err":float(error), "dataset_size":len(targets), "training_size":len(train),
        "validation_indices":list(val), "validation_role":"training_control_not_independent_test",
        "steps_executed":step,"stop_reason":stop,"target_scale":scale,
        "policy":dataclasses.asdict(policy),"lam_data":lam_data,"lam_phys":lam_phys,
        "numerical_protocol":NUMERICAL_PROTOCOL, "packet_size":TRAINING_PACKET_SIZE,
        "architecture":repr(model),"parameter_count":sum(p.numel() for p in model.parameters()),
        "curve":curve}
    return report


class BaseSolver(ABC):
    """Abstrakte Solver-Basis mit transparentem Result-Caching.

    Jeder konkrete Solver implementiert ``_compute`` (die kostenintensive
    Numerik). ``solve`` kapselt das in einen Datei-Cache:

    1. Aus ``(Solver-Klassenname, Config-Felder, Problem-Felder, Seed)``
       wird ein SHA-256-Fingerprint berechnet.
    2. Existiert eine Cache-Datei mit diesem Fingerprint, wird das
       gespeicherte ``SolverResult`` ohne Neuberechnung zurückgegeben.
    3. Andernfalls läuft ``_compute``; das Ergebnis wird unter dem
       Fingerprint persistiert.

    Cache-Granularität: Callable-Felder (Callbacks, Factories) werden
    ausgespart — sie sind nicht JSON-serialisierbar und beeinflussen in
    Default-Verwendung das numerische Ergebnis nicht.

    Parameters
    ----------
    problem:
        Problem-Definition (Mesh, BCs, Materialdaten).
    config:
        Solver-Hyperparameter (Dataclass).
    cache_dir:
        Cache-Verzeichnis. Default: ``./solver_cache`` (relativ zum
        Arbeitsverzeichnis beim Modul-Import).
    cache_enabled:
        ``False`` umgeht die Cache-Lookup-Logik (jeder ``solve``-Aufruf
        wird neu gerechnet, kein Schreiben).
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init=cls.__dict__.get("__init__")
        if original_init is not None:
            @functools.wraps(original_init)
            def measured_init(self, *args, **kwargs):
                if _ACTIVE_AUDIT.get() is not None:
                    return original_init(self,*args,**kwargs)
                construction=RunAudit()
                token=_ACTIVE_AUDIT.set(construction)
                try:
                    with audit_scope("construction"):
                        original_init(self,*args,**kwargs)
                    self._construction_evidence=construction.snapshot()
                finally:
                    _ACTIVE_AUDIT.reset(token)
            cls.__init__=measured_init
        stages = {"_train_surrogate": "surrogate_training",
                  "_latent_optimize": "candidate_search",
                  "_latent_multistart": "candidate_search",
                  "_select_topk": "candidate_selection",
                  "_initial_dataset": "initial_data",
                  "_build_initial_dataset": "initial_data",
                  "_generator_from_dataset": "generator_build",
                  "_build_diagnostic_holdout": "diagnostic_data",
                  "_diagnose_surrogate": "diagnostics"}
        for name, stage in stages.items():
            if name in cls.__dict__:
                setattr(cls, name, _audited_method(stage, cls.__dict__[name]))

    def __init__(self, problem: Any, config: Any, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        if _PACKET_EXECUTOR.get() is None:
            configure_execution(_EXECUTION_WORKERS)
        self.problem = problem
        self.config = config
        self.cache_dir: Path = Path(cache_dir) if cache_dir is not None \
            else DEFAULT_CACHE_DIR
        self.cache_enabled: bool = bool(cache_enabled)
        self.training_policy = TrainingPolicy()

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle (konkret) — kümmert sich um den Cache
    # ------------------------------------------------------------------

    def solve(self, *, seed: int) -> SolverResult:
        """Liefert das ``SolverResult`` — entweder aus dem Cache, oder
        nach frischer Berechnung. Bei Cache-Treffer wird
        ``extra['from_cache'] = True`` gesetzt."""
        if self.cache_enabled:
            path = self._cache_path(seed)
            if path.exists():
                try:
                    with open(path, "rb") as fh:
                        bundle = pickle.load(fh)
                    result: SolverResult = bundle["result"]
                    result.extra = {**result.extra, "from_cache": True,
                                     "cache_path": str(path)}
                    return result
                except (pickle.UnpicklingError, EOFError, KeyError) as exc:
                    import sys
                    print(f"[solvers] Cache-Datei beschädigt ({path}: "
                          f"{exc}). Wird neu berechnet.", file=sys.stderr)

        global _STIFFNESS_CACHE_BYTES
        _STIFFNESS_CACHE.clear()
        _STIFFNESS_CACHE_BYTES = 0
        audit = RunAudit()
        audit.training_policy = self.training_policy
        live = _LIVE_PROGRESS.get()
        previous_live_audit = live.audit if live is not None else None
        if live is not None:
            live.audit = audit
        token = _ACTIVE_AUDIT.set(audit)
        try:
            with _packet_execution():
                result = self._compute(seed=seed)
            evidence = audit.snapshot()
            evidence["numerical_execution"] = execution_metadata()
            evidence["construction"] = getattr(self,"_construction_evidence",None)
            evidence["configuration"] = self._cache_payload(seed)
            evidence["environment"] = {"python": platform.python_version(),
                "platform": platform.platform(), "numpy": np.__version__,
                "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "device": "cpu", "processor": platform.processor()}
            evidence["logical_search_fem_calls"] = int(result.fem_calls_total)
            evidence["actual_recorded_fem_calls"] = len(audit.events)
            result.extra["r2_evidence"] = evidence
            result.extra["from_cache"] = False
        finally:
            if live is not None and previous_live_audit is not None:
                live.audit = previous_live_audit
            _ACTIVE_AUDIT.reset(token)
            _STIFFNESS_CACHE.clear()
            _STIFFNESS_CACHE_BYTES = 0

        if self.cache_enabled:
            self._persist(result, seed=seed)
        return result

    # ------------------------------------------------------------------
    # Solver-spezifische Numerik (Subklassen-Pflicht)
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, *, seed: int) -> SolverResult:
        """Eigentliche Solver-Implementation; nur bei Cache-Miss aufgerufen."""

    # ------------------------------------------------------------------
    # Cache-Management
    # ------------------------------------------------------------------

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        """Stabile Repräsentation der Eingabe für den Cache-Key.
        Callable-Felder werden ausgespart, unbekannte Typen via ``repr``
        als Fallback serialisiert."""
        def _serialize(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                out: dict[str, Any] = {}
                for f in dataclasses.fields(obj):
                    v = getattr(obj, f.name)
                    if callable(v) and not isinstance(v, type):
                        continue
                    out[f.name] = _serialize(v)
                return out
            if isinstance(obj, (tuple, list)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            return repr(obj)

        return {
            "source_sha256": _checked_source_digest(),
            "executed_sources": {p.name:digest for p,digest in _SOURCE_GUARDS.items()},
            "evidence_schema": R2_EVIDENCE_SCHEMA,
            "numerical_execution": execution_metadata(),
            "training_policy": dataclasses.asdict(self.training_policy),
            "solver": type(self).__name__,
            "config": _serialize(self.config),
            "problem": _serialize(self.problem),
            "generator_box": _serialize(getattr(getattr(self,"generator",None),"box",None)),
            "seed": int(seed),
        }

    def _cache_key(self, seed: int) -> str:
        """SHA-256-Fingerprint des Eingabe-Snapshots."""
        canonical = json.dumps(self._cache_payload(seed),
                                sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, seed: int) -> Path:
        """Vollständiger Pfad zur Cache-Datei (Klassenname + Hash)."""
        key = self._cache_key(seed)
        return self.cache_dir / f"{type(self).__name__}_{key}.pkl"

    def _persist(self, result: SolverResult, *, seed: int) -> None:
        """Atomares Schreiben (write+rename) — abgebrochene Läufe
        hinterlassen keine halben Cache-Dateien."""
        path = self._cache_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "result": result,
            "fingerprint": self._cache_payload(seed),
            "timestamp": time.time(),
            "key": self._cache_key(seed),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def clear_cache(cls, cache_dir: Optional[Path] = None) -> int:
        """Lösche Cache-Dateien dieses Solver-Typs in ``cache_dir``.
        ``BaseSolver.clear_cache()`` löscht *alle* Solver-Caches."""
        d = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        if not d.exists():
            return 0
        if cls is BaseSolver:
            pattern = "*.pkl"
        else:
            pattern = f"{cls.__name__}_*.pkl"
        count = 0
        for f in d.glob(pattern):
            f.unlink()
            count += 1
        return count

    @staticmethod
    def _seed_everything(seed: int) -> np.random.Generator:
        """Setze alle relevanten RNG-Quellen (NumPy + Torch) auf denselben
        Seed; gibt einen NumPy-Generator zurück."""
        rng = np.random.default_rng(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
        except ImportError:
            pass
        return rng


# =====================================================================
# 8. PISCO — Generator-Hierarchie und Solver
# =====================================================================
#
# Der Generator ist *PISCO-spezifisch*: SIMP optimiert direkt das
# Dichtefeld, DE direkt den Parametervektor. Generatoren als externes
# Plugin würden die methodische Trennung aus Abschn. 4 verwischen —
# sie gehören in diesen Block.
# =====================================================================

class Generator(ABC):
    """Schnittstelle für Designgeneratoren in PISCO.

    Bildet einen niedrigdimensionalen Latentvektor $z$ auf eine
    vollständige Designrepräsentation ab. Macht Fall 1 (analytische
    Parametrisierung) und Fall 2 (gelernter Decoder) aus Sicht des
    Solvers austauschbar."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension $d_z$ des Latentraums."""

    def fit(self, dataset: list[dict]) -> None:
        """Initialer Fit aus dem Startdatensatz (Fall 2: SVD bzw. VAE-
        Training). Default ist ein No-op."""
        return None

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent → Design (Forward-Pass)."""

    @abstractmethod
    def sample_latents(self, n: int, rng: np.random.Generator) -> torch.Tensor:
        """``n`` Latentvektoren als Multi-Start-Saaten."""


class AnalyticEllipseGenerator(Generator):
    """Fall 1 — analytische Ellipsenparametrisierung.

    Der Latentvektor *ist* der Designparameter
    ``(xc, yc, rx, ry, theta)``. Keine lernbaren Parameter, ``fit``
    ist No-op."""

    def __init__(self, problem: EllipseProblem, box: EllipseBox):
        self.problem = problem
        self.box = box

    @property
    def latent_dim(self) -> int:
        return 5

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def sample_latents(self, n: int, rng: np.random.Generator) -> torch.Tensor:
        out = []
        for _ in range(n):
            a = _sample_design(rng, self.box, self.problem)
            if a is None:
                rx = rng.uniform(*self.box.rx)
                if self.problem.fixed_area:
                    ry = self.problem.ry_from_rx(rx)
                else:
                    ry = rng.uniform(*self.box.ry)
                a = (rng.uniform(*self.box.xc), rng.uniform(*self.box.yc),
                     rx, ry, rng.uniform(*self.box.th))
            out.append(np.asarray(a, dtype=np.float32))
        return torch.tensor(np.stack(out))


@dataclass
class PISCOConfig:
    """Hyperparameter des PISCO-Solvers.

    Defaults entsprechen exakt der Veröffentlichungs-Konfiguration für
    Beispiel~1 (Tab. 2 im Paper): ``T_cycles=12, n_top=8, n_init=60``,
    Surrogat-MLP mit ``surrogate_hidden=128`` Tanh-Units in
    ``surrogate_depth=4`` versteckten Schichten, ``surrogate_steps=400``
    Adam-Schritte pro Zyklus, Multi-Start mit ``M=16`` plus
    ``M_explore=256`` Random-Kandidaten und ``multistart_lr=0.4`` über
    ``multistart_steps=200`` Adam-Schritte, ``lam_data=1``,
    ``lam_phys=1e-2``.
    """
    T_cycles: int = 12
    n_init: int = 60
    n_top: int = 8

    # Multi-Start-Latent-Suche
    M: int = 16
    M_explore: int = 256
    multistart_steps: int = 200
    multistart_lr: float = 0.4
    multistart_jitter_scale: tuple[float, ...] = (3.0, 1.5, 0.5, 0.5, 0.1)

    # Surrogat-Training
    surrogate_steps: int = 400
    surrogate_lr: float = 2e-3
    lam_data: float = 1.0
    lam_phys: float = 1e-2
    grad_clip_norm: float = 1.0
    surrogate_hidden: int = 128
    surrogate_depth: int = 4

    # Konvergenz-basierte Termination — Abbruch sobald sich die beste
    # Compliance über ``convergence_patience`` aufeinanderfolgende
    # \\emph{aufgewärmte} Zyklen nicht mehr um mindestens
    # ``convergence_tol_rel`` (relativ zum aktuellen Bestwert) verbessert.
    # ``convergence_warmup`` Zyklen werden vorab vollständig durchlaufen,
    # damit das Surrogat sinnvoll trainiert ist, bevor Stagnation
    # gezählt wird. ``T_cycles`` bleibt als Sicherheits-Obergrenze; bei
    # ``convergence_patience=None`` ist Early-Stopping deaktiviert.
    convergence_tol_rel: float = 1e-4
    convergence_patience: Optional[int] = 3
    convergence_warmup: int = 3

    # Optionale Hooks
    surrogate_factory: Optional[Callable[..., nn.Module]] = None
    on_cycle_end: Optional[Callable[[dict], None]] = None


class PISCOSolver(BaseSolver):
    """PISCO-Solver für parametrische Designräume (Fall 1).

    Algorithmus-Skizze (vgl. Abb. 4 / Abschn. 4 des Papers)
    --------------------------------------------------------
    1. **Initialdatensatz**: ``n_init`` zufällige zulässige Designs + FEM.
    2. **Generator**: einmal aus $\\mathcal{D}$ gefittet (No-op für
       analytische Generatoren).
    3. Für jeden Zyklus $t = 1, \\dots, T$:
       a. Surrogat trainieren (Daten + PDE-Residuum).
       b. Latent-Multi-Start auf eingefrorenem Kritiker.
       c. Top-$n_\\text{top}$ Kandidaten selektieren.
       d. FEM-Revalidierung → $\\mathcal{D}$ erweitern.
    4. Bestes FEM-validiertes Design zurückgeben.

    Erweiterung auf Fall 2 (Topologie): analoge Klasse mit
    PCA-/VAE-Generator und CNN-Surrogat — Loop-Struktur identisch.
    """

    def __init__(self, problem: EllipseProblem, config: PISCOConfig, *,
                 generator: Generator,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self.generator = generator

        # FEM-Geometrie und Last hängen am Problem, nicht an Zyklen —
        # einmal vorab bauen.
        F_np, fixed = _cantilever_load(
            problem.nelx, problem.nely,
            Lx=problem.Lx_mm, Ly=problem.Ly_mm,
            load_y1=problem.load_y1, load_y2=problem.load_y2,
            q_line=problem.q_line, thickness=problem.thickness_mm)
        self._F_np: np.ndarray = F_np
        self._F_torch: torch.Tensor = torch.tensor(F_np, dtype=torch.float32)
        self._fixed_dofs: np.ndarray = fixed
        self._ndof: int = int(F_np.size)
        self._free_dofs: np.ndarray = np.setdiff1d(np.arange(self._ndof), fixed)
        self._solid_compliance: float = _solid_case(problem)["compliance"]

    # -----------------------------------------------------------------
    # Hauptablauf — direkt im Stil von Abschnitt 4 des Papers
    # -----------------------------------------------------------------

    def _compute(self, *, seed: int) -> SolverResult:
        """Vollständiger PISCO-Loop. Wird vom Cache-Wrapper
        ``BaseSolver.solve`` nur bei Cache-Miss aufgerufen."""
        t0 = time.perf_counter()
        rng = self._seed_everything(seed)

        # Phase 1: Initialdatensatz
        dataset, fem_calls, trajectory = self._build_initial_dataset(rng)
        self.generator.fit(dataset)
        surrogate = self._make_surrogate(dataset)
        K_caches = [self._assemble_design_K(s) for s in dataset] if self.config.lam_phys else []

        best = self._best_feasible(dataset)
        cycle_metrics: list[dict] = []

        # Convergence-tracking state — Anzahl aufeinanderfolgender Zyklen,
        # in denen sich die beste Compliance um weniger als
        # ``convergence_tol_rel * best`` verbessert hat.
        stagnation = 0
        actual_T = 0
        stop_reason = "T_cycles_cap"

        # Phase 2 — Aktive Zyklen mit Early-Stopping bei Konvergenz
        for t in range(1, self.config.T_cycles + 1):
            _progress_cycle(t, self.config.T_cycles)
            actual_T = t
            J_before = float(best["compliance"])

            train_info = self._train_surrogate(surrogate, dataset, K_caches)
            cands = self._latent_multistart(surrogate, best, dataset,
                                              run_seed=seed, cycle=t)
            top_k = self._select_topk(surrogate, cands)
            for a in top_k:
                with torch.no_grad():
                    prediction = float((surrogate(torch.tensor(_design_to_input(a))[None]) @ self._F_torch)[0])
                sol = _eval_fem(self.problem, a)
                _record_prediction(prediction, sol["compliance"], t, np.asarray(a))
                fem_calls += 1
                dataset.append(sol)
                if self.config.lam_phys:
                    K_caches.append(self._assemble_design_K(sol))
                if (_design_feasible_full(sol["a"], self.problem)
                        and sol["compliance"] < best["compliance"]):
                    best = sol
                _append_trajectory(trajectory, (fem_calls, best["compliance"]))

            metric = self._record_cycle(t, dataset, fem_calls, best,
                                         surrogate, train_info)
            metric["search_random_streams"] = _e1_stream_record(seed, t)
            cycle_metrics.append(metric)
            if self.config.on_cycle_end is not None:
                self.config.on_cycle_end(metric)

            # Convergence check: relative Verbesserung gegenüber J_before.
            # Erst nach Warmup zählen, damit das Surrogat aus
            # frühen Zyklen nicht fälschlich als "konvergiert" durchgeht.
            patience = self.config.convergence_patience
            if (patience is not None and patience > 0
                    and t > self.config.convergence_warmup):
                tol = self.config.convergence_tol_rel * max(J_before, 1e-12)
                improved = (J_before - float(best["compliance"])) > tol
                stagnation = 0 if improved else stagnation + 1
                if stagnation >= patience:
                    stop_reason = "convergence"
                    break

        # Independent post-search test: separate RNG, no feedback into selection/training.
        with audit_scope("independent_test"):
            test_rng = np.random.default_rng(_e1_search_seed(seed, 0, 2))
            audit = _ACTIVE_AUDIT.get()
            for _ in range(self.training_policy.independent_test_size):
                design = _sample_design(test_rng, self.generator.box, self.problem)
                while design is None:
                    design = _sample_design(test_rng, self.generator.box, self.problem)
                with torch.no_grad():
                    prediction = float((surrogate(torch.tensor(_design_to_input(design))[None]) @ self._F_torch)[0])
                sample = _eval_fem(self.problem, design)
                if audit is not None:
                    audit.test_predictions.append({"predicted":prediction,"actual":float(sample["compliance"]),
                        "representation":"parametric_ellipse","sampling":"independent feasible box samples", "design":np.asarray(design).tolist()})

        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=best["a"],
            fem_calls_total=fem_calls,
            trajectory=trajectory,
            seed=seed,
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "solid_compliance": self._solid_compliance,
                "cycle_metrics": cycle_metrics,
                "lam_phys": float(self.config.lam_phys),
                "n_init": int(self.config.n_init),
                "n_top": int(self.config.n_top),
                "T_cycles": int(self.config.T_cycles),
                "actual_T": int(actual_T),
                "stop_reason": stop_reason,
            },
        )

    # -----------------------------------------------------------------
    # Phasen-Methoden (privat) — entsprechen 1:1 den Schritten in §4
    # -----------------------------------------------------------------

    def _build_initial_dataset(self, rng: np.random.Generator
                                  ) -> tuple[list[dict], int,
                                              list[tuple[int, float]]]:
        """``n_init`` zulässige Zufallsdesigns + FEM-Auswertung."""
        dataset: list[dict] = []
        fem_calls = 0
        trajectory: list[tuple[int, float]] = []
        best_running = float("inf")
        while len(dataset) < self.config.n_init:
            a = _sample_design(rng, self.generator.box, self.problem)
            if a is None:
                continue
            sol = _eval_fem(self.problem, a)
            dataset.append(sol)
            fem_calls += 1
            if (_design_feasible_full(sol["a"], self.problem)
                    and sol["compliance"] < best_running):
                best_running = float(sol["compliance"])
            _append_trajectory(trajectory, (fem_calls, best_running))
        return dataset, fem_calls, trajectory

    def _make_surrogate(self, dataset: list[dict]) -> nn.Module:
        """Konstruiere das Surrogat-MLP mit harter Dirichlet-Maske.
        Eingabenormalisierung wird am Initialdatensatz gefittet und über
        Zyklen hinweg eingefroren."""
        A_np = np.stack([_design_to_input(s["a"]) for s in dataset])
        d_in = A_np.shape[1]
        a_mean = torch.tensor(A_np.mean(axis=0))
        a_std = torch.tensor(A_np.std(axis=0) + 1e-6)
        u_scale = float(np.std([np.linalg.norm(s["U"]) for s in dataset])
                        / math.sqrt(self._ndof))
        if self.config.surrogate_factory is not None:
            return self.config.surrogate_factory(
                d_in=d_in, d_out=self._ndof, fixed_dofs=self._fixed_dofs)
        return _make_field_surrogate(
            d_in=d_in, d_out=self._ndof,
            hidden=self.config.surrogate_hidden,
            depth=self.config.surrogate_depth,
            a_mean=a_mean, a_std=a_std, u_scale=u_scale,
            fixed_dofs=self._fixed_dofs)

    def _train_surrogate(self, surrogate: nn.Module, dataset: list[dict],
                          K_caches: list[Any]) -> dict[str, float]:
        """Surrogat-Update: Datenverlust + PDE-Residuum."""
        return _train_field_surrogate(
            surrogate, dataset, self._F_torch, K_caches,
            steps=self.config.surrogate_steps, lr=self.config.surrogate_lr,
            lam_data=self.config.lam_data, lam_phys=self.config.lam_phys,
            free_dofs=self._free_dofs, clip=self.config.grad_clip_norm)

    def _latent_multistart(self, surrogate: nn.Module, best: dict,
                            dataset: list[dict],
                            *, run_seed: int, cycle: int) -> torch.Tensor:
        """Multi-Start-Optimierung im Latentraum + Box-Sampler-Exploration."""
        cand = _multistart_optimize(
            surrogate, self._F_torch, self.problem,
            self.generator.box,
            seed_designs=[best["a"]] +
                          [r["a"] for r in
                            sorted(dataset, key=lambda s: s["compliance"])[:5]],
            M=self.config.M, jitter_frac=8,
            steps=self.config.multistart_steps,
            lr=self.config.multistart_lr,
            c_solid=self._solid_compliance,
            seed=_e1_search_seed(run_seed, cycle, 0),
            jitter_scale=self.config.multistart_jitter_scale)
        rng2 = np.random.default_rng(_e1_search_seed(run_seed, cycle, 1))
        explore = []
        while len(explore) < self.config.M_explore:
            a = _sample_design(rng2, self.generator.box, self.problem)
            if a is not None:
                explore.append(np.array(a, dtype=np.float32))
        return torch.cat([cand, torch.tensor(np.stack(explore))], dim=0)

    def _select_topk(self, surrogate: nn.Module,
                      candidates: torch.Tensor) -> list[np.ndarray]:
        """Wähle die ``n_top`` vielversprechendsten *zulässigen* Kandidaten."""
        return _select_topk(
            surrogate, self._F_torch, candidates,
            self.problem, self.generator.box, self.config.n_top,
            c_solid=self._solid_compliance)

    # -----------------------------------------------------------------
    # Diagnostik-Helfer
    # -----------------------------------------------------------------

    def _record_cycle(self, cycle: int, dataset: list[dict], fem_calls: int,
                       best: dict, surrogate: nn.Module,
                       train_info: dict[str, float]) -> dict:
        """Erfasst Surrogatfehler und Best-so-Far-Compliance je Zyklus —
        diese Werte gehen u. a. in $\\varepsilon_S$ in Tab. 2 und in
        die Trajektorienplots."""
        with torch.no_grad():
            A_t = torch.tensor(np.stack(
                [_design_to_input(s["a"]) for s in dataset]),
                dtype=torch.float32)
            J_pred = (surrogate(A_t) @ self._F_torch).numpy()
            J_true = np.array([s["compliance"] for s in dataset])
            surr_err = float(np.mean(np.abs(J_pred - J_true)
                                       / np.maximum(np.abs(J_true), 1e-9)))
        return {
            "cycle": cycle,
            "n_dataset": len(dataset),
            "fem_calls": fem_calls,
            "best_compliance": float(best["compliance"]),
            "best_normalized": float(best["compliance"] / self._solid_compliance),
            "surrogate_rel_err": surr_err,
            "surrogate_loss_data": train_info["final_data"],
            "surrogate_loss_phys": train_info["final_phys"],
        }

    def _assemble_design_K(self, sol: dict) -> torch.Tensor:
        """Steifigkeitsmatrix der Dichteverteilung als sparse Torch-Tensor."""
        K = _assemble_K(self.problem.nelx, self.problem.nely,
                         sol["rho"], penal=self.problem.penal,
                         E0=self.problem.E0, Emin=self.problem.Emin,
                         nu=self.problem.nu, hx=self.problem.Lx_mm/self.problem.nelx,
                         hy=self.problem.Ly_mm/self.problem.nely,
                         thickness=self.problem.thickness_mm)
        return _csr_to_torch(K)

    def _best_feasible(self, dataset: list[dict]) -> dict:
        """Bester zulässiger Lauf; Fallback auf Compliance-Minimum, falls
        keiner zulässig (theoretisch möglich, hier praktisch nie)."""
        feas = [s for s in dataset
                if _design_feasible_full(s["a"], self.problem)]
        return min(feas if feas else dataset, key=lambda s: s["compliance"])


# =====================================================================
# 8b. PISCO-PINN — kontinuierliches Surrogat mit Raumkollokation
# =====================================================================
#
# Variante von PISCO mit *kontinuierlichem* Surrogat
# $u_\\omega(a, x): \\mathbb{R}^d \\times \\Omega \\to \\mathbb{R}^2$.
# Der entscheidende Unterschied zur Mesh-DOF-Variante:
#
#   * **Daten-Loss** wirkt nur an den Mesh-Knoten, wo $u_\\text{FEM}$
#     als Ground-Truth vorliegt.
#   * **Physik-Loss** wird an *zufällig gesampelten Raumpunkten*
#     $x_k \\in \\Omega$ ausgewertet, die gerade *nicht* in den
#     Trainingsdaten stecken.
#
# Damit ist der Physik-Term nicht redundant zur Daten-Loss: an
# Kollokationspunkten zwischen den Mesh-Knoten gibt es kein direktes
# Daten-Feedback, dort liefert die Physik die einzige
# Strukturinformation. Klassische PINN-Idee, übertragen auf den
# PISCO-Workflow.
#
# Strong-Form-Residuum für plane stress mit Material $E(\\rho(a, x))$:
#   $\\sigma_{ij,j}(x) + b_i(x) = 0$ in $\\Omega$
# mit $b = 0$ in unseren Beispielen.
# =====================================================================


def _node_coords(nelx: int, nely: int, Lx: float, Ly: float) -> np.ndarray:
    """Mesh-Knotenkoordinaten in physikalischen Einheiten (mm); Reihenfolge
    spaltenweise wie ``_prepare_fem`` sie nummeriert (column-major)."""
    xs = np.linspace(0.0, Lx, nelx + 1)
    ys = np.linspace(0.0, Ly, nely + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.flatten(order="F"), Y.flatten(order="F")], axis=-1)


def _ellipse_density_torch(a_input: torch.Tensor, x: torch.Tensor,
                             problem: EllipseProblem) -> torch.Tensor:
    """Differenzierbares Dichtefeld $\\rho(a, x)$ für die Ellipse.

    ``a_input`` ist die 6-dim Surrogat-Eingabe ``(xc, yc, rx, ry, sin θ,
    cos θ)``; ``x`` sind Raumpunkte (in mm). Liefert $\\rho \\in
    [\\rho_\\text{void}, 1]$ als glatte tanh-Funktion des Levelsets.
    Die Funktion ist in beiden Argumenten differenzierbar — Voraussetzung
    für die Strong-Form-Residual-Auswertung."""
    xc = a_input[..., 0]; yc = a_input[..., 1]
    rx = a_input[..., 2]; ry = a_input[..., 3]
    st = a_input[..., 4]; ct = a_input[..., 5]
    xx = x[..., 0]; yy = x[..., 1]
    xt = ct * (xx - xc) + st * (yy - yc)
    yt = -st * (xx - xc) + ct * (yy - yc)
    phi = (xt / rx) ** 2 + (yt / ry) ** 2 - 1.0
    rho = problem.rho_void + (1.0 - problem.rho_void) * 0.5 * (
        1.0 + torch.tanh(problem.beta * phi))
    return rho


class _ContinuousFieldSurrogate(nn.Module):
    """Kontinuierliches Surrogat $u_\\omega(a, x)$ für plane Elastizität.

    Eingabe: Designparameter (Surrogat-Form, 6-dim) + Raumkoordinate
    (2-dim, normiert). Ausgabe: Verschiebungsvektor $u = (u_x, u_y)$.

    Hard-Dirichlet auf der linken Kante ($x=0$) wird über einen
    multiplikativen Gate $g(x) = x/L_x$ erzwungen — damit ist
    $u_\\omega(a, 0, y) = 0$ exakt, ohne dass die Loss das einüben muss.
    """

    def __init__(self, d_design: int, Lx: float, Ly: float,
                 hidden: int = 128, depth: int = 4):
        super().__init__()
        self.Lx = float(Lx); self.Ly = float(Ly)
        self.register_buffer("a_mean", torch.zeros(d_design))
        self.register_buffer("a_std", torch.ones(d_design))
        layers: list[nn.Module] = [nn.Linear(d_design + 2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def set_normalization(self, a_mean: torch.Tensor, a_std: torch.Tensor):
        """Eingabenormalisierung am Initialdatensatz fitten (einmalig)."""
        self.a_mean.copy_(a_mean)
        self.a_std.copy_(a_std)

    def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``a`` und ``x`` haben dieselbe Batch-Dimension ``N``;
        gibt $u(a_i, x_i)$ als ``(N, 2)`` zurück.

        Für Auswertung an einem Designgitter aus Designs × Raumpunkten
        nutzt der Aufrufer ``_eval_grid`` (siehe unten)."""
        an = (a - self.a_mean) / self.a_std
        x_norm = torch.stack([x[..., 0] / self.Lx,
                                x[..., 1] / self.Ly - 0.5], dim=-1)
        inp = torch.cat([an, x_norm], dim=-1)
        u_raw = self.net(inp)
        gate = (x[..., 0:1] / self.Lx).clamp(min=0.0)
        return gate * u_raw


def _eval_grid(model: _ContinuousFieldSurrogate, a_batch: torch.Tensor,
               x_grid: torch.Tensor) -> torch.Tensor:
    """Evaluiere das Surrogat an allen Paaren ``(a_i, x_k)`` aus
    ``a_batch (B, d)`` und ``x_grid (M, 2)``. Liefert ``(B, M, 2)``."""
    B = a_batch.shape[0]
    M = x_grid.shape[0]
    a_rep = a_batch.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
    x_rep = x_grid.unsqueeze(0).expand(B, -1, -1).reshape(B * M, 2)
    u = model(a_rep, x_rep)
    return u.reshape(B, M, 2)


def _pinn_strong_form_residual(model: _ContinuousFieldSurrogate,
                                  a_pair: torch.Tensor,
                                  x_pair: torch.Tensor,
                                  problem: EllipseProblem) -> torch.Tensor:
    """Strong-Form-Residuum $\\nabla\\cdot\\sigma$ an den Paaren $(a_i, x_i)$
    bei plane stress, isotrop, mit $E(\\rho(a, x))$.

    Liefert den mittleren quadrierten Residuumsbetrag über die ``N`` Paare.
    Body-Force $b = 0$ in unseren Beispielen."""
    x = x_pair.detach().clone().requires_grad_(True)
    u = model(a_pair, x)  # (N, 2)
    # ∂u_α/∂x_β für α, β ∈ {x, y}
    grad_ux = torch.autograd.grad(u[:, 0].sum(), x, create_graph=True)[0]  # (N, 2)
    grad_uy = torch.autograd.grad(u[:, 1].sum(), x, create_graph=True)[0]  # (N, 2)
    eps_xx = grad_ux[:, 0]
    eps_yy = grad_uy[:, 1]
    eps_xy = 0.5 * (grad_ux[:, 1] + grad_uy[:, 0])
    # Lokales Material
    rho = _ellipse_density_torch(a_pair, x, problem)
    E = problem.Emin + (problem.E0 - problem.Emin) * rho.pow(problem.penal)
    nu = problem.nu
    fac = E / (1.0 - nu * nu)
    sigma_xx = fac * (eps_xx + nu * eps_yy)
    sigma_yy = fac * (eps_yy + nu * eps_xx)
    sigma_xy = E / (1.0 + nu) * eps_xy
    # ∂σ/∂x über autograd je Komponente
    grad_sxx = torch.autograd.grad(sigma_xx.sum(), x, create_graph=True)[0]
    grad_sxy = torch.autograd.grad(sigma_xy.sum(), x, create_graph=True)[0]
    grad_syy = torch.autograd.grad(sigma_yy.sum(), x, create_graph=True)[0]
    # Equilibrium ohne Body-Force
    res_x = grad_sxx[:, 0] + grad_sxy[:, 1]
    res_y = grad_sxy[:, 0] + grad_syy[:, 1]
    # Normierung auf charakteristische Spannung E0
    return ((res_x ** 2 + res_y ** 2)).mean() / (problem.E0 ** 2)


def _pinn_neumann_residual(model: _ContinuousFieldSurrogate,
                              a_pair: torch.Tensor, x_pair: torch.Tensor,
                              problem: EllipseProblem,
                              t_target: torch.Tensor) -> torch.Tensor:
    """Neumann-Residuum auf einem Stück Rand: $\\sigma \\cdot n - t = 0$.

    Wir gehen von der rechten Kante des Cantilevers aus (äußere Normale
    $n = (1, 0)$). Damit ist die Spaltenforderung an $\\sigma \\cdot n$:
    $\\sigma_{xx} = t_x$, $\\sigma_{xy} = t_y$. ``t_target`` ist
    ``(t_x, t_y)`` als Skalar-Tuple oder ``(2,)``-Tensor; wird auf alle
    ``N`` Paare gebroadcastet."""
    x = x_pair.detach().clone().requires_grad_(True)
    u = model(a_pair, x)
    grad_ux = torch.autograd.grad(u[:, 0].sum(), x, create_graph=True)[0]
    grad_uy = torch.autograd.grad(u[:, 1].sum(), x, create_graph=True)[0]
    eps_xx = grad_ux[:, 0]
    eps_yy = grad_uy[:, 1]
    eps_xy = 0.5 * (grad_ux[:, 1] + grad_uy[:, 0])
    rho = _ellipse_density_torch(a_pair, x, problem)
    E = problem.Emin + (problem.E0 - problem.Emin) * rho.pow(problem.penal)
    nu = problem.nu
    fac = E / (1.0 - nu * nu)
    sigma_xx = fac * (eps_xx + nu * eps_yy)
    sigma_xy = E / (1.0 + nu) * eps_xy
    res_x = sigma_xx - t_target[0]
    res_y = sigma_xy - t_target[1]
    return ((res_x ** 2 + res_y ** 2)).mean() / (problem.E0 ** 2)


def _train_pinn_surrogate(model: _ContinuousFieldSurrogate,
                            dataset: list[dict],
                            x_nodes: torch.Tensor,
                            x_bnd_quad: torch.Tensor,
                            t_y_bnd: float,
                            problem: EllipseProblem,
                            *, steps: int = 400, lr: float = 2e-3,
                            lam_data: float = 1.0, lam_phys: float = 1e-3,
                            M_int: int = 256,
                            rng: np.random.Generator = None
                            ) -> dict[str, float]:
    """Trainiert das kontinuierliche Surrogat mit *dreigeteilter*
    Loss-Architektur (vgl. methodische Diskussion in Abschn. 5.2):

    1. **Daten-Loss** an Mesh-Knoten —
       $u_\\omega(a, x_\\text{node}) \\approx U_\\text{FEM}$, exakt anchored.
    2. **Innere Physik-Kollokation** (zufällig in $\\Omega$) — strong-form
       $\\nabla \\cdot \\sigma = 0$. Sorgt für ein global konsistentes Feld
       und damit für Folge-Korrektheit der Rand-Vorhersagen.
    3. **Rand-Physik-Kollokation** (auf $\\Gamma_N$) — Neumann-Bedingung
       $\\sigma \\cdot n = t$. Kalibriert das Feld an den Stützstellen,
       die später in die Compliance-Bewertung als feinere Quadratur
       eingehen.

    ``x_bnd_quad`` ist die feste Stützstellen-Tabelle auf $\\Gamma_N$
    (wird auch in der Bewertung verwendet); ``t_y_bnd`` ist die
    y-Komponente der Traktion (in unseren Beispielen $-q_\\text{line}$).
    ``M_int`` steuert die Anzahl Innen-Kollokationspunkte je Schritt.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    A = torch.tensor(np.stack([_design_to_input(s["a"]) for s in dataset]),
                       dtype=torch.float32)
    n_node = int(x_nodes.shape[0])
    U_target = torch.tensor(
        np.stack([s["U"].reshape(n_node, 2) for s in dataset]),
        dtype=torch.float32)
    u_scale_sq = float((U_target ** 2).sum(dim=(-1, -2)).mean()) + 1e-12

    # Rand-Traktion als Tensor (für Broadcasting in der Neumann-Residuum-
    # Funktion verbleibt sie konstant pro Aufruf).
    t_target = torch.tensor([0.0, float(t_y_bnd)], dtype=torch.float32)

    n_bnd = int(x_bnd_quad.shape[0])

    final_data = final_phys = 0.0
    for _ in range(steps):
        optim.zero_grad()

        # 1) Daten-Loss an Mesh-Knoten
        u_hat_nodes = _eval_grid(model, A, x_nodes)
        loss_data = ((u_hat_nodes - U_target) ** 2).sum(dim=(-1, -2)).mean() \
                    / u_scale_sq

        # 2) Innen-Kollokation — pro Schritt frisch gesampelt
        x_int = torch.tensor(
            rng.uniform(low=[0.0, 0.0],
                          high=[problem.Lx_mm, problem.Ly_mm],
                          size=(M_int, 2)),
            dtype=torch.float32)
        a_int_pair = A.unsqueeze(1).expand(-1, M_int, -1).reshape(-1, A.shape[-1])
        x_int_pair = x_int.unsqueeze(0).expand(A.shape[0], -1, -1).reshape(-1, 2)
        loss_int = _pinn_strong_form_residual(model, a_int_pair, x_int_pair, problem)

        # 3) Rand-Kollokation — feste Stützstellen, Neumann-Bedingung
        a_bnd_pair = A.unsqueeze(1).expand(-1, n_bnd, -1).reshape(-1, A.shape[-1])
        x_bnd_pair = x_bnd_quad.unsqueeze(0).expand(A.shape[0], -1, -1).reshape(-1, 2)
        loss_bnd = _pinn_neumann_residual(model, a_bnd_pair, x_bnd_pair,
                                            problem, t_target)

        # Beide Physik-Anteile gleich gewichten unter lam_phys
        loss_phys = loss_int + loss_bnd

        loss = lam_data * loss_data + lam_phys * loss_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        final_data, final_phys = float(loss_data.item()), float(loss_phys.item())

    return {"final_data": final_data, "final_phys": final_phys}


def _pinn_compliance(model: _ContinuousFieldSurrogate,
                       a_batch: torch.Tensor,
                       x_quad: torch.Tensor,
                       t_quad_per_point: torch.Tensor) -> torch.Tensor:
    """Compliance $\\hat J(a) = \\int_{\\Gamma_N} u_\\omega(a, x) \\cdot t(x)\\,ds$.

    ``x_quad`` ist eine ``(n_q, 2)``-Tabelle Quadraturpunkte auf der
    Neumann-Randkurve, ``t_quad_per_point`` ist ``(n_q, 2)`` mit dem
    Quadraturgewicht-getragenen Lasttraktionsvektor an jedem Punkt
    (also $w_q \\cdot t(x_q)$, sodass die einfache Summe das Linienintegral
    approximiert).

    Im Diskret-Mesh-Fall reduziert sich das auf die klassische Form
    $U^T F$, wenn ``x_quad`` die Mesh-Knoten und ``t_quad_per_point`` die
    Knoten-Lastvektoren sind. Mit *feinerer* Quadratur als das FEM-Mesh
    werden Quadraturpunkte zwischen den Mesh-Knoten ausgewertet — dort
    ist der Daten-Loss inaktiv und Physik prägt die Surrogat-Vorhersage
    (vgl. Diskussion in Abschnitt 5.2 des Papers)."""
    u_load = _eval_grid(model, a_batch, x_quad)  # (B, n_q, 2)
    return (u_load * t_quad_per_point.unsqueeze(0)).sum(dim=(-1, -2))


@dataclass
class PISCOConfigPINN(PISCOConfig):
    """Erweiterte PISCO-Konfiguration für die PINN-Variante.

    Zusätzlich zur PISCO-Standard-Konfiguration:

    Attributes
    ----------
    M_coll:
        Anzahl Kollokationspunkte pro Trainingsschritt.
    surrogate_hidden_pinn / surrogate_depth_pinn:
        Architektur des kontinuierlichen Surrogats — typischerweise
        größer als die Mesh-DOF-Variante, weil ein höher-dimensionaler
        Eingaberaum bedient wird.
    """
    M_coll: int = 256
    surrogate_hidden_pinn: int = 128
    surrogate_depth_pinn: int = 4
    bewertung_n_quad: int = 0
    """Anzahl zusätzlicher Quadraturpunkte für die Bewertung auf
    $\\Gamma_N$. ``0`` (Default): Bewertung an Mesh-Knoten (klassisch).
    ``> 0``: feinere Trapez-Quadratur über das Last-Intervall, wodurch
    die Physik an Punkten zwischen den Mesh-Knoten Hebel bekommt."""


class PISCOSolverPINN(BaseSolver):
    """PISCO-Solver mit kontinuierlichem Surrogat und Raumkollokation.

    Identisch zu ``PISCOSolver`` im Workflow (Initialdatensatz →
    Surrogat-Training → Latent-Suche → FEM-Revalidierung), aber:

    * Das Surrogat ist eine Funktion von Designparameter *und* Raumpunkt.
    * Daten-Loss wirkt an Mesh-Knoten (Ground-Truth $u_\\text{FEM}$).
    * Physik-Loss wird an zufälligen Kollokationspunkten ausgewertet —
      genau dort, wo der Daten-Loss kein Feedback liefert. Damit ist der
      Physik-Term nicht redundant.
    """

    def __init__(self, problem: EllipseProblem, config: PISCOConfigPINN, *,
                 generator: Generator,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self.generator = generator

        # Mesh-Knoten + Lastvektor pro Knoten
        F_np, fixed = _cantilever_load(
            problem.nelx, problem.nely,
            Lx=problem.Lx_mm, Ly=problem.Ly_mm,
            load_y1=problem.load_y1, load_y2=problem.load_y2,
            q_line=problem.q_line, thickness=problem.thickness_mm)
        self._F_np: np.ndarray = F_np
        self._fixed_dofs: np.ndarray = fixed
        self._ndof: int = int(F_np.size)

        nodes = _node_coords(problem.nelx, problem.nely,
                              problem.Lx_mm, problem.Ly_mm)
        self._x_nodes_torch: torch.Tensor = torch.tensor(
            nodes, dtype=torch.float32)

        # Rand-Kollokation auf der Last-Strecke $\Gamma_N$ — gleichmäßig
        # zwischen ``load_y1`` und ``load_y2``. Diese Stützstellen werden
        # *gleichzeitig* für (a) das Neumann-Residuum-Training
        # ($\sigma \cdot n = t$) und (b) die feine Bewertungs-Quadratur
        # verwendet — so liegt jeder Bewertungspunkt im Trainings-Loop des
        # Surrogats. Anzahl: ``max(bewertung_n_quad, 16)`` Stützstellen,
        # also mindestens 16 selbst wenn der Nutzer keinen Wert setzt
        # (damit das Neumann-Residuum genug Auflösung hat).
        n_q = max(int(config.bewertung_n_quad), 16)
        ys = np.linspace(problem.load_y1, problem.load_y2, n_q)
        xs = np.full(n_q, problem.Lx_mm)
        x_bnd = np.stack([xs, ys], axis=-1)
        # Trapezgewichte
        dy = (problem.load_y2 - problem.load_y1) / max(n_q - 1, 1)
        w = np.full(n_q, dy)
        w[0] = w[-1] = 0.5 * dy
        # Stress traction is q_line/thickness; compliance quadrature uses the
        # line force q_line itself, consistently with the discrete FEM load.
        t_y = -problem.q_line / problem.thickness_mm
        t_quad = np.zeros((n_q, 2)); t_quad[:, 1] = -w * problem.q_line

        self._x_bnd_torch: torch.Tensor = torch.tensor(
            x_bnd, dtype=torch.float32)
        self._t_y_bnd: float = float(t_y)
        self._F_bnd_torch: torch.Tensor = torch.tensor(
            t_quad, dtype=torch.float32)

        # Bewertungs-Quadratur: entweder klassisch an Mesh-Knoten
        # (``bewertung_n_quad == 0``) — kompatibel zur diskreten
        # FEM-Compliance, oder feiner via Trapezregel auf $\Gamma_N$ —
        # die feinere Quadratur wertet das Surrogat zwischen Mesh-Knoten
        # aus, wo die Physik-Loss greift.
        if config.bewertung_n_quad > 0:
            self._x_load_nodes_torch: torch.Tensor = self._x_bnd_torch
            self._F_load_per_node_torch: torch.Tensor = self._F_bnd_torch
        else:
            F_per_node = F_np.reshape(-1, 2)
            load_mask = np.any(F_per_node != 0.0, axis=1)
            self._x_load_nodes_torch = torch.tensor(
                nodes[load_mask], dtype=torch.float32)
            self._F_load_per_node_torch = torch.tensor(
                F_per_node[load_mask], dtype=torch.float32)

        self._solid_compliance: float = _solid_case(problem)["compliance"]

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        rng = self._seed_everything(seed)

        # Phase 1: Initialdatensatz
        dataset, fem_calls, trajectory = self._build_initial_dataset(rng)
        self.generator.fit(dataset)

        # Surrogat aufsetzen + Eingabenormalisierung am Initialdatensatz fitten
        surrogate = self._make_surrogate(dataset)

        best = self._best_feasible(dataset)
        cycle_metrics: list[dict] = []

        for t in range(1, self.config.T_cycles + 1):
            _progress_cycle(t, self.config.T_cycles)
            train_info = _train_pinn_surrogate(
                surrogate, dataset, self._x_nodes_torch,
                self._x_bnd_torch, self._t_y_bnd, self.problem,
                steps=self.config.surrogate_steps,
                lr=self.config.surrogate_lr,
                lam_data=self.config.lam_data,
                lam_phys=self.config.lam_phys,
                M_int=self.config.M_coll,
                rng=rng)

            cands = self._latent_multistart(surrogate, best, dataset,
                                              run_seed=seed, cycle=t)
            top_k = self._select_topk(surrogate, cands)

            for a in top_k:
                sol = _eval_fem(self.problem, a)
                fem_calls += 1
                dataset.append(sol)
                if (_design_feasible_full(sol["a"], self.problem)
                        and sol["compliance"] < best["compliance"]):
                    best = sol
                _append_trajectory(trajectory, (fem_calls, best["compliance"]))

            metric = self._record_cycle(t, dataset, fem_calls, best,
                                         surrogate, train_info)
            cycle_metrics.append(metric)
            if self.config.on_cycle_end is not None:
                self.config.on_cycle_end(metric)

        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=best["a"],
            fem_calls_total=fem_calls,
            trajectory=trajectory,
            seed=seed,
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "solid_compliance": self._solid_compliance,
                "cycle_metrics": cycle_metrics,
                "lam_phys": float(self.config.lam_phys),
                "M_coll": int(self.config.M_coll),
                "n_init": int(self.config.n_init),
                "n_top": int(self.config.n_top),
                "T_cycles": int(self.config.T_cycles),
                "architecture": "PINN-continuous",
            },
        )

    def _build_initial_dataset(self, rng):
        dataset: list[dict] = []
        fem_calls = 0
        trajectory: list[tuple[int, float]] = []
        best_running = float("inf")
        while len(dataset) < self.config.n_init:
            a = _sample_design(rng, self.generator.box, self.problem)
            if a is None:
                continue
            sol = _eval_fem(self.problem, a)
            dataset.append(sol)
            fem_calls += 1
            if (_design_feasible_full(sol["a"], self.problem)
                    and sol["compliance"] < best_running):
                best_running = float(sol["compliance"])
            _append_trajectory(trajectory, (fem_calls, best_running))
        return dataset, fem_calls, trajectory

    def _make_surrogate(self, dataset):
        A_np = np.stack([_design_to_input(s["a"]) for s in dataset])
        d_in = A_np.shape[1]
        a_mean = torch.tensor(A_np.mean(axis=0), dtype=torch.float32)
        a_std = torch.tensor(A_np.std(axis=0) + 1e-6, dtype=torch.float32)
        model = _ContinuousFieldSurrogate(
            d_design=d_in, Lx=self.problem.Lx_mm, Ly=self.problem.Ly_mm,
            hidden=self.config.surrogate_hidden_pinn,
            depth=self.config.surrogate_depth_pinn)
        model.set_normalization(a_mean, a_std)
        return model

    def _latent_multistart(self, surrogate, best, dataset, *, run_seed, cycle):
        """Latent-Suche — nutzt PINN-Compliance als Kritiker-Score."""
        rng = np.random.default_rng(_e1_search_seed(run_seed, cycle, 0))
        box = self.generator.box
        starts = []
        seed_designs = [best["a"]] + [r["a"] for r in
                                        sorted(dataset, key=lambda s: s["compliance"])[:5]]
        if self.problem.fixed_area:
            if seed_designs:
                best_a = seed_designs[0]
                for _ in range(8):
                    jitter = rng.normal(scale=[3.0, 1.5, 0.5, 0.1], size=4)
                    clipped = np.clip(
                        np.array([best_a[0], best_a[1], best_a[2], best_a[4]]) + jitter,
                        [box.xc[0], box.yc[0], box.rx[0], box.th[0]],
                        [box.xc[1], box.yc[1], box.rx[1], box.th[1]])
                    ry = self.problem.ry_from_rx(clipped[2])
                    starts.append(np.array([clipped[0], clipped[1], clipped[2],
                                            ry, clipped[3]], dtype=np.float64))
            while len(starts) < self.config.M:
                a = _sample_design(rng, box, self.problem)
                if a is not None:
                    starts.append(np.array(a, dtype=np.float64))
            A = torch.tensor(np.stack([_design_to_input(s) for s in starts]),
                              dtype=torch.float32, requires_grad=True)
            optim = torch.optim.Adam([A], lr=self.config.multistart_lr)
            for search_step in range(self.config.multistart_steps):
                _progress_step("candidate search", search_step, self.config.multistart_steps)
                optim.zero_grad()
                a_4 = torch.stack([A[:, 0], A[:, 1], A[:, 2],
                                    torch.atan2(A[:, 3], A[:, 4])], dim=-1)
                J_hat = _pinn_compliance(surrogate, A,
                                           self._x_load_nodes_torch,
                                           self._F_load_per_node_torch) / self._solid_compliance
                pen = _ellipse_penalty(a_4, self.problem, box)
                loss = (J_hat + pen).sum()
                loss.backward()
                optim.step()
            a_4 = torch.stack([A[:, 0], A[:, 1], A[:, 2],
                                torch.atan2(A[:, 3], A[:, 4])], dim=-1)
            a_4 = _project_box_e1(a_4.detach(), box, fixed_area=True)
            ry = self.problem.target_hole_area / (math.pi * a_4[:, 2])
            cand = torch.stack([a_4[:, 0], a_4[:, 1], a_4[:, 2],
                                ry, a_4[:, 3]], dim=-1)
        else:
            if seed_designs:
                best_a = seed_designs[0]
                for _ in range(8):
                    jitter = rng.normal(scale=[3.0, 1.5, 0.5, 0.5, 0.1], size=5)
                    starts.append(np.clip(best_a + jitter,
                                            [box.xc[0], box.yc[0], box.rx[0],
                                             box.ry[0], box.th[0]],
                                            [box.xc[1], box.yc[1], box.rx[1],
                                             box.ry[1], box.th[1]]))
            while len(starts) < self.config.M:
                a = _sample_design(rng, box, self.problem)
                if a is not None:
                    starts.append(np.array(a, dtype=np.float64))
            A = torch.tensor(np.stack([_design_to_input(s) for s in starts]),
                              dtype=torch.float32, requires_grad=True)
            optim = torch.optim.Adam([A], lr=self.config.multistart_lr)
            for search_step in range(self.config.multistart_steps):
                _progress_step("candidate search", search_step, self.config.multistart_steps)
                optim.zero_grad()
                a_5 = torch.stack([A[:, 0], A[:, 1], A[:, 2], A[:, 3],
                                    torch.atan2(A[:, 4], A[:, 5])], dim=-1)
                J_hat = _pinn_compliance(surrogate, A,
                                           self._x_load_nodes_torch,
                                           self._F_load_per_node_torch) / self._solid_compliance
                pen = _ellipse_penalty(a_5, self.problem, box)
                loss = (J_hat + pen).sum()
                loss.backward()
                optim.step()
            a5 = torch.stack([A[:, 0], A[:, 1], A[:, 2], A[:, 3],
                                torch.atan2(A[:, 4], A[:, 5])], dim=-1)
            cand = _project_box_e1(a5.detach(), box, fixed_area=False)
        rng2 = np.random.default_rng(_e1_search_seed(run_seed, cycle, 1))
        explore = []
        while len(explore) < self.config.M_explore:
            a = _sample_design(rng2, box, self.problem)
            if a is not None:
                explore.append(np.array(a, dtype=np.float32))
        return torch.cat([cand, torch.tensor(np.stack(explore))], dim=0)

    def _select_topk(self, surrogate, candidates):
        surrogate.eval()
        with torch.no_grad():
            a_in = torch.tensor(np.stack([_design_to_input(c.numpy())
                                           for c in candidates]),
                                  dtype=torch.float32)
            J_hat = _pinn_compliance(surrogate, a_in,
                                       self._x_load_nodes_torch,
                                       self._F_load_per_node_torch) / self._solid_compliance
            pen = _ellipse_penalty(candidates, self.problem, self.generator.box)
        score = (J_hat + pen).numpy()
        feas = []
        for i, c in enumerate(candidates.numpy()):
            if _design_feasible_full(c, self.problem):
                feas.append((score[i], c))
        feas.sort(key=lambda t: t[0])
        return [c for _, c in feas[:self.config.n_top]]

    def _record_cycle(self, cycle, dataset, fem_calls, best, surrogate, train_info):
        with torch.no_grad():
            A_t = torch.tensor(np.stack(
                [_design_to_input(s["a"]) for s in dataset]), dtype=torch.float32)
            J_pred = _pinn_compliance(surrogate, A_t,
                                       self._x_load_nodes_torch,
                                       self._F_load_per_node_torch).numpy()
            J_true = np.array([s["compliance"] for s in dataset])
            surr_err = float(np.mean(np.abs(J_pred - J_true)
                                       / np.maximum(np.abs(J_true), 1e-9)))
        return {
            "cycle": cycle, "n_dataset": len(dataset), "fem_calls": fem_calls,
            "best_compliance": float(best["compliance"]),
            "best_normalized": float(best["compliance"] / self._solid_compliance),
            "surrogate_rel_err": surr_err,
            "surrogate_loss_data": train_info["final_data"],
            "surrogate_loss_phys": train_info["final_phys"],
        }

    def _best_feasible(self, dataset):
        feas = [s for s in dataset
                if _design_feasible_full(s["a"], self.problem)]
        return min(feas if feas else dataset, key=lambda s: s["compliance"])


# =====================================================================
# 8b. PISCO Fall 2 (Topology) — CNN-Surrogat + niedrigdim. Decoder
# =====================================================================

# ── Initial-Datensatz: kurze SIMP-Läufe aus zufälligen Startdichten ──

def _build_initial_lbracket_dataset(prob: LBracketProblem, n: int, seed: int,
                                      short_simp_iters: int = 6, init_amp: float = .20):
    """Compatibility entry point for physically volume-constrained SIMP anchors."""
    return _build_revision_simp_dataset(prob, n, seed, short_simp_iters, init_amp)


# ── Niedrigdimensionale Decoder ─────────────────────────────────────

def _make_conv_vae(dz: int = 16) -> nn.Module:
    """Konvolutionaler VAE für das L-Bracket-Dichtefeld ($60\\times 36$)."""

    class _ConvVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.dz = dz
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            feat = 64 * 5 * 8
            self.fc_mu  = nn.Linear(feat, dz)
            self.fc_lv  = nn.Linear(feat, dz)
            self.fc_dec = nn.Linear(dz, feat)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                     output_padding=(0, 0)), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                                     output_padding=(1, 1)), nn.ReLU(),
                nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1,
                                     output_padding=(1, 1)),
                nn.Sigmoid())

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_lv(h)

        def decode(self, z):
            h = self.fc_dec(z).view(-1, 64, 5, 8)
            return self.dec(h)

        def forward(self, x):
            mu, lv = self.encode(x)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            return self.decode(z), mu, lv

    return _ConvVAE()


def _train_vae(vae: nn.Module, fields: torch.Tensor, *,
                epochs: int = 200, lr: float = 3e-3, beta_kl: float = 1e-3
                ) -> dict:
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    final_recon = final_kl = 0.0
    for _ in range(epochs):
        optim.zero_grad()
        recon, mu, lv = vae(fields)
        recon = recon[..., : fields.shape[-2], : fields.shape[-1]]
        loss_recon = ((recon - fields) ** 2).mean()
        loss_kl    = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        (loss_recon + beta_kl * loss_kl).backward()
        optim.step()
        final_recon = float(loss_recon.item())
        final_kl    = float(loss_kl.item())
    return {"final_recon": final_recon, "final_kl": final_kl}


def _make_conv_ae(dz: int = 16) -> nn.Module:
    """Plain Conv-AE: gleiche Architektur wie der VAE, aber ohne
    KL-Regularisierung und ohne Reparametrisierung. Generator-Ablation."""

    class _ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.dz = dz
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            feat = 64 * 5 * 8
            self.fc_z   = nn.Linear(feat, dz)
            self.fc_dec = nn.Linear(dz, feat)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                     output_padding=(0, 0)), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                                     output_padding=(1, 1)), nn.ReLU(),
                nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1,
                                     output_padding=(1, 1)),
                nn.Sigmoid())

        def encode(self, x):
            return self.fc_z(self.enc(x))

        def decode(self, z):
            h = self.fc_dec(z).view(-1, 64, 5, 8)
            return self.dec(h)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    return _ConvAE()


def _train_ae(ae: nn.Module, fields: torch.Tensor, *,
               epochs: int = 200, lr: float = 3e-3) -> dict:
    optim = torch.optim.Adam(ae.parameters(), lr=lr)
    final_recon = 0.0
    for _ in range(epochs):
        optim.zero_grad()
        recon, _z = ae(fields)
        recon = recon[..., : fields.shape[-2], : fields.shape[-1]]
        loss_recon = ((recon - fields) ** 2).mean()
        loss_recon.backward()
        optim.step()
        final_recon = float(loss_recon.item())
    return {"final_recon": final_recon}


def _make_pca_decoder(fields_np: np.ndarray, dz: int = 16
                        ) -> tuple[nn.Module, dict]:
    """Linearer Decoder aus SVD der Initialfelder:
    ``rho_hat = mean + sum_i z_i * (s_i / sqrt(N-1)) * V_i``. Erfüllt das
    Decoder-Interface (``.decode(z)``, ``.dz``)."""
    N, H, W = fields_np.shape
    X = fields_np.reshape(N, -1).astype(np.float32)
    mean_np = X.mean(axis=0)
    Xc = X - mean_np
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    kmax = int(len(S))                                   # alle verfuegbaren Moden behalten
    k = int(min(dz, kmax))
    components_np = Vt[:kmax]
    scales_np = (S[:kmax] / np.sqrt(max(N - 1, 1))).astype(np.float32)

    class _PCADecoder(nn.Module):
        """Additiver (nested) PCA-Decoder: ``decode`` nutzt so viele Moden wie der
        Latentvektor lang ist; ``extend(new_dz)`` erhoeht die aktive Latentdim
        (gedeckelt durch ``max_dz``). Bestehende Moden bleiben erhalten -> ein um
        Nullen erweiterter z liefert dasselbe Design (perfekter Warmstart)."""

        def __init__(self):
            super().__init__()
            self.dz = k
            self.max_dz = kmax
            self.H, self.W = H, W
            self.register_buffer("mean", torch.tensor(
                mean_np.reshape(1, 1, H, W).astype(np.float32)))
            self.register_buffer("components", torch.tensor(
                components_np.astype(np.float32)))
            self.register_buffer("scales", torch.tensor(scales_np))

        def decode_raw(self, z):
            d = z.shape[-1]                              # additiv: nutze genau dim(z)
            scaled = z * self.scales[:d]
            flat   = scaled @ self.components[:d]
            return flat.view(-1, 1, self.H, self.W) + self.mean

        def decode(self, z):
            return self.decode_raw(z).clamp(0.0, 1.0)

        def extend(self, new_dz):
            """Aktive Latentdim erhoehen (gedeckelt durch max_dz). Gibt neue dz zurueck."""
            self.dz = int(min(max(new_dz, self.dz), self.max_dz))
            return self.dz

    return _PCADecoder(), {"final_recon": float(
        ((Xc - Xc @ Vt[:k].T @ Vt[:k]) ** 2).mean()), "max_dz": kmax}


def _volfrac_threshold(gm: torch.Tensor, vf: float, sharp: float = 8.0,
                       iters: int = 30) -> torch.Tensor:
    """Bisektion: Schwelle ``tau`` pro Sample so, dass der mittlere weiche
    Materialanteil ``sigmoid(sharp*(g - tau))`` im Designgebiet EXAKT ``vf`` trifft.

    Damit erfüllt jeder Latentvektor das Volumen-Constraint durch Konstruktion
    (kein Volumen-Drift, kein Strafterm nötig). ``gm`` enthält die Strukturwerte
    nur im Designgebiet, Form ``(B, n_mask)``; Rückgabe ``(B, 1)``.
    """
    lo = gm.min(dim=1, keepdim=True).values - 1.0
    hi = gm.max(dim=1, keepdim=True).values + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = torch.sigmoid(sharp * (gm - mid)).mean(dim=1, keepdim=True)
        lo = torch.where(v > vf, mid, lo)          # v fällt monoton mit tau
        hi = torch.where(v > vf, hi, mid)
    return 0.5 * (lo + hi)


class _VolumeProjection(torch.autograd.Function):
    """Euclidean projection onto [0,1]^n with a fixed sum, with exact VJP.

    Away from active-set transitions the free-cell Jacobian is I-11^T/n.
    Clipped cells have zero derivative. At a transition we choose the closed
    active-set subgradient (zero on the boundary); an all-clipped row has VJP 0.
    """

    @staticmethod
    def forward(ctx, active, volume):
        if active.ndim != 2 or not active.shape[1] or not 0. <= volume <= 1.:
            raise ValueError("Volume projection needs nonempty rows and a fraction in [0,1]")
        if not torch.isfinite(active).all():
            raise FloatingPointError("Non-finite volume projection input")
        lower = -active.amax(dim=1, keepdim=True)
        upper = 1. - active.amin(dim=1, keepdim=True)
        for _ in range(64):
            middle = (lower + upper) * .5
            below = (active + middle).clamp(0., 1.).mean(dim=1, keepdim=True) < volume
            lower = torch.where(below, middle, lower)
            upper = torch.where(below, upper, middle)
        output = (active + (lower + upper) * .5).clamp(0., 1.)
        if volume == 0. or volume == 1.:
            output = torch.full_like(active, volume)
        ctx.save_for_backward((output > 0.) & (output < 1.))
        return output

    @staticmethod
    def backward(ctx, upstream):
        free, = ctx.saved_tensors
        count = free.sum(dim=1, keepdim=True).clamp_min(1)
        average = (upstream * free).sum(dim=1, keepdim=True) / count
        return torch.where(free, upstream - average, torch.zeros_like(upstream)), None


def project_volume_torch(field, mask, volume):
    """Project each batch member on the mask; outside the mask remains zero."""
    flat = field.reshape(field.shape[0], -1)
    mask = torch.as_tensor(mask, device=field.device, dtype=torch.bool).reshape(-1)
    active = _VolumeProjection.apply(flat[:, mask], float(volume))
    output = torch.zeros_like(flat)
    output[:, mask] = active
    return output.reshape_as(field)


def top_k_overlap(actual, predicted, k):
    """Top-k set overlap; None if either membership is ambiguous at the cutoff.

    Internal ties wholly within/outside the selected set are harmless. Ties
    crossing the cutoff have no uniquely defined top-k set and are not broken
    by input order. Missing/nonfinite populations also have no defined score.
    """
    a, p = np.asarray(actual), np.asarray(predicted)
    if a.shape != p.shape or a.ndim != 1:
        raise ValueError("Expected equally sized one-dimensional score arrays")
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():
        return None
    k = min(int(k), len(a))
    if k < 1:
        return None
    for values in (a, p):
        ordered = np.sort(values)
        if k < len(values) and ordered[k-1] == ordered[k]:
            return None
    return len(set(np.argsort(a)[:k]) & set(np.argsort(p)[:k])) / k


class _VolumeProjected(nn.Module):
    """Hülle, die einen beliebigen Decoder volumen-treu macht: JEDER Latentvektor
    wird auf ein Design mit exaktem Materialanteil ``volfrac`` abgebildet
    (Shift+Clip-Projektion mit exakter aktiver Rückwärtsableitung).

    Damit ist das Volumen-Constraint durch Konstruktion erfüllt; die Latent-Suche
    benötigt KEINEN Volumen-Strafterm (Paper-Formulierung Fall 2: Constraints im
    Latentraum, nicht als Minimierungsterm)."""

    def __init__(self, inner: nn.Module, mask: np.ndarray, volfrac: float):
        super().__init__()
        self.inner = inner
        self.dz = int(inner.dz)
        self.max_dz = int(getattr(inner, "max_dz", inner.dz))
        self.vf = float(volfrac)
        self.register_buffer("maskflat",
                             torch.tensor(np.asarray(mask, bool).reshape(-1)))

    def extend(self, new_dz):
        """Aktive Latentdim erhöhen (delegiert an den inneren additiven Generator)."""
        if hasattr(self.inner, "extend"):
            self.dz = int(self.inner.extend(new_dz))
        return self.dz

    def _project(self, rho: torch.Tensor) -> torch.Tensor:
        return project_volume_torch(rho, self.maskflat, self.vf)

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder-Ausgabe vor Dichte-Clipping und Volumenprojektion."""
        raw_decoder = getattr(self.inner, "decode_raw", None)
        return raw_decoder(z) if raw_decoder is not None else self.inner.decode(z)

    def decode_with_diagnostics(self, z: torch.Tensor
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.decode_raw(z)
        bounded = raw.clamp(0.0, 1.0)
        projected = self._project(bounded)
        B = raw.shape[0]
        raw_flat = raw.reshape(B, -1)[:, self.maskflat]
        clipped_fraction = ((raw_flat < 0.0) | (raw_flat > 1.0)).float().mean(dim=1)
        diagnostics = {
            "raw_min": raw_flat.min(dim=1).values,
            "raw_max": raw_flat.max(dim=1).values,
            "clipped_fraction": clipped_fraction,
            "projected_volume": projected.reshape(B, -1)[:, self.maskflat].mean(dim=1),
            # Kept only until top-candidate selection; lets the common
            # post-processing protocol start from pre-projection variables.
            "bounded_raw_design": bounded,
        }
        return projected, diagnostics

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        projected, _ = self.decode_with_diagnostics(z)
        return projected


def _make_wavelet_decoder(nelx: int, nely: int, dz: int = 16,
                         levels: int = 5, slope: float = 6.0,
                         mask: Optional[np.ndarray] = None,
                         max_dz: Optional[int] = None) -> nn.Module:
    """Fixer, nicht-lernender Wavelet-Decoder mit vorgegebenen Atomen.

    Der Decoder baut aus einem latenten Vektor eine lineare Überlagerung von
    vorinitialisierten, Haar-ähnlichen Wavelet-Atomen auf (grob->fein geordnet)
    und projiziert das Ergebnis anschließend auf ``[0, 1]``. Differentiabel in ``z``.

    *Additiv/nested:* ``max_dz`` baut eine Reserve feinerer Atome; ``decode`` nutzt
    so viele wie der Latentvektor lang ist, ``extend(new_dz)`` erhöht die aktive
    Latentdim (feinere Skalen). Im Grenzfall vollständiger Skalen -> verlustfrei.
    """
    mask_np = (np.ones((nely, nelx), dtype=bool)
               if mask is None else np.asarray(mask, dtype=bool))
    cap = int(dz) if max_dz is None else max(int(max_dz), int(dz))

    # Haar-ähnliche Mutterwellen (Details). Die Skaliertion ist ein
    # Normalisierungsfaktor für einen stabilen Latentscale.
    mother = {
        "h": np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32) / 2.0,
        "v": np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32) / 2.0,
        "d": np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32) / 2.0,
    }
    starts = []
    for s in range(levels, 0, -1):
        block_h = max(2, 2 ** s)
        block_w = max(2, 2 ** s)
        ys = list(range(0, max(1, nely - block_h + 1), block_h))
        if nely > block_h and ys[-1] != nely - block_h:
            ys.append(nely - block_h)
        xs = list(range(0, max(1, nelx - block_w + 1), block_w))
        if nelx > block_w and xs[-1] != nelx - block_w:
            xs.append(nelx - block_w)
        starts.append((block_h, block_w, ys, xs))

    atoms: list[np.ndarray] = []
    # 1) Konstantes Niedrigfrequenzatom (damit das Modell nicht "nur
    #    Kontrast" optimiert).
    const_atom = np.ones((nely, nelx), dtype=np.float32)
    const_atom /= float(np.linalg.norm(const_atom) + 1e-12)
    atoms.append(const_atom.reshape(-1))

    # 2) Wavelet-Atome in gröber zu feiner Skala (coarse-first),
    #    bis die gewünschte latent_dim erreicht ist.
    for block_h, block_w, ys, xs in starts:
        for oy in ys:
            for ox in xs:
                rep_h = max(1, block_h // 2)
                rep_w = max(1, block_w // 2)
                for ori in ("h", "v", "d"):
                    atom_block = np.kron(mother[ori],
                                         np.ones((rep_h, rep_w), dtype=np.float32))
                    atom_block = atom_block[:block_h, :block_w]
                    atom = np.zeros((nely, nelx), dtype=np.float32)
                    atom[oy:oy + block_h, ox:ox + block_w] = atom_block[:min(block_h, nely - oy),
                                                                         :min(block_w, nelx - ox)]
                    atom *= mask_np
                    atom = atom.reshape(-1)
                    a_norm = float(np.linalg.norm(atom) + 1e-12)
                    atom = atom / a_norm
                    atoms.append(atom)
                    if len(atoms) >= cap:
                        break
                if len(atoms) >= cap:
                    break
            if len(atoms) >= cap:
                break
        if len(atoms) >= cap:
            break

    # 3) Sicherheitsnetz für kleine Auflösungen / zu kleine Basis:
    #    orthogonale Sinus-Kanäle ergänzen, damit gen.dz immer erreichbar ist.
    if len(atoms) < cap:
        yy = np.linspace(0.0, np.pi * 2.0, nely, endpoint=False)
        xx = np.linspace(0.0, np.pi * 2.0, nelx, endpoint=False)
        X, Y = np.meshgrid(xx, yy)
        k = len(atoms)
        phase = 0.0
        while len(atoms) < cap:
            f = 1 + k
            fallback = (np.cos(f * X + phase) + np.sin((f + 1) * Y + phase)).astype(np.float32)
            fallback = fallback * mask_np
            fallback = fallback.reshape(-1)
            n = float(np.linalg.norm(fallback) + 1e-12)
            atoms.append((fallback / n))
            phase += 0.17
            k += 1

    A = np.stack(atoms[:cap], axis=0).astype(np.float32)
    # Optional: robustere Spannweite / bessere Konditionierung in der Projektion
    nz = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / np.maximum(nz, 1e-12)
    atoms_t = torch.tensor(A[:, :], dtype=torch.float32)
    mask_t = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32).reshape(nely, nelx)
    mask_flat_t = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32).reshape(-1)

    class _WaveletDecoder(nn.Module):
        """Additiver (nested) Wavelet-Decoder: grob->fein geordnete Atome; ``decode``
        nutzt genau ``dim(z)`` Atome (ein um Nullen erweiterter z liefert dasselbe
        Design -> Warmstart), ``extend(new_dz)`` nimmt feinere Atome hinzu."""

        def __init__(self):
            super().__init__()
            self.dz = int(dz)
            self.max_dz = int(atoms_t.shape[0])
            self.slope = float(slope)
            self.register_buffer("atoms", atoms_t)
            self.register_buffer("mask", mask_t)
            self.register_buffer("mask_flat", mask_flat_t)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            d = z.shape[-1]                              # additiv: genau dim(z) Atome
            raw = z @ self.atoms[:d]                     # (B,d) × (d,H*W) -> (B,H*W)
            rho = torch.sigmoid(self.slope * raw) * self.mask_flat[None, :]
            return rho.view(z.shape[0], 1, nely, nelx)

        def extend(self, new_dz):
            """Aktive Latentdim erhöhen (feinere Skalen, gedeckelt durch max_dz)."""
            self.dz = int(min(max(new_dz, self.dz), self.max_dz))
            return self.dz

    return _WaveletDecoder()


def _make_lattice_wavelet_decoder(
    nelx: int,
    nely: int,
    dz: int = 16,
    mask: Optional[np.ndarray] = None,
    volfrac: float = 0.4,
    residual_weight: float = 0.18,
    wavelet_levels: int = 5,
) -> nn.Module:
    """Hybrid-Decoder: Lattice als tragfähige Grundstruktur + lokaler Wavelet-Rest.

    - Lattice liefert stabile, fast binäre Basistrukturen mit expliziter
      Volumensteuerung.
    - Wavelet liefert feine, lokal variierende Korrekturen.
    - Ergebnis wird erneut auf das Zielvolumen projiziert.
    """
    if dz < 2:
        raise ValueError(f"lattice_wavelet requires dz >= 2, got {dz}")
    if dz < 14:
        # Für sehr kleine latente Dimensionen bleibt nur die Wavelet-Familie stabil.
        return _make_wavelet_decoder(
            nelx, nely, dz=dz, levels=wavelet_levels, slope=6.0, mask=mask)
    nz = max(1, int(0.65 * dz))
    nz_w = max(1, dz - nz)
    if nz + nz_w != dz:
        nz = dz - nz_w
    # Lattice-Decoder benötigt mindestens 13 latente Achsen (3×4+1).
    # Restliche Achsen gehen an die Wavelet-Restkomponente.
    nz = min(max(13, nz), dz - 1)
    nz_w = max(1, dz - nz)
    lattice = _make_lattice_decoder(
        nelx, nely, dz=nz, mask=mask, volfrac=volfrac)
    wavelet = _make_wavelet_decoder(
        nelx, nely, dz=nz_w, levels=wavelet_levels, slope=6.0, mask=mask)

    class _LatticeWaveletDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.dz = int(dz)
            self.nz = int(nz)
            self.nz_w = int(nz_w)
            self.vf = float(volfrac)
            self.omega = float(residual_weight)
            self.register_buffer(
                "maskflat",
                torch.tensor(np.asarray(mask if mask is not None else np.ones((nely, nelx), bool),
                                       dtype=bool).reshape(-1)))

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            z_lat = z[:, : self.nz]
            z_wav = z[:, self.nz:self.nz + self.nz_w]
            rho_lat = lattice.decode(z_lat)[:, 0]
            rho_wav = wavelet.decode(z_wav)[:, 0]
            mixed = rho_lat + self.omega * (rho_wav - 0.5)
            mixed = torch.clamp(mixed, 0.0, 1.0)
            gm = mixed.reshape(z.shape[0], -1)[:, self.maskflat]
            with torch.no_grad():
                tau = _volfrac_threshold(gm, self.vf, sharp=8.0).view(z.shape[0], 1, 1)
            rho = torch.sigmoid(8.0 * (mixed - tau))
            return rho.unsqueeze(1)

    return _LatticeWaveletDecoder()


def _make_lattice_decoder(nelx: int, nely: int, dz: int = 16,
                          mask: Optional[np.ndarray] = None,
                          volfrac: float = 0.4) -> nn.Module:
    """Differenzierbarer Gitter-Decoder als alternativer Generator.

    Erzeugt aus dem latenten Vektor ein **gradiertes, orientiertes Gitter**
    (scharf von Bauart, kein Grau): ein grobes Orientierungsfeld plus globale
    Zellfrequenz. Im Gegensatz zu PCA/VAE wird NICHTS aus den Ankern gelernt —
    eine fixe parametrische Familie (keine trainierbaren Parameter). Geeignet für
    „diffuse" Probleme (verteilte Last/Lagerung), auf denen Continuous-Density-
    Decoder zu nicht-fertigbarem Grau kollabieren; dort ist das Optimum eine
    verteilte Gitterstruktur, die dieser Decoder direkt und fertigbar darstellt.

    **Volumen wird erzwungen:** die Strebendicke (Schwelle) wird pro Sample so
    gesetzt, dass der Materialanteil im Designgebiet exakt ``volfrac`` trifft —
    dadurch kein Volumen-Drift (im Gegensatz zur weichen Volumenstrafe).

    Erfordert ``dz >= gh*gw + 1`` (Default 3*4 + 1 = 13)."""
    gh, gw = 3, 4
    if dz < gh * gw + 1:
        raise ValueError(f"lattice decoder benötigt dz >= {gh*gw+1}, hat {dz}")
    mask_np = (np.ones((nely, nelx), bool) if mask is None
               else np.asarray(mask, bool))

    class _LatticeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.dz = dz
            self.gh, self.gw = gh, gw
            self._ny, self._nx = nely, nelx
            self.vf = float(volfrac)
            ys = torch.linspace(0.0, 1.0, nely)
            xs = torch.linspace(0.0, float(nelx) / nely, nelx)   # Seitenverhältnis
            Y, X = torch.meshgrid(ys, xs, indexing="ij")
            self.register_buffer("X", X.unsqueeze(0))            # (1, ny, nx)
            self.register_buffer("Y", Y.unsqueeze(0))
            self.register_buffer("maskflat",
                                 torch.tensor(mask_np.reshape(-1)))

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            B = z.shape[0]
            ng = self.gh * self.gw
            theta_c = z[:, :ng].view(B, 1, self.gh, self.gw) * math.pi
            theta = torch.nn.functional.interpolate(
                theta_c, size=(self._ny, self._nx),
                mode="bilinear", align_corners=False)[:, 0]      # (B, ny, nx)
            omega = (4.0 + 8.0 * torch.sigmoid(z[:, ng])).view(B, 1, 1)
            u = torch.cos(theta) * self.X + torch.sin(theta) * self.Y
            v = -torch.sin(theta) * self.X + torch.cos(theta) * self.Y
            # Nähe zur nächsten Strebe (kontinuierlich, nicht gesättigt):
            g = torch.maximum(torch.cos(2 * math.pi * omega * u),
                              torch.cos(2 * math.pi * omega * v))   # (B, ny, nx)
            # Schwelle pro Sample so, dass der Materialanteil EXAKT volfrac trifft
            # (Bisektion auf dem weichen Mittel) -> jeder Latentvektor erfüllt V*:
            with torch.no_grad():
                gm = g.reshape(B, -1)[:, self.maskflat]            # (B, n_mask)
                tau = _volfrac_threshold(gm, self.vf, sharp=8.0).view(B, 1, 1)
            rho = torch.sigmoid(8.0 * (g - tau))                   # near-binär @ V*
            return rho.unsqueeze(1)                                # (B, 1, ny, nx)

    return _LatticeDecoder()


def _build_initial_generator_dataset(prob: "LBracketProblem", gen: nn.Module,
                                      n_init: int, seed: int
                                      ) -> tuple[list[dict], int, list[tuple]]:
    """Initialdatensatz aus zufälligen Generator-Samples — für nicht-lernende
    Generatoren (z. B. Gitter): $z sim cal(N)(0, I)$ → decode → FEM. So sieht
    das Surrogat genau die Designklasse, die die Latent-Suche später erkundet."""
    mask = _l_bracket_mask(prob.nelx, prob.nely)
    F_vec, fixed = _l_bracket_load(prob.nelx, prob.nely)
    # 2x überstichproben und die besten n_init behalten -> filtert degenerierte
    # (unverbundene) Zufallsgitter aus, die das Surrogat-Training destabilisieren.
    g = torch.Generator().manual_seed(seed + 7)
    Z = torch.randn(2 * n_init, gen.dz, generator=g)
    with torch.no_grad():
        fields = gen.decode(Z)[..., :prob.nely, :prob.nelx][:, 0].numpy()
    cand: list[dict] = []
    fem = 0
    for k in range(fields.shape[0]):
        # Volumen-Projektion auf volfrac: dichteste Designzellen werden Material.
        rho_b = np.zeros((prob.nely, prob.nelx), dtype=float)
        vals = fields[k][mask]
        tau = np.quantile(vals, 1.0 - prob.volfrac)
        rho_b[mask] = (fields[k][mask] >= tau).astype(float)
        U, comp, _, _ = _fea(prob.nelx, prob.nely, rho_b, fixed, F_vec,
                             penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
        fem += 1
        cand.append({"rho": rho_b.astype(np.float32), "U": U.astype(np.float32),
                     "compliance": float(comp)})
    cand.sort(key=lambda d: d["compliance"])
    out = cand[:n_init]
    best = float("inf"); traj: list[tuple[int, float]] = []
    for i, d in enumerate(out, start=1):
        best = min(best, d["compliance"]); _append_trajectory(traj, (i, best))
    return out, fem, traj


# ── CNN-Surrogat für das Verschiebungsfeld ────────────────────────────

def _make_field_surrogate_cnn(ndof: int, u_scale: float,
                                fixed_dofs: Optional[np.ndarray] = None
                                ) -> nn.Module:
    """CNN-FieldSurrogate mit harter Dirichlet-Maske auf den fixierten
    DOFs. Liefert ``\\hat U \\in \\mathbb{R}^{n_\\text{dof}}`` aus dem
    Dichtefeld."""
    free_mask_np = np.ones(ndof, dtype=np.float32)
    if fixed_dofs is not None:
        free_mask_np[np.asarray(fixed_dofs, dtype=np.int64)] = 0.0

    class _FieldSurrogateCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.u_scale = float(u_scale)
            self.register_buffer("free_mask", torch.tensor(
                free_mask_np, dtype=torch.float32))
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            feat = 32 * 9 * 15
            self.head = nn.Sequential(
                nn.Linear(feat, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, ndof))

        def forward(self, rho):
            h = self.enc(rho)
            U_raw = self.u_scale * self.head(h)
            return U_raw * self.free_mask

    return _FieldSurrogateCNN()


# ── PISCO-Solver für Fall 2 ──────────────────────────────────────────

@dataclass
class PISCOConfigTopology:
    """Hyperparameter des PISCO-Solvers für freie Dichtefelder (Fall 2).

    Defaults entsprechen exakt der Paper-Konfiguration für Beispiel 2
    (Tab. 3): ``T_cycles=12, n_init=40, n_top=6, short_simp_iters=6``,
    PCA-Decoder mit ``latent_dim=16``, CNN-Surrogat mit
    ``surrogate_steps=300`` Adam-Schritten bei ``surrogate_lr=2e-3``,
    Multi-Start mit ``M=12`` Starts und ``multistart_steps=150``
    Schritten bei ``multistart_lr=0.05``, ``lam_data=1``,
    ``lam_phys=1e-2``.
    """
    # Active-Loop
    T_cycles: int = 12
    n_init: int = 40
    n_top: int = 6
    short_simp_iters: int = 6

    # Generator
    generator_type: str = "pca"  # "pca", "vae", "ae", "lattice", "wavelet", "lattice_wavelet"
    latent_dim: int = 16
    lattice_wavelet_weight: float = 0.18
    lattice_wavelet_levels: int = 5
    vae_epochs: int = 200
    vae_lr: float = 3e-3
    vae_beta_kl: float = 1e-3

    # CNN-Surrogat
    surrogate_steps: int = 300
    surrogate_lr: float = 2e-3
    lam_data: float = 1.0
    lam_phys: float = 1e-2
    grad_clip_norm: float = 1.0

    # Latent-Multi-Start
    M: int = 12
    multistart_steps: int = 150
    multistart_lr: float = 0.05
    lam_z_prior: float = 1e-3          # Gauß-Prior ||z||^2 — EINZIGER weicher Term (Paper Fall 2)
    latent_clip: float = 3.0           # harte Latent-Box |z_i| <= c (Paper Fall 2, Projektionsregel)
    lam_volume_penalty: float = 50.0   # unbenutzt: Volumen ist durch Decoder-Konstruktion erfüllt

    # Coarse-to-fine (additive Mannigfaltigkeit): bei kleiner Latentdim starten und
    # die Generator-Dimension schrittweise erhöhen (nested, warmgestartet), gegen das
    # BINARISIERTE (fertigbare) Ziel suchen. Nur mit additivem Generator (pca/wavelet).
    coarse_to_fine: bool = False
    ctf_iters: int = 60                # gradientenfreie Latent-Schritte pro Dimensionsstufe
    ctf_n_steps: int = 6               # Zahl der Dimensionsstufen (dz0 -> max_dz)
    ctf_sigma: float = 0.30            # Perturbationsstärke der Latentsuche
    ctf_max_dz: int = 200              # Atom-/Moden-Reserve für die Erweiterung (z.B. wavelet)
    simp_free_anchors: bool = False    # Anker SIMP-FREI aus Lattice-Zufallssamples (statt kurzer SIMP)
    anchor_init_amp: float = 0.20      # Init-Amplitude der Anker-Startfelder (höher = diversere Anker)

    # Ablations- und Diagnostikschalter. ``active_retraining=False`` hält
    # das nach Zyklus 1 trainierte Surrogat für alle weiteren, budgetgleichen
    # Kandidatenbewertungen fest; FEM-Prüfungen fließen dann nicht ins Modell zurück.
    active_retraining: bool = True
    diagnostic_holdout_size: int = 0   # separate FEM-Diagnostik, nicht Teil des Optimierungsbudgets
    diagnostic_top_k: int = 5

    # Konvergenz-basierte Termination (analog zu PISCOConfig / E1)
    convergence_tol_rel: float = 1e-4
    convergence_patience: Optional[int] = 3
    convergence_warmup: int = 3


class PISCOSolverTopology(BaseSolver):
    """PISCO für freie Dichtefelder (Fall 2, L-Bracket).

    Pipeline:
      1. Initialer Datensatz: ``n_init`` kurze SIMP-Läufe aus zufälligen
         Startdichten liefern strukturell vielfältige Anker-Designs.
      2. Niedrigdimensionaler Decoder (PCA/VAE/AE) wird einmalig auf den
         Initialfeldern angepasst und für den Rest des Laufs eingefroren.
      3. Aktive Schleife für ``T_cycles`` Zyklen: CNN-Surrogat auf
         aktuellem Datensatz trainieren, ``M``-fache latente
         Multi-Start-Suche gegen den eingefrorenen Kritiker, Top-``n_top``
         Kandidaten via FEM revalidieren, an Datensatz anhängen.
      3. ``best_design`` ist das Dichtefeld mit der niedrigsten
         FEM-validierten Compliance über die gesamte Pipeline.
    """

    def __init__(self, problem: LBracketProblem, config: PISCOConfigTopology, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._mask = _l_bracket_mask(problem.nelx, problem.nely)
        self._F_np, self._fixed = _l_bracket_load(problem.nelx, problem.nely)
        self._ndof = int(self._F_np.size)
        if config.generator_type not in {"pca", "vae", "ae", "lattice", "wavelet", "lattice_wavelet"}:
            raise ValueError(
                f"generator_type must be 'pca' | 'vae' | 'ae' | 'lattice' | 'wavelet' | "
                f"'lattice_wavelet', got "
                f"{config.generator_type!r}")

    # ─── core compute ───────────────────────────────────────────────

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        # Match legacy seed convention from generate_all_figures.run_framework_example2
        torch.manual_seed(seed + 1)
        prob = self.problem
        cfg  = self.config

        # Step 1+2 — Initialdatensatz + Generator (nach Vortraining eingefroren)
        if cfg.generator_type == "lattice":
            # Fixe parametrische Gitter-Familie: kein Lernen aus Ankern; der
            # Initialdatensatz entsteht aus zufälligen Gitter-Samples, damit das
            # Surrogat genau die erkundete Designklasse sieht.
            gen = _make_lattice_decoder(prob.nelx, prob.nely, dz=cfg.latent_dim,
                                        mask=self._mask, volfrac=prob.volfrac)
            dataset, fem_calls, init_traj = _build_initial_generator_dataset(
                prob, gen, cfg.n_init, seed)
            gen_info = {"final_recon": 0.0}
        elif cfg.generator_type == "lattice_wavelet":
            # Mischklasse: Lattice als Basismorphologie plus Wavelet-Rest für
            # lokale Anpassung ohne vollständigen Verlust der Strukturtreue.
            gen = _make_lattice_wavelet_decoder(
                prob.nelx, prob.nely, dz=cfg.latent_dim, mask=self._mask,
                volfrac=prob.volfrac, residual_weight=cfg.lattice_wavelet_weight,
                wavelet_levels=cfg.lattice_wavelet_levels)
            dataset, fem_calls, init_traj = _build_initial_generator_dataset(
                prob, gen, cfg.n_init, seed)
            gen_info = {"final_recon": 0.0}
        elif cfg.generator_type == "wavelet":
            # Fester Wavelet-Generator ohne Lernphase (ähnlich PCA/PCA-lite):
            # Initialfeld und Surrogat sehen damit die gleiche Familie möglicher
            # Strukturen wie die spätere latente Optimierung.
            _mxdz = max(cfg.latent_dim, cfg.ctf_max_dz) if cfg.coarse_to_fine else None
            gen = _make_wavelet_decoder(
                prob.nelx, prob.nely, dz=cfg.latent_dim,
                levels=5, slope=6.0, mask=self._mask, max_dz=_mxdz)
            dataset, fem_calls, init_traj = _build_initial_generator_dataset(
                prob, gen, cfg.n_init, seed)
            gen_info = {"final_recon": 0.0}
        else:
            if cfg.simp_free_anchors:
                # SIMP-FREI: strukturierte, volumen-exakte Anker aus Lattice-Zufalls-
                # samples (gradientenfrei) statt kurzer SIMP-Läufe. Erfüllt die drei
                # Paper-Anforderungen (Struktur, Volumen, Diversität) ohne SIMP.
                _lat = _make_lattice_decoder(
                    prob.nelx, prob.nely, dz=max(13, cfg.latent_dim),
                    mask=self._mask, volfrac=prob.volfrac)
                dataset, fem_calls, init_traj = _build_initial_generator_dataset(
                    prob, _lat, cfg.n_init, seed)
            else:
                dataset, fem_calls, init_traj = _build_initial_lbracket_dataset(
                    prob, cfg.n_init, seed,
                    short_simp_iters=cfg.short_simp_iters,
                    init_amp=cfg.anchor_init_amp)
            fields_np = np.stack([s["rho"] for s in dataset])
            if cfg.generator_type == "vae":
                fields_t = torch.tensor(fields_np[:, None, :, :],
                                          dtype=torch.float32)
                gen = _make_conv_vae(dz=cfg.latent_dim)
                gen_info = _train_vae(gen, fields_t,
                                        epochs=cfg.vae_epochs, lr=cfg.vae_lr,
                                        beta_kl=cfg.vae_beta_kl)
            elif cfg.generator_type == "ae":
                fields_t = torch.tensor(fields_np[:, None, :, :],
                                          dtype=torch.float32)
                gen = _make_conv_ae(dz=cfg.latent_dim)
                gen_info = _train_ae(gen, fields_t,
                                       epochs=cfg.vae_epochs, lr=cfg.vae_lr)
            else:  # "pca"
                gen, gen_info = _make_pca_decoder(fields_np, dz=cfg.latent_dim)
        # Volumen-Constraint durch Konstruktion: jeder Latentvektor -> exakt V*.
        # Damit braucht die Latent-Suche keinen Volumen-Strafterm (Paper Fall 2).
        gen = _VolumeProjected(gen, self._mask, prob.volfrac)
        trajectory: list[tuple[int, float]] = list(init_traj)
        for p in gen.parameters():
            p.requires_grad_(False)

        # Ein fester, vom aktiven Datensatz getrennter Diagnostiksatz. Seine
        # FEM-Kosten werden separat ausgewiesen und zählen ausdrücklich nicht
        # zum Optimierungsbudget.
        holdout, diagnostic_fem_calls = self._build_diagnostic_holdout(gen, seed)

        # Coarse-to-fine (additive Mannigfaltigkeit): surrogat-freier Pfad, der die
        # Latentdim schrittweise erhöht und gegen das binarisierte Ziel sucht.
        if (cfg.coarse_to_fine and hasattr(gen, "extend")
                and getattr(gen, "max_dz", gen.dz) > gen.dz):
            return self._coarse_to_fine_solve(gen, fem_calls, init_traj, seed, t0)

        # Step 3 — CNN surrogate
        F_torch  = torch.tensor(self._F_np, dtype=torch.float32)
        u_scale  = float(np.std([np.linalg.norm(s["U"]) for s in dataset])
                          / math.sqrt(self._ndof))
        model    = _make_field_surrogate_cnn(
            ndof=self._ndof, u_scale=u_scale, fixed_dofs=self._fixed)
        free_idx = torch.tensor(
            np.setdiff1d(np.arange(self._ndof), self._fixed),
            dtype=torch.long)
        F_norm   = float(np.linalg.norm(self._F_np))
        mask_t   = torch.tensor(self._mask, dtype=torch.float32)

        # Step 4 — active loop
        best = min(dataset, key=lambda s: s["compliance"])
        stagnation = 0
        actual_T = cfg.T_cycles
        stop_reason = "T_cycles_cap"
        cycle_metrics: list[dict] = []
        train_info: Optional[dict[str, float]] = None
        timing = {"surrogate_training_s": 0.0,
                  "latent_optimisation_s": 0.0,
                  "active_fem_s": 0.0}
        decoder_raw_min_all: list[float] = []
        decoder_raw_max_all: list[float] = []
        decoder_clipped_all: list[float] = []
        decoder_volume_all: list[float] = []
        for t in range(1, cfg.T_cycles + 1):
            _progress_cycle(t, cfg.T_cycles)
            J_before = best["compliance"]
            if cfg.active_retraining or train_info is None:
                _tic = time.perf_counter()
                train_info = self._train_surrogate(
                    model, dataset, F_torch, free_idx, F_norm)
                timing["surrogate_training_s"] += time.perf_counter() - _tic
            _tic = time.perf_counter()
            rho_cands, J_cands, decoder_diag = self._latent_optimize(
                gen, model, F_torch, mask_t)
            timing["latent_optimisation_s"] += time.perf_counter() - _tic
            decoder_raw_min_all.extend(decoder_diag["raw_min"].tolist())
            decoder_raw_max_all.extend(decoder_diag["raw_max"].tolist())
            decoder_clipped_all.extend(
                decoder_diag["clipped_fraction"].tolist())
            decoder_volume_all.extend(
                decoder_diag["projected_volume"].tolist())
            order = np.argsort(J_cands)
            top_indices = order[: cfg.n_top]
            for idx in top_indices:
                rho = rho_cands[idx]
                _tic = time.perf_counter()
                U, comp, _, Eelem = _fea(
                    prob.nelx, prob.nely, rho, self._fixed, self._F_np,
                    penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
                timing["active_fem_s"] += time.perf_counter() - _tic
                fem_calls += 1
                entry = {"rho": rho.astype(np.float32),
                         "raw_design": np.asarray(
                             decoder_diag["bounded_raw_design"][idx, 0,
                                 :prob.nely, :prob.nelx], dtype=np.float32),
                         "U":   U.astype(np.float32),
                         "compliance": float(comp),
                         "Eelem": Eelem.astype(np.float32),
                         "decoder_raw_min": float(decoder_diag["raw_min"][idx]),
                         "decoder_raw_max": float(decoder_diag["raw_max"][idx]),
                         "decoder_clipped_fraction": float(
                             decoder_diag["clipped_fraction"][idx]),
                         "decoder_projected_volume": float(
                             decoder_diag["projected_volume"][idx])}
                dataset.append(entry)
                if comp < best["compliance"]:
                    best = entry
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))

            diagnostic = self._diagnose_surrogate(
                model, holdout, F_torch, free_idx, F_norm)
            cycle_metrics.append({
                "cycle": t, "n_dataset": len(dataset),
                "fem_calls": fem_calls,
                "best_compliance": float(best["compliance"]),
                "surrogate_rel_err": train_info["surrogate_rel_err"],
                "surrogate_loss_data": train_info["final_data"],
                "surrogate_loss_phys": train_info["final_phys"],
                "active_retraining": bool(cfg.active_retraining),
                **diagnostic,
            })

            patience = cfg.convergence_patience
            if (patience is not None and patience > 0
                    and t > cfg.convergence_warmup):
                tol = cfg.convergence_tol_rel * max(J_before, 1e-12)
                improved = (J_before - best["compliance"]) > tol
                stagnation = 0 if improved else stagnation + 1
                if stagnation >= patience:
                    stop_reason = "convergence"
                    actual_T = t
                    break

        _full = int(self._mask.sum())
        print(f"[PISCO] voller Designraum = {_full} | Latentdim = {gen.dz} | "
              f"Reduktion = {_full / max(gen.dz, 1):.0f}x"
              + ("  -> KEIN Vorteil (Dimensionen gleich!)" if gen.dz >= _full else ""),
              flush=True)
        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=np.asarray(best["rho"], dtype=np.float32),
            fem_calls_total=int(fem_calls),
            trajectory=trajectory,
            seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "generator_type": cfg.generator_type,
                "latent_dim":     gen.dz,
                "generator_recon": gen_info.get("final_recon"),
                "best_U":         np.asarray(best["U"], dtype=np.float32),
                "best_Eelem":     np.asarray(best.get("Eelem"), dtype=np.float32)
                                    if best.get("Eelem") is not None else None,
                "dataset_size":   len(dataset),
                "full_design_dim": int(self._mask.sum()),
                "latent_dim_final": int(gen.dz),
                "actual_T": actual_T,
                "stop_reason": stop_reason,
                "cycle_metrics": cycle_metrics,
                "diagnostic_holdout_size": len(holdout),
                "diagnostic_fem_calls_excluded": diagnostic_fem_calls,
                "timing_breakdown_s": timing,
                "best_decoder_raw_min": best.get("decoder_raw_min"),
                "best_decoder_raw_max": best.get("decoder_raw_max"),
                "best_decoder_clipped_fraction": best.get(
                    "decoder_clipped_fraction"),
                "best_decoder_projected_volume": best.get(
                    "decoder_projected_volume"),
                "raw_design": np.asarray(
                    best.get("raw_design", best["rho"]), dtype=np.float32),
                "decoder_diagnostics_all_candidates": {
                    "n": len(decoder_raw_min_all),
                    "raw_min": (float(np.min(decoder_raw_min_all))
                                if decoder_raw_min_all else None),
                    "raw_max": (float(np.max(decoder_raw_max_all))
                                if decoder_raw_max_all else None),
                    "clipped_fraction_mean": (
                        float(np.mean(decoder_clipped_all))
                        if decoder_clipped_all else None),
                    "clipped_fraction_max": (
                        float(np.max(decoder_clipped_all))
                        if decoder_clipped_all else None),
                    "projected_volume_max_abs_error": (
                        float(np.max(np.abs(
                            np.asarray(decoder_volume_all) - prob.volfrac)))
                        if decoder_volume_all else None),
                },
            },
        )

    def _build_diagnostic_holdout(self, gen: nn.Module, seed: int
                                  ) -> tuple[list[dict], int]:
        n = max(0, int(self.config.diagnostic_holdout_size))
        if n == 0:
            return [], 0
        rng = torch.Generator().manual_seed(100_003 + int(seed))
        Z = torch.randn((n, gen.dz), generator=rng)
        Z.clamp_(-self.config.latent_clip, self.config.latent_clip)
        with torch.no_grad():
            rho = gen.decode(Z)[..., :self.problem.nely, :self.problem.nelx]
            mask_t = torch.tensor(self._mask, dtype=torch.bool)
            rho = torch.where(mask_t, rho[:, 0].clamp(0.001, 1.0),
                              torch.zeros_like(rho[:, 0]))
        holdout: list[dict] = []
        for rho_i in rho.numpy():
            U, comp, _, _ = _fea(
                self.problem.nelx, self.problem.nely, rho_i,
                self._fixed, self._F_np, penal=self.problem.penal,
                E0=1.0, Emin=1e-9, nu=0.30)
            holdout.append({"rho": rho_i.astype(np.float32),
                            "U": U.astype(np.float32),
                            "compliance": float(comp)})
        return holdout, len(holdout)

    def _diagnose_surrogate(self, model: nn.Module, holdout: list[dict],
                            F_torch: torch.Tensor, free_idx: torch.Tensor,
                            F_norm: float) -> dict[str, Any]:
        if not holdout:
            return {}
        from scipy.stats import spearmanr

        rho = torch.tensor(np.stack([s["rho"] for s in holdout])[:, None],
                           dtype=torch.float32)
        true = np.asarray([s["compliance"] for s in holdout], dtype=float)
        with torch.no_grad():
            U_hat = model(rho)
            pred = (U_hat @ F_torch).numpy()
        rel = np.abs(pred - true) / np.maximum(np.abs(true), 1e-12)
        rank_value = (float(spearmanr(true, pred).statistic)
                      if len(true) > 1 and np.ptp(true) > 0 and np.ptp(pred) > 0 else float("nan"))
        rank = rank_value if np.isfinite(rank_value) else None
        k = min(max(1, int(self.config.diagnostic_top_k)), len(true))
        top_true = set(np.argsort(true)[:k].tolist())
        top_pred = set(np.argsort(pred)[:k].tolist())

        residuals = []
        for i, sample in enumerate(holdout if self.config.lam_phys else []):
            K_t = _csr_to_torch(_assemble_K(
                self.problem.nelx, self.problem.nely, sample["rho"],
                penal=self.problem.penal, E0=1.0, Emin=1e-9, nu=0.30))
            KU = torch.sparse.mm(K_t, U_hat[i:i + 1].T).squeeze(-1)
            residuals.append(float(torch.linalg.vector_norm(
                (KU - F_torch)[free_idx]).item() / max(F_norm, 1e-12)))
        return {
            "holdout_rel_error_mean": float(rel.mean()),
            "holdout_rel_error_max": float(rel.max()),
            "holdout_spearman": rank,
            "holdout_top_k": k,
            "holdout_top_k_overlap": top_k_overlap(true, pred, k),
            "holdout_equilibrium_residual_mean": float(np.mean(residuals)) if residuals else None,
            "holdout_representations":[sample.get("representation","continuous") for sample in holdout],
            "holdout_role":"independent initial-generator samples; never used for training or stopping",
            "holdout_true_compliance": true.tolist(),
            "holdout_predicted_compliance": pred.tolist(),
        }

    def _coarse_to_fine_solve(self, gen, fem_calls, init_traj, seed, t0):
        """Additive Mannigfaltigkeit: bei kleiner Latentdim starten, ``gen.extend``
        + warmgestartet (bestes z mit Nullen aufgefüllt -> identisches Design)
        weitersuchen, gegen das BINARISIERTE (fertigbare) Ziel. Volumen hart
        (Quantil-Binarisierung == exakt V*). Gibt voller- vs. latenter Dim aus —
        sind sie gleich, gibt es keine Reduktion und damit keinen Vorteil."""
        prob = self.problem; cfg = self.config
        rng = np.random.default_rng(seed)
        m = np.asarray(self._mask, bool)
        full_dim = int(m.sum())
        g = getattr(gen, "inner", gen)             # rohes Decode: Volumen via Binarisierung hart,
                                                   # ohne die volumen-projizierende Hülle

        n_design = int(m.sum())
        nsel = int(round(prob.volfrac * n_design))         # EXAKTE Materialzahl

        def bin_c(z):
            with torch.no_grad():
                rho = g.decode(torch.tensor(np.asarray(z, dtype=np.float32)
                                            ).reshape(1, -1))[0, 0].numpy()
            vals = rho[m]
            b = np.zeros_like(rho)
            if 0 < nsel < vals.size:                       # Top-k: exakt V* Pixel,
                top = np.argpartition(vals, vals.size - nsel)[vals.size - nsel:]
                bm = np.zeros(vals.size, np.float32); bm[top] = 1.0
                b[m] = bm                                  # robust auch bei konstantem Feld
            elif nsel >= vals.size:
                b[m] = 1.0
            _, c, _, _ = _fea(prob.nelx, prob.nely, b, self._fixed, self._F_np,
                              penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
            return float(c), b

        dz0 = int(g.dz); dzmax = int(getattr(g, "max_dz", dz0))
        schedule = sorted({int(round(x)) for x in
                           np.linspace(dz0, dzmax, max(1, cfg.ctf_n_steps))})
        best_z = np.zeros(dz0, dtype=np.float32)
        best_c, best_b = bin_c(best_z); best_dz = dz0; fem_calls += 1
        traj = list(init_traj); _append_trajectory(traj, (fem_calls, best_c))
        for dz in schedule:
            g.extend(dz)
            if best_z.size < g.dz:
                best_z = np.concatenate(
                    [best_z, np.zeros(g.dz - best_z.size, np.float32)])
            for _ in range(cfg.ctf_iters):
                z = best_z + cfg.ctf_sigma * rng.standard_normal(g.dz)
                c, b = bin_c(z); fem_calls += 1
                if c < best_c:
                    best_c, best_z, best_b, best_dz = c, z.astype(np.float32), b, int(g.dz)
                _append_trajectory(traj, (fem_calls, best_c))

        red = full_dim / max(best_dz, 1)
        no_adv = best_dz >= full_dim
        print(f"[coarse-to-fine] voller Designraum = {full_dim} | "
              f"Latentdim (final) = {best_dz} | Reduktion = {red:.0f}x"
              + ("  -> KEIN Vorteil (Dimensionen gleich!)" if no_adv else ""), flush=True)
        return SolverResult(
            best_compliance=float(best_c),
            best_design=np.asarray(best_b, dtype=np.float32),
            fem_calls_total=int(fem_calls), trajectory=traj, seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={"architecture": "coarse_to_fine_additive",
                   "generator_type": cfg.generator_type,
                   "full_design_dim": full_dim,
                   "latent_dim_final": int(best_dz),
                   "dim_reduction": float(red),
                   "no_advantage": bool(no_adv)})

    # ─── surrogate training ────────────────────────────────────────

    def _train_surrogate(self, model: nn.Module, dataset: list[dict],
                          F_torch: torch.Tensor, free_idx: torch.Tensor,
                          F_norm: float) -> dict:
        cfg, prob = self.config, self.problem
        with audit_scope("training_preparation"):
            densities = torch.tensor(np.stack([item["rho"] for item in dataset])[:,None], dtype=torch.float32)
            targets = torch.tensor([item["compliance"] for item in dataset], dtype=torch.float32)
            matrices = ([_training_stiffness(prob.nelx, prob.nely, item["rho"], prob.penal)
                         for item in dataset] if cfg.lam_phys else [])
        return _fit_field(model, densities, targets, F_torch, matrices, free_idx, F_norm,
                          steps=cfg.surrogate_steps, lr=cfg.surrogate_lr,
                          lam_data=cfg.lam_data, lam_phys=cfg.lam_phys, clip=cfg.grad_clip_norm)

    # ─── latent multi-start optimisation ───────────────────────────

    def _latent_optimize(self, gen: nn.Module, model: nn.Module,
                          F_torch: torch.Tensor, mask_t: torch.Tensor
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        cfg  = self.config
        prob = self.problem
        Z = torch.randn(cfg.M, gen.dz, requires_grad=True)
        def objective(Z):
            rho = gen.decode(Z)[..., :prob.nely, :prob.nelx][:, 0] * mask_t
            return model(rho.unsqueeze(1)) @ F_torch + cfg.lam_z_prior * Z.pow(2).sum(dim=-1)
        Z, _ = _packet_adam(Z, objective, steps=cfg.multistart_steps,
            lr=cfg.multistart_lr, project=lambda value: value.clamp(-cfg.latent_clip, cfg.latent_clip))
        with torch.no_grad():
            if hasattr(gen, "decode_with_diagnostics"):
                rho_final, diagnostics = gen.decode_with_diagnostics(Z)
            else:
                rho_final = gen.decode(Z)
                flat = rho_final.reshape(rho_final.shape[0], -1)
                diagnostics = {
                    "raw_min": flat.min(dim=1).values,
                    "raw_max": flat.max(dim=1).values,
                    "clipped_fraction": torch.zeros(rho_final.shape[0]),
                    "projected_volume": flat.mean(dim=1),
                }
            rho_final = rho_final[..., : prob.nely, : prob.nelx]
            rho_final = torch.where(
                mask_t.bool(), rho_final[:, 0].clamp(0.001, 1.0),
                torch.zeros_like(rho_final[:, 0]))
            J_pred    = (model(rho_final.unsqueeze(1)) @ F_torch).numpy()
        diagnostics_np = {k: v.detach().cpu().numpy()
                          for k, v in diagnostics.items()}
        return rho_final.numpy(), J_pred, diagnostics_np


# =====================================================================
# 9. SIMP — klassisches Topologie-Optimierungsverfahren
# =====================================================================

@dataclass
class SIMPConfig:
    """Hyperparameter des SIMP-Solvers.

    Defaults entsprechen der Paper-Konfiguration für die L-Bracket-Domäne
    (Abschn. 5.3.1): ``volfrac=0.45, penal=3.0, rmin=2.4, maxiter=80``.
    Single-Start ``n_starts=1``, Multi-Start ``n_starts=4``.
    """
    n_starts: int = 1
    volfrac: float = 0.45
    penal: float = 3.0
    rmin: float = 2.4
    maxiter: int = 80
    tol: float = 1e-3
    perturbation_sigma: float = 0.25


class SIMPSolver(BaseSolver):
    """SIMP-Solver für freie Topologie-Optimierung auf 2D-Domänen.

    Aktuell auf die L-Bracket-Domäne zugeschnitten (nutzt
    ``_l_bracket_mask`` / ``_l_bracket_load``); domänenagnostische
    Erweiterung über injizierbare Maske + Last ist trivial.
    """

    def __init__(self, problem: LBracketProblem, config: SIMPConfig, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._mask: np.ndarray = _l_bracket_mask(problem.nelx, problem.nely)
        F, fixed = _l_bracket_load(problem.nelx, problem.nely)
        self._F: np.ndarray = F
        self._fixed: np.ndarray = fixed

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        self._seed_everything(seed)

        if self.config.n_starts <= 1:
            res = self._run_single(x_init=None)
            best_compliance = float(res["compliance"])
            best_design = res["xPhys"]
            fem_calls = int(res["fem_calls"])
            extra: dict[str, Any] = {
                "objective_history": list(res["objective_history"]),
                "raw_design": np.asarray(res["x"], dtype=np.float32),
                "n_starts": 1,
                "starts": [{"start_idx": 0, "compliance": best_compliance,
                             "fem_calls": fem_calls}],
            }
        else:
            best_compliance, best_design, fem_calls, extra = \
                self._run_multistart(seed=seed)

        # Trajektorie aus der Compliance-Historie (ein Eintrag pro
        # OC-Schritt = pro FEM-Aufruf); kompatibel zu PISCO-Trajektorien
        # für gemeinsame Plots.
        trajectory: list[tuple[int, float]] = []
        running_best = float("inf")
        for step, c in enumerate(extra.get("objective_history", []), start=1):
            running_best = min(running_best, float(c))
            _append_trajectory(trajectory, (step, running_best))

        return SolverResult(
            best_compliance=float(best_compliance),
            best_design=best_design,
            fem_calls_total=int(fem_calls),
            trajectory=trajectory,
            seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra=extra,
        )

    def _run_single(self, *, x_init) -> dict:
        """Ein einzelner SIMP-Lauf (klassisch oder mit perturbiertem Start)."""
        return _simp(
            self.problem.nelx, self.problem.nely, self._mask, self._F,
            self._fixed,
            volfrac=self.config.volfrac, penal=self.config.penal,
            rmin=self.config.rmin,
            maxiter=self.config.maxiter, tol=self.config.tol,
            verbose=False, x_init=x_init)

    def _run_multistart(self, *, seed: int) -> tuple[float, np.ndarray, int, dict]:
        """``n_starts`` unabhängige Läufe; bestes Ergebnis + Statistik."""
        best_run, summary = _simp_multistart(
            self.problem.nelx, self.problem.nely, self._mask, self._F,
            self._fixed,
            n_starts=self.config.n_starts, seed=seed,
            volfrac=self.config.volfrac, penal=self.config.penal,
            rmin=self.config.rmin,
            maxiter=self.config.maxiter, tol=self.config.tol,
            verbose=False)
        extra: dict[str, Any] = {
            "n_starts": int(self.config.n_starts),
            "starts": list(summary["runs"]),
            "compliance_mean": float(summary["compliance_mean"]),
            "compliance_std": float(summary["compliance_std"]),
            "objective_history": list(best_run["objective_history"]),
            "raw_design": np.asarray(best_run["x"], dtype=np.float32),
        }
        return (float(best_run["compliance"]), best_run["xPhys"],
                int(summary["fem_calls_total"]), extra)


# =====================================================================
# 10. DE — Differential Evolution als gradientenfreie Black-Box-Baseline
# =====================================================================

@dataclass
class DEConfig:
    """Hyperparameter für die DE-Baseline. Defaults aus Tab. 2:
    ``maxiter=30, popsize=10, polish=True``.
    """
    maxiter: int = 30
    popsize: int = 10
    polish: bool = True
    tol: float = 0.01
    atol: float = 0.0
    feasibility_penalty_geom: float = 100.0
    feasibility_penalty_area: float = 120.0


class DESolver(BaseSolver):
    """Differential-Evolution-Baseline für parametrische Ellipsendesigns.

    Aktuell auf das Ellipse-Cantilever-Problem (Beispiel 1) zugeschnitten;
    Übertragung trivial: ``_make_objective`` gegen die problemspezifische
    Penalty-/Feasibility-Funktion tauschen.
    """

    def __init__(self, problem: EllipseProblem, config: DEConfig, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._box = EllipseBox()
        if problem.fixed_area:
            self._bounds = [self._box.xc, self._box.yc, self._box.rx,
                            self._box.th]
        else:
            self._bounds = [self._box.xc, self._box.yc, self._box.rx,
                            self._box.ry, self._box.th]
        self._solid_compliance: float = _solid_case(problem)["compliance"]

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        self._seed_everything(seed)

        # Tracker per Closure — derselbe Zähler für DE-Iterationen *und*
        # den optionalen L-BFGS-Polish, obwohl scipy beide Phasen intern
        # entkoppelt durchläuft.
        fem_calls = [0]
        best_running = [float("inf")]
        best_design: list[Optional[np.ndarray]] = [None]
        trajectory: list[tuple[int, float]] = []

        objective = self._make_objective(fem_calls, best_running,
                                           best_design, trajectory)

        res = opt.differential_evolution(
            objective, bounds=self._bounds,
            maxiter=self.config.maxiter, popsize=self.config.popsize,
            polish=self.config.polish, seed=seed, disp=False,
            tol=self.config.tol, atol=self.config.atol)

        # Benchmark-Wert ist die niedrigste *zulässige* Compliance, die
        # während der gesamten Suche (inkl.\ Polish-Auswertungen)
        # registriert wurde — sie liegt häufig leicht unter dem
        # Polish-Endpunkt ``res.x``, da der Polish den penalisierten
        # Zielwert minimiert und dabei kleine Compliance-Erhöhungen in
        # Kauf nehmen kann.
        if best_design[0] is None:
            if self.problem.fixed_area:
                rx = res.x[2]
                ry = self.problem.ry_from_rx(rx)
                x5 = np.array([res.x[0], res.x[1], rx, ry, res.x[3]])
            else:
                x5 = np.asarray(res.x, dtype=float)
            best_design[0] = x5
            best_running[0] = float(_solve_ellipse(self.problem, x5)["compliance"])

        if self.problem.fixed_area:
            rx_p = res.x[2]
            ry_p = self.problem.ry_from_rx(rx_p)
            polish_x5 = np.array([res.x[0], res.x[1], rx_p, ry_p, res.x[3]])
        else:
            polish_x5 = np.asarray(res.x, dtype=float)

        return SolverResult(
            best_compliance=float(best_running[0]),
            best_design=np.asarray(best_design[0], dtype=float),
            fem_calls_total=int(fem_calls[0]),
            trajectory=list(trajectory),
            seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "solid_compliance": self._solid_compliance,
                "de_nit": int(res.nit),
                "de_message": str(res.message),
                "de_success": bool(res.success),
                "polish_compliance": float(_solve_ellipse(self.problem, polish_x5)["compliance"]),
            },
        )

    def _make_objective(self, fem_calls: list[int],
                          best_running: list[float],
                          best_design: list[Optional[np.ndarray]],
                          trajectory: list[tuple[int, float]]):
        """Erzeuge die DE-Zielfunktion mit Closure-State für Tracker."""
        problem = self.problem
        target_area = problem.target_hole_area
        Lx, Ly = problem.Lx_mm, problem.Ly_mm
        gm = 2.0 * (Lx / problem.nelx)
        c_solid = self._solid_compliance
        pen_geom = self.config.feasibility_penalty_geom
        pen_area_w = self.config.feasibility_penalty_area

        if problem.fixed_area:
            def objective(p: np.ndarray) -> float:
                rx = p[2]
                ry = problem.ry_from_rx(rx)
                p5 = np.array([p[0], p[1], rx, ry, p[3]])
                sol = _solve_ellipse(problem, p5)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                feasible_geom = _ellipse_feasible(
                    p5, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                if feasible_geom and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p5.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_geom
                return c_norm
        else:
            def objective(p: np.ndarray) -> float:
                sol = _solve_ellipse(problem, p)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                rx, ry = p[2], p[3]
                pen_area = pen_area_w * (
                    (math.pi * rx * ry - target_area) / target_area) ** 2
                feasible_geom = _ellipse_feasible(
                    p, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                feasible_area = (
                    abs(math.pi * rx * ry - target_area) / target_area) <= 0.05
                if feasible_geom and feasible_area and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_area + pen_geom
                return c_norm + pen_area

        return objective


# =====================================================================
# 11. BO — Bayesian Optimisation als Surrogat-basierte Black-Box-Baseline
# =====================================================================
#
# Reviewer-Antwort: DE ist auf glatten 5D-Problemen eine schwache Latte;
# BO mit Matern-5/2-Kernel und Expected-Improvement-Akquisition ist die
# kanonische Surrogat-Vergleichsbasis bei teuren Black-Box-Funktionen.
# Defaults (n_init=20, n_iters=136) summieren auf 156 FEM-Aufrufe —
# identisches Budget wie PISCO.

@dataclass
class BOConfig:
    """Hyperparameter für die BO-Baseline (sklearn-GP + Matern-5/2 + EI).

    ``n_iters`` ist der \\emph{Sicherheits-Cap} der EI-Iterationen
    (Default 136, was zusammen mit ``n_init=20`` einem matched-PISCO-
    Budget entspricht). Convergence-basiertes Early-Stopping ist per
    Default an: nach ``convergence_warmup`` aufgewärmten Iterationen
    bricht der Loop ab, sobald die laufende beste Compliance über
    ``convergence_patience`` aufeinanderfolgende Iterationen keine
    relative Verbesserung größer als ``convergence_tol_rel`` zeigt.
    Auf ``convergence_patience=None`` deaktiviert."""
    n_init: int = 20
    n_iters: int = 136
    matern_nu: float = 2.5            # Matern-5/2
    xi: float = 0.01                  # EI-Exploration
    n_restarts_acq: int = 8           # Akquisition-Multi-Start
    gp_n_restarts: int = 2            # GP-Hyperopt-Restarts
    length_scale_init: float = 0.3
    length_scale_lb: float = 1e-2
    length_scale_ub: float = 1e4
    convergence_tol_rel: float = 1e-4
    convergence_patience: Optional[int] = 20
    convergence_warmup: int = 30
    feasibility_penalty_geom: float = 100.0
    feasibility_penalty_area: float = 120.0


class BOSolver(BaseSolver):
    """Bayesian-Optimisation-Baseline für parametrische Designs.

    GP mit Matern-5/2-Kernel + WhiteKernel + Expected-Improvement-
    Akquisition. Bei ``fixed_area=True`` wird auf dem 4D-Raum
    ``(xc, yc, rx, θ)`` optimiert.

    Erfordert ``scikit-learn`` (``GaussianProcessRegressor``).
    """

    def __init__(self, problem: EllipseProblem, config: BOConfig, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._box = EllipseBox()
        if problem.fixed_area:
            self._bounds = np.array([self._box.xc, self._box.yc, self._box.rx,
                                      self._box.th], dtype=float)
        else:
            self._bounds = np.array([self._box.xc, self._box.yc, self._box.rx,
                                      self._box.ry, self._box.th], dtype=float)
        self._solid_compliance: float = _solid_case(problem)["compliance"]

    def _compute(self, *, seed: int) -> SolverResult:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                Matern, ConstantKernel, WhiteKernel)
            from sklearn.exceptions import ConvergenceWarning
        except ImportError as e:
            raise ImportError(
                "BOSolver requires scikit-learn. "
                "Install via `pip install scikit-learn`.") from e
        from scipy.stats import norm as _gauss
        import warnings as _warnings

        t0 = time.perf_counter()
        self._seed_everything(seed)
        rng = np.random.default_rng(seed)

        fem_calls = [0]
        best_running = [float("inf")]
        best_design: list[Optional[np.ndarray]] = [None]
        trajectory: list[tuple[int, float]] = []
        objective = self._make_objective(fem_calls, best_running,
                                           best_design, trajectory)

        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]
        d = len(lb)

        def denorm(xn: np.ndarray) -> np.ndarray:
            return lb + xn * (ub - lb)

        X_n = rng.uniform(size=(self.config.n_init, d))
        y = np.array([objective(denorm(x)) for x in X_n], dtype=float)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.full(d, self.config.length_scale_init),
                      length_scale_bounds=(self.config.length_scale_lb,
                                              self.config.length_scale_ub),
                      nu=self.config.matern_nu)
            + WhiteKernel(noise_level=1e-6,
                            noise_level_bounds=(1e-10, 1e-2))
        )

        def neg_ei(xn: np.ndarray, gp, y_best: float) -> float:
            mu, sigma = gp.predict(np.atleast_2d(xn), return_std=True)
            sigma_val = max(float(sigma[0]), 1e-12)
            imp = y_best - float(mu[0]) - self.config.xi
            z = imp / sigma_val
            ei = imp * _gauss.cdf(z) + sigma_val * _gauss.pdf(z)
            return -float(ei)

        prev_best = float("inf")
        stagnation = 0
        actual_iters = 0
        stop_reason = "n_iters_cap"
        for iter_idx in range(self.config.n_iters):
            actual_iters = iter_idx + 1
            gp = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                n_restarts_optimizer=self.config.gp_n_restarts,
                random_state=int(rng.integers(2**31 - 1)))
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", category=ConvergenceWarning)
                with audit_scope("surrogate_training"):
                    gp.fit(X_n, y)
            y_best = float(np.min(y))

            best_xn, best_val = None, np.inf
            for _ in range(self.config.n_restarts_acq):
                x0 = rng.uniform(size=d)
                res = opt.minimize(
                    neg_ei, x0, args=(gp, y_best), method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * d)
                if res.fun < best_val:
                    best_val = res.fun
                    best_xn = res.x
            if best_xn is None:
                best_xn = rng.uniform(size=d)

            y_next = objective(denorm(best_xn))
            X_n = np.vstack([X_n, best_xn[None, :]])
            y = np.append(y, y_next)

            # Konvergenz-Check (nach Warmup): relative Verbesserung
            # der laufenden besten \\emph{feasiblen} Compliance.
            patience = self.config.convergence_patience
            if (patience is not None and patience > 0
                    and actual_iters > self.config.convergence_warmup):
                cur_best = float(best_running[0])
                if np.isfinite(prev_best):
                    tol = self.config.convergence_tol_rel * max(prev_best, 1e-12)
                    improved = (prev_best - cur_best) > tol
                    stagnation = 0 if improved else stagnation + 1
                    if stagnation >= patience:
                        stop_reason = "convergence"
                        prev_best = cur_best
                        break
                prev_best = cur_best

        if best_design[0] is None:
            raw = denorm(X_n[int(np.argmin(y))])
            if self.problem.fixed_area:
                rx = raw[2]
                ry = self.problem.ry_from_rx(rx)
                best_design[0] = np.array([raw[0], raw[1], rx, ry, raw[3]])
            else:
                best_design[0] = raw.copy()
            best_running[0] = float(
                _solve_ellipse(self.problem, best_design[0])["compliance"])

        return SolverResult(
            best_compliance=float(best_running[0]),
            best_design=np.asarray(best_design[0], dtype=float),
            fem_calls_total=int(fem_calls[0]),
            trajectory=list(trajectory),
            seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "solid_compliance": self._solid_compliance,
                "bo_kernel": "Matern-5/2 + WhiteKernel",
                "bo_n_init": int(self.config.n_init),
                "bo_n_iters": int(self.config.n_iters),
                "actual_iters": int(actual_iters),
                "stop_reason": stop_reason,
            },
        )

    def _make_objective(self, fem_calls, best_running, best_design, trajectory):
        """Penalisierte Box-Zielfunktion — identische Form wie DE/CMA-ES."""
        problem = self.problem
        target_area = problem.target_hole_area
        Lx, Ly = problem.Lx_mm, problem.Ly_mm
        gm = 2.0 * (Lx / problem.nelx)
        c_solid = self._solid_compliance
        pen_geom = self.config.feasibility_penalty_geom
        pen_area_w = self.config.feasibility_penalty_area

        if problem.fixed_area:
            def objective(p: np.ndarray) -> float:
                rx = p[2]
                ry = problem.ry_from_rx(rx)
                p5 = np.array([p[0], p[1], rx, ry, p[3]])
                sol = _solve_ellipse(problem, p5)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                feasible_geom = _ellipse_feasible(
                    p5, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                if feasible_geom and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p5.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_geom
                return c_norm
        else:
            def objective(p: np.ndarray) -> float:
                sol = _solve_ellipse(problem, p)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                rx, ry = p[2], p[3]
                pen_area = pen_area_w * (
                    (math.pi * rx * ry - target_area) / target_area) ** 2
                feasible_geom = _ellipse_feasible(
                    p, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                feasible_area = (
                    abs(math.pi * rx * ry - target_area) / target_area) <= 0.05
                if feasible_geom and feasible_area and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_area + pen_geom
                return c_norm + pen_area

        return objective


# =====================================================================
# 12. CMA-ES — Evolution Strategy als gradientenfreie Baseline
# =====================================================================
#
# Reviewer-Antwort: zweite gradientenfreie Black-Box-Baseline neben DE.
# CMA-ES~\\cite{hansen2016cmatutorial} ist auf glatten,
# nicht-separablen Problemen typischerweise dem Standard-DE überlegen
# und damit eine härtere Latte. Pure-numpy-Implementation (kein
# externes ``cma``-Paket nötig); folgt Hansens Tutorial-Pseudocode mit
# positiven Recombination-Weights (kein Active-CMA-Variant).

@dataclass
class CMAESConfig:
    """Hyperparameter für CMA-ES.

    ``n_generations`` ist der \\emph{Sicherheits-Cap} der CMA-ES-
    Generationen (Default 19, also 152 FEM bei popsize=8 im festen
    4D-Flächenraum). Der separate Fixed-Budget-Lauf verwendet 20 volle
    Generationen und liest die Trajektorie nach Aufruf 156 aus.
    Convergence-basiertes Early-
    Stopping ist per Default an: nach ``convergence_warmup``
    aufgewärmten Generationen bricht der Loop ab, sobald die laufende
    beste Compliance über ``convergence_patience`` aufeinanderfolgende
    Generationen keine relative Verbesserung größer als
    ``convergence_tol_rel`` zeigt."""
    n_generations: int = 19
    popsize: Optional[int] = None     # None → CMA-Default 4 + 3 ln(n)
    sigma_init: float = 0.25          # in normierten [0,1]^d-Koordinaten
    convergence_tol_rel: float = 1e-4
    convergence_patience: Optional[int] = 5
    convergence_warmup: int = 3
    feasibility_penalty_geom: float = 100.0
    feasibility_penalty_area: float = 120.0


class CMAESSolver(BaseSolver):
    """CMA-ES-Baseline für parametrische Designs.

    Pure-Numpy-Implementation nach \\cite{hansen2016cmatutorial}.
    Bei ``fixed_area=True`` wird auf dem 4D-Raum ``(xc, yc, rx, θ)``
    optimiert."""

    def __init__(self, problem: EllipseProblem, config: CMAESConfig, *,
                 cache_dir: Optional[Path] = None,
                 cache_enabled: bool = True):
        super().__init__(problem, config,
                          cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._box = EllipseBox()
        if problem.fixed_area:
            self._bounds = np.array([self._box.xc, self._box.yc, self._box.rx,
                                      self._box.th], dtype=float)
        else:
            self._bounds = np.array([self._box.xc, self._box.yc, self._box.rx,
                                      self._box.ry, self._box.th], dtype=float)
        self._solid_compliance: float = _solid_case(problem)["compliance"]

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        self._seed_everything(seed)
        rng = np.random.default_rng(seed)

        fem_calls = [0]
        best_running = [float("inf")]
        best_design: list[Optional[np.ndarray]] = [None]
        trajectory: list[tuple[int, float]] = []
        objective = self._make_objective(fem_calls, best_running,
                                           best_design, trajectory)

        lb = self._bounds[:, 0]
        ub = self._bounds[:, 1]

        def denorm(xn: np.ndarray) -> np.ndarray:
            return lb + xn * (ub - lb)

        n = len(lb)
        lam = self.config.popsize or (4 + int(3 * math.log(n)))
        mu = lam // 2

        # Positive Recombination-Weights (Hansen 2016, Eq. 49)
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mueff = 1.0 / (weights ** 2).sum()

        # Adaptations-Konstanten
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = (1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1)
                    + c_sigma)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1,
                    2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        # Strategievariablen — Start auf Box-Mitte
        m = np.full(n, 0.5)
        sigma = float(self.config.sigma_init)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        C_invsqrt = np.eye(n)
        eigeneval = 0

        prev_best = float("inf")
        stagnation = 0
        actual_generations = 0
        stop_reason = "n_generations_cap"
        for g in range(self.config.n_generations):
            actual_generations = g + 1
            # Sample λ Nachkommen
            X = np.zeros((lam, n))
            f = np.zeros(lam)
            for k in range(lam):
                z = rng.standard_normal(n)
                xk = m + sigma * (B @ (D * z))
                xk = np.clip(xk, 0.0, 1.0)
                X[k] = xk
                f[k] = objective(denorm(xk))

            # Sortieren nach Fitness
            order = np.argsort(f)
            X = X[order]

            # Recombination
            m_old = m.copy()
            m = (weights[:, None] * X[:mu]).sum(axis=0)

            # Evolution paths
            p_sigma = ((1 - c_sigma) * p_sigma
                        + math.sqrt(c_sigma * (2 - c_sigma) * mueff)
                          * (C_invsqrt @ (m - m_old)) / sigma)

            hsig_lhs = (np.linalg.norm(p_sigma)
                         / math.sqrt(1 - (1 - c_sigma) ** (2 * (g + 1))))
            hsig = float(hsig_lhs / chi_n < 1.4 + 2 / (n + 1))
            p_c = ((1 - c_c) * p_c
                    + hsig * math.sqrt(c_c * (2 - c_c) * mueff)
                      * (m - m_old) / sigma)

            # Kovarianz-Update (rank-1 + rank-mu)
            artmp = (X[:mu] - m_old) / sigma
            C = ((1 - c_1 - c_mu) * C
                  + c_1 * (np.outer(p_c, p_c)
                             + (1 - hsig) * c_c * (2 - c_c) * C)
                  + c_mu * (artmp.T @ np.diag(weights) @ artmp))

            # Step-size update
            sigma = sigma * math.exp(
                (c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Periodische Eigen-Zerlegung
            if (g + 1) - eigeneval > lam / (c_1 + c_mu) / n / 10:
                eigeneval = g + 1
                C = 0.5 * (C + C.T)
                eigvals, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-20))
                C_invsqrt = B @ np.diag(1.0 / D) @ B.T

            # Konvergenz-Check (nach Warmup): relative Verbesserung
            # der laufenden besten feasiblen Compliance pro Generation.
            patience = self.config.convergence_patience
            if (patience is not None and patience > 0
                    and actual_generations > self.config.convergence_warmup):
                cur_best = float(best_running[0])
                if np.isfinite(prev_best):
                    tol = self.config.convergence_tol_rel * max(prev_best, 1e-12)
                    improved = (prev_best - cur_best) > tol
                    stagnation = 0 if improved else stagnation + 1
                    if stagnation >= patience:
                        stop_reason = "convergence"
                        prev_best = cur_best
                        break
                prev_best = cur_best

        if best_design[0] is None:
            raw = denorm(np.clip(m, 0.0, 1.0))
            if self.problem.fixed_area:
                rx = raw[2]
                ry = self.problem.ry_from_rx(rx)
                best_design[0] = np.array([raw[0], raw[1], rx, ry, raw[3]])
            else:
                best_design[0] = raw.copy()
            best_running[0] = float(
                _solve_ellipse(self.problem, best_design[0])["compliance"])

        return SolverResult(
            best_compliance=float(best_running[0]),
            best_design=np.asarray(best_design[0], dtype=float),
            fem_calls_total=int(fem_calls[0]),
            trajectory=list(trajectory),
            seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={
                "solid_compliance": self._solid_compliance,
                "cma_n_generations": int(self.config.n_generations),
                "cma_popsize": int(lam),
                "cma_sigma_final": float(sigma),
                "actual_generations": int(actual_generations),
                "stop_reason": stop_reason,
            },
        )

    def _make_objective(self, fem_calls, best_running, best_design, trajectory):
        """Penalisierte Box-Zielfunktion — identische Form wie DE/BO."""
        problem = self.problem
        target_area = problem.target_hole_area
        Lx, Ly = problem.Lx_mm, problem.Ly_mm
        gm = 2.0 * (Lx / problem.nelx)
        c_solid = self._solid_compliance
        pen_geom = self.config.feasibility_penalty_geom
        pen_area_w = self.config.feasibility_penalty_area

        if problem.fixed_area:
            def objective(p: np.ndarray) -> float:
                rx = p[2]
                ry = problem.ry_from_rx(rx)
                p5 = np.array([p[0], p[1], rx, ry, p[3]])
                sol = _solve_ellipse(problem, p5)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                feasible_geom = _ellipse_feasible(
                    p5, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                if feasible_geom and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p5.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_geom
                return c_norm
        else:
            def objective(p: np.ndarray) -> float:
                sol = _solve_ellipse(problem, p)
                c_abs = float(sol["compliance"])
                c_norm = c_abs / c_solid
                fem_calls[0] += 1
                rx, ry = p[2], p[3]
                pen_area = pen_area_w * (
                    (math.pi * rx * ry - target_area) / target_area) ** 2
                feasible_geom = _ellipse_feasible(
                    p, Lx=Lx, Ly=Ly,
                    load_y1=problem.load_y1, load_y2=problem.load_y2,
                    margin=gm, load_clearance=gm)
                feasible_area = (
                    abs(math.pi * rx * ry - target_area) / target_area) <= 0.05
                if feasible_geom and feasible_area and c_abs < best_running[0]:
                    best_running[0] = c_abs
                    best_design[0] = p.copy()
                _append_trajectory(trajectory, (fem_calls[0], best_running[0]))
                if not feasible_geom:
                    return c_norm + pen_area + pen_geom
                return c_norm + pen_area

        return objective


# =====================================================================
# 13. PUBLIC API — was importierbar ist
# =====================================================================

__all__ = [
    # Cache
    "DEFAULT_CACHE_DIR",
    # Probleme
    "EllipseProblem", "EllipseBox", "LBracketProblem",
    # Gemeinsamer Contract
    "BaseSolver", "SolverResult",
    # PISCO
    "PISCOSolver", "PISCOConfig",
    "PISCOSolverPINN", "PISCOConfigPINN",
    "PISCOSolverTopology", "PISCOConfigTopology",
    "Generator", "AnalyticEllipseGenerator",
    # SIMP
    "SIMPSolver", "SIMPConfig",
    # Gradientenfreie Baselines (Beispiel 1)
    "DESolver", "DEConfig",
    "BOSolver", "BOConfig",
    "CMAESSolver", "CMAESConfig",
]


# =====================================================================
# 14. MAJOR-REVISION PROTOCOL — canonical public implementation
# =====================================================================

import os
import torch.nn.functional as functional


REVISION_PROTOCOL = "e2_exact_filter_and_projection_v4"
PUBLICATION_PROTOCOL = "eaai_revision2_2026_11_ccw_independent_streams_v4"
PUBLICATION_SEEDS = tuple(range(7, 16))
FIN_SEEDS = tuple(range(9))


@dataclass
class RevisionPISCOConfigTopology(PISCOConfigTopology):
    """Configuration of the dynamic-rank topology solver used in the paper."""

    adaptive_pca_enabled: bool = False
    adaptive_pca_max_dim: int = 40
    adaptive_pca_n_stages: int = 6
    adaptive_pca_expand_cycles: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    adaptive_pca_warm_start_fraction: float = 0.5
    adaptive_pca_convergence_after_final_expansion: bool = True
    latent_prior_reference_dim: int = 16
    local_candidate_pool_size: int = 0
    local_candidate_sigma: float = 0.3


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_path(solver: Any, seed: int) -> Path:
    fingerprint = _payload_hash(solver._cache_payload(int(seed)))
    return (Path(solver.cache_dir).resolve().parent / "checkpoints"
            / type(solver).__name__
            / f"seed_{int(seed):04d}_{fingerprint}.pkl")


def load_checkpoint(solver: Any, seed: int) -> dict[str, Any] | None:
    path = _checkpoint_path(solver, seed)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("protocol") != PUBLICATION_PROTOCOL:
            return None
        state = bundle.get("state")
        audit = _ACTIVE_AUDIT.get()
        if audit is not None and state:
            audit.restore(state.get("r2_audit"))
        return state
    except (OSError, EOFError, pickle.UnpicklingError, KeyError):
        return None


def save_checkpoint(solver: Any, seed: int, state: dict[str, Any]) -> Path:
    path = _checkpoint_path(solver, seed)
    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        state = {**state, "r2_audit": audit.snapshot()}
    _atomic_pickle_dump({
        "protocol": PUBLICATION_PROTOCOL,
        "saved_at_unix": time.time(),
        "state": state,
    }, path)
    return path


class ResultStore:
    """Atomic, per-seed result storage shared by every public benchmark."""

    def __init__(self, root: Path | str, training_policy=None):
        self.root = Path(root)
        self.training_policy = training_policy or TrainingPolicy()
        self.progress_interval = 30.0
        self._batch = None
        self._fresh_durations = {}

    def _path(self, group: str, method: str, solver: BaseSolver,
              seed: int) -> Path:
        fingerprint = _payload_hash(solver._cache_payload(int(seed)))
        return (self.root / group / method
                / f"seed_{int(seed):04d}_{fingerprint}.pkl")

    def solve(self, group, method, solver, seed):
        durations = self._fresh_durations.setdefault((group,method), [])
        with LiveProgress(self.root.parent, group, method, seed, batch=self._batch,
                          durations=durations, interval=self.progress_interval) as live:
            result = self._solve(group,method,solver,seed)
            cached = bool(result.extra.get("from_result_store") or result.extra.get("from_cache"))
            live.emit("cached" if cached else "completed",cached=cached)
            if not cached and not result.extra.get("r2_evidence",{}).get("resumed"):
                durations.append(time.perf_counter()-live.started)
            return result

    def _solve(self, group: str, method: str, solver: BaseSolver,
              seed: int) -> SolverResult:
        # Keep every cache/checkpoint inside the selected reproduction output
        # tree; importing or running the public files never scatters state in
        # the repository root.
        solver.training_policy = self.training_policy
        solver.cache_dir = self.root / "_solver_cache"
        path = self._path(group, method, solver, seed)
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    result = pickle.load(handle)["result"]
                result.extra = {**result.extra, "from_result_store": True}
                return result
            except (OSError, EOFError, pickle.UnpicklingError, KeyError):
                pass
        result = solver.solve(seed=int(seed))
        _atomic_pickle_dump({
            "protocol": PUBLICATION_PROTOCOL,
            "result": result,
        }, path)
        return result

    def solve_many(self, group: str, method: str, seeds: Any,
                   factory: Callable[[], BaseSolver]) -> list[SolverResult]:
        values = []
        seed_list = [int(seed) for seed in seeds]
        for index, seed in enumerate(seed_list, start=1):
            print(f"[{index:>2}/{len(seed_list)}] {group}/{method}, "
                  f"seed={seed}", flush=True)
            solver = factory()
            self._batch = (index,len(seed_list))
            try:
                values.append(self.solve(group, method, solver, seed))
            finally:
                self._batch = None
        return values


def result_at_fem_checkpoint(result: SolverResult,
                             budget: int) -> SolverResult:
    """Return the recorded best value at an exact FEM-call checkpoint.

    Population methods may complete a generation a few calls beyond the
    common checkpoint.  Their full trajectory is retained by the cached raw
    run; this lightweight publication result exposes only information known
    by ``budget`` calls and deliberately carries no later design field.
    """
    budget = int(budget)
    trajectory = [
        (int(call), float(value))
        for call, value in result.trajectory if int(call) <= budget]
    if result.fem_calls_total < budget or not trajectory:
        raise ValueError(
            f"Seed {result.seed} reached {result.fem_calls_total}, "
            f"not the requested {budget} FEM calls")
    best = min(value for _call, value in trajectory)
    return SolverResult(
        best_compliance=float(best), best_design=None,
        fem_calls_total=budget, trajectory=trajectory, seed=int(result.seed),
        wall_time_s=float(result.wall_time_s),
        extra={
            **result.extra,
            "fem_checkpoint": budget,
            "raw_run_fem_calls": int(result.fem_calls_total),
            "checkpoint_design_not_stored": True,
        })


def make_publication_ellipse_problem() -> EllipseProblem:
    return EllipseProblem(
        nelx=36, nely=18, Lx_mm=120.0, Ly_mm=60.0,
        E0=210000.0, nu=0.3, penal=3.0, q_line=100.0,
        thickness_mm=5.0, load_y1=25.0, load_y2=35.0,
        target_hole_radius_mm=10.0, beta=8.0, rho_void=0.001,
        fixed_area=True)


def make_publication_ellipse_box() -> EllipseBox:
    return EllipseBox(
        xc=(25.0, 90.0), yc=(15.0, 45.0), rx=(7.15, 14.0),
        ry=(7.14, 14.0), th=(0.0, math.pi))


def make_publication_lbracket_problem() -> LBracketProblem:
    return LBracketProblem(
        nelx=60, nely=36, volfrac=0.45, penal=3.0, rmin=2.4)


def make_publication_pisco_e1_config(**overrides: Any) -> PISCOConfig:
    values = dict(
        T_cycles=12, n_init=60, n_top=8, M=16, M_explore=256,
        multistart_steps=200, multistart_lr=0.4,
        surrogate_steps=400, surrogate_lr=0.002,
        lam_data=1.0, lam_phys=0.01, grad_clip_norm=1.0,
        surrogate_hidden=128, surrogate_depth=4,
        convergence_patience=3, convergence_warmup=3,
        convergence_tol_rel=0.0001)
    values.update(overrides)
    return PISCOConfig(**values)


def make_publication_pisco_e2_config(**overrides: Any
                                     ) -> RevisionPISCOConfigTopology:
    values = dict(
        T_cycles=12, n_init=40, n_top=6, short_simp_iters=6,
        generator_type="pca", latent_dim=16, latent_clip=3.0,
        M=12, multistart_steps=150, multistart_lr=0.05,
        surrogate_steps=300, surrogate_lr=0.002,
        lam_data=1.0, lam_phys=0.01, lam_z_prior=0.001,
        grad_clip_norm=1.0, active_retraining=True,
        diagnostic_holdout_size=16, diagnostic_top_k=5,
        convergence_patience=3, convergence_warmup=3,
        convergence_tol_rel=0.0001,
        adaptive_pca_enabled=True, adaptive_pca_max_dim=40,
        adaptive_pca_n_stages=6,
        adaptive_pca_expand_cycles=(1, 2, 3, 4, 5, 6),
        adaptive_pca_warm_start_fraction=0.5,
        adaptive_pca_convergence_after_final_expansion=True,
        latent_prior_reference_dim=16, local_candidate_pool_size=60,
        local_candidate_sigma=0.3)
    values.update(overrides)
    if values["generator_type"] != "pca":
        values["adaptive_pca_enabled"] = False
        values["local_candidate_pool_size"] = 0
    return RevisionPISCOConfigTopology(**values)


def make_publication_baseline_configs(
        ) -> tuple[DEConfig, BOConfig, CMAESConfig, SIMPConfig]:
    return (
        DEConfig(maxiter=30, popsize=10, polish=True),
        BOConfig(n_init=20, n_iters=136, convergence_patience=20,
                 convergence_warmup=30),
        CMAESConfig(n_generations=19, sigma_init=0.25,
                    convergence_patience=5),
        SIMPConfig(maxiter=80, tol=0.001, n_starts=1),
    )


# =====================================================================
# 15. FILTERED DYNAMIC-RANK TOPOLOGY SOLVER
# =====================================================================

"""Filtered topology protocol used for the EAAI major revision.

The revised Example-2 implementation applies the following physical-density
path consistently:

    bounded decoder field -> common cone filter -> exact volume projection

Consequently, the density seen by the surrogate and FEM already satisfies the
same spatial filtering restriction used for the revised SIMP comparison.  A
protocol identifier is included in cache keys and solver outputs so old and
new cached runs cannot be confused.
"""

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional



REVISION_PROTOCOL = "e2_exact_filter_and_projection_v4"


class FilteredVolumeProjected(nn.Module):
    """Revision decoder with a differentiable cone filter before projection.

    The cone kernel and boundary normalization reproduce ``_build_filter``.
    Values outside the L-bracket design domain are zero before filtering, as
    in SIMP, while the normalization still refers to the rectangular mesh.
    The final global shift-and-clip projection enforces the requested mean
    density over the active mask.
    """

    def __init__(self, inner: nn.Module, mask: np.ndarray, volfrac: float,
                 rmin: float = 2.4):
        super().__init__()
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.ndim != 2:
            raise ValueError("The revision filter requires a two-dimensional mask")
        if rmin <= 0.0:
            raise ValueError("rmin must be positive")
        self.inner = inner
        self.dz = int(inner.dz)
        self.max_dz = int(getattr(inner, "max_dz", inner.dz))
        self.vf = float(volfrac)
        self.rmin = float(rmin)
        self.height, self.width = mask_np.shape
        self.register_buffer(
            "mask2d", torch.tensor(mask_np, dtype=torch.bool)[None, None])

        radius = int(np.floor(self.rmin))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
        kernel = np.maximum(self.rmin - np.sqrt(xx * xx + yy * yy), 0.0)
        self.padding = radius
        self.register_buffer(
            "cone_kernel",
            torch.tensor(kernel, dtype=torch.float32)[None, None])
        with torch.no_grad():
            ones = torch.ones((1, 1, self.height, self.width))
            normalization = functional.conv2d(
                ones, self.cone_kernel, padding=self.padding)
        self.register_buffer("cone_normalization", normalization)

    def extend(self, new_dz: int) -> int:
        if hasattr(self.inner, "extend"):
            self.dz = int(self.inner.extend(new_dz))
        return self.dz

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        raw_decoder = getattr(self.inner, "decode_raw", None)
        return raw_decoder(z) if raw_decoder is not None else self.inner.decode(z)

    def _crop(self, rho: torch.Tensor) -> torch.Tensor:
        if rho.ndim != 4 or rho.shape[1] != 1:
            raise ValueError("Decoder output must have shape (batch, 1, ny, nx)")
        if rho.shape[-2] < self.height or rho.shape[-1] < self.width:
            raise ValueError(
                f"Decoder field {tuple(rho.shape[-2:])} is smaller than "
                f"mask {(self.height, self.width)}")
        return rho[..., :self.height, :self.width]

    def _filter(self, bounded: torch.Tensor) -> torch.Tensor:
        mask = self.mask2d.to(device=bounded.device)
        design_variables = torch.where(mask, bounded, torch.zeros_like(bounded))
        filtered = functional.conv2d(
            design_variables, self.cone_kernel.to(bounded),
            padding=self.padding)
        filtered = filtered / self.cone_normalization.to(bounded).clamp_min(1e-12)
        return torch.where(mask, filtered, torch.zeros_like(filtered))

    def _project(self, filtered: torch.Tensor, iters: int = 64) -> torch.Tensor:
        return project_volume_torch(filtered, self.mask2d, self.vf)

    def decode_with_diagnostics(self, z: torch.Tensor
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self._crop(self.decode_raw(z))
        bounded = raw.clamp(0.0, 1.0)
        filtered = self._filter(bounded)
        projected = self._project(filtered)
        batch = raw.shape[0]
        mask_flat = self.mask2d.reshape(-1).to(device=raw.device)
        raw_active = raw.reshape(batch, -1)[:, mask_flat]
        filtered_active = filtered.reshape(batch, -1)[:, mask_flat]
        projected_active = projected.reshape(batch, -1)[:, mask_flat]
        diagnostics = {
            "raw_min": raw_active.min(dim=1).values,
            "raw_max": raw_active.max(dim=1).values,
            "clipped_fraction": (
                (raw_active < 0.0) | (raw_active > 1.0)).float().mean(dim=1),
            "filtered_min": filtered_active.min(dim=1).values,
            "filtered_max": filtered_active.max(dim=1).values,
            "projected_volume": projected_active.mean(dim=1),
            "bounded_raw_design": torch.where(
                self.mask2d.to(device=bounded.device), bounded,
                torch.zeros_like(bounded)),
        }
        return projected, diagnostics

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        projected, _ = self.decode_with_diagnostics(z)
        return projected


def _project_numpy(filtered: np.ndarray, mask: np.ndarray, volfrac: float,
                   iters: int = 60) -> np.ndarray:
    """Exact shift-and-clip projection used for revision initial anchors."""
    out = np.zeros_like(np.asarray(filtered, dtype=float))
    active = np.asarray(filtered, dtype=float)[mask]
    lo, hi = -1.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.clip(active + mid, 0.0, 1.0).mean() < volfrac:
            lo = mid
        else:
            hi = mid
    out[mask] = np.clip(active + 0.5 * (lo + hi), 0.0, 1.0)
    return out


def _build_revision_simp_dataset(prob: LBracketProblem, n: int,
                                 seed: int, short_simp_iters: int = 6,
                                 init_amp: float = 0.20
                                 ) -> tuple[list[dict], int, list[tuple]]:
    """Short-SIMP anchors with distinct raw and physical density fields.

    ``generator_rho`` is the unfiltered design-variable field learned by the
    decoder. ``rho`` is filtered exactly once with the physical volume constrained by OC.
    This separation prevents the extra filter in the revision decoder from
    double-filtering the training manifold.
    """
    rng = np.random.default_rng(seed)
    mask = _l_bracket_mask(prob.nelx, prob.nely)
    force, fixed = _l_bracket_load(prob.nelx, prob.nely)
    filter_matrix, filter_sum = _build_filter(
        prob.nelx, prob.nely, prob.rmin)
    fem_calls = 0
    best_so_far = float("inf")
    trajectory: list[tuple[int, float]] = []
    dataset: list[dict] = []

    for _ in range(n):
        vf_init = float(np.clip(
            prob.volfrac + 0.15 * rng.standard_normal(), 0.30, 0.60))
        raw = np.zeros((prob.nely, prob.nelx), dtype=float)
        noise = rng.uniform(0.0, 1.0, size=raw.shape)
        noise = ((filter_matrix @ noise.ravel(order="F")) / filter_sum).reshape(
            raw.shape, order="F")
        raw[mask] = vf_init + init_amp * (
            noise[mask] - noise[mask].mean())
        raw = np.clip(raw, 0.001, 1.0)
        raw[~mask] = 0.0
        raw = _initial_filtered_volume(raw, mask, filter_matrix, filter_sum, prob.volfrac)
        n_iters = max(1, short_simp_iters + int(rng.integers(-2, 3)))

        evaluated_raw = raw.copy()
        for iteration in range(n_iters):
            evaluated_raw = raw.copy()
            rho = _density_filter(evaluated_raw, mask, filter_matrix, filter_sum)
            displacement, compliance, strain_energy, _ = _fea(
                prob.nelx, prob.nely, rho, fixed, force,
                penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
            fem_calls += 1
            best_so_far = min(best_so_far, float(compliance))
            _append_trajectory(trajectory, (fem_calls, best_so_far))
            if iteration == n_iters - 1:
                break

            physical_dc = (-prob.penal * (1. - 1e-9)
                           * np.power(np.maximum(rho, 1e-9), prob.penal - 1.) * strain_energy)
            raw = _density_oc_update(raw, physical_dc, mask, filter_matrix,
                                     filter_sum, prob.volfrac)

        dataset.append({
            "rho": rho.astype(np.float32),
            "generator_rho": evaluated_raw.astype(np.float32),
            "raw_design": evaluated_raw.astype(np.float32),
            "U": displacement.astype(np.float32),
            "compliance": float(compliance),
        })
    return dataset, fem_calls, trajectory


def _build_revision_generator_dataset(prob: LBracketProblem,
                                      generator: nn.Module, n_init: int,
                                      seed: int
                                      ) -> tuple[list[dict], int, list[tuple]]:
    """Initial data for fixed generators under the same revision filter."""
    mask = _l_bracket_mask(prob.nelx, prob.nely)
    force, fixed = _l_bracket_load(prob.nelx, prob.nely)
    projected = FilteredVolumeProjected(
        generator, mask, prob.volfrac, rmin=prob.rmin)
    torch_generator = torch.Generator().manual_seed(seed + 7)
    latent = torch.randn(2 * n_init, generator.dz,
                         generator=torch_generator)
    with torch.no_grad():
        physical, diagnostics = projected.decode_with_diagnostics(latent)
        fields = physical[:, 0].numpy()
        raw_fields = diagnostics["bounded_raw_design"][:, 0].numpy()

    evaluated: list[dict] = []
    trajectory: list[tuple[int, float]] = []
    best_so_far = float("inf")
    for index, rho in enumerate(fields, start=1):
        displacement, compliance, _, _ = _fea(
            prob.nelx, prob.nely, rho, fixed, force,
            penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
        best_so_far = min(best_so_far, float(compliance))
        _append_trajectory(trajectory, (index, best_so_far))
        evaluated.append({
            "rho": rho.astype(np.float32),
            "generator_rho": raw_fields[index - 1].astype(np.float32),
            "raw_design": raw_fields[index - 1].astype(np.float32),
            "U": displacement.astype(np.float32),
            "compliance": float(compliance),
        })
    evaluated.sort(key=lambda sample: sample["compliance"])
    return evaluated[:n_init], len(evaluated), trajectory


class RevisionPISCOSolverTopology(PISCOSolverTopology):
    """Topology solver using the revision-only filtered decoder protocol."""

    binary_objective = False

    def __init__(self, problem, config, **kwargs):
        super().__init__(problem, config, **kwargs)
        if not getattr(config, "adaptive_pca_enabled", False):
            return
        if config.generator_type != "pca":
            raise ValueError("adaptive PCA requires generator_type='pca'")
        cycles = tuple(int(cycle) for cycle in
                       config.adaptive_pca_expand_cycles)
        if len(cycles) != int(config.adaptive_pca_n_stages):
            raise ValueError(
                "adaptive_pca_expand_cycles must contain exactly "
                "adaptive_pca_n_stages entries")
        if (not cycles or any(cycle < 1 or cycle > config.T_cycles
                              for cycle in cycles)
                or tuple(sorted(set(cycles))) != cycles):
            raise ValueError(
                "adaptive PCA expansion cycles must be unique, increasing, "
                "and lie within 1..T_cycles")
        fraction = float(config.adaptive_pca_warm_start_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "adaptive_pca_warm_start_fraction must lie in [0, 1]")
        if int(getattr(config, "latent_prior_reference_dim", 1)) < 1:
            raise ValueError("latent_prior_reference_dim must be positive")
        if int(getattr(config, "local_candidate_pool_size", 0)) < 0:
            raise ValueError("local_candidate_pool_size must be non-negative")
        if float(getattr(config, "local_candidate_sigma", 0.0)) < 0.0:
            raise ValueError("local_candidate_sigma must be non-negative")

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload["revision_protocol"] = REVISION_PROTOCOL
        payload["revision_filter_rmin"] = float(self.problem.rmin)
        return payload

    def _adaptive_pca_schedule(self, generator: nn.Module
                               ) -> list[dict[str, int]]:
        """Return the predeclared nested PCA schedule for this anchor rank."""
        config = self.config
        if not getattr(config, "adaptive_pca_enabled", False):
            return []
        start = int(config.latent_dim)
        maximum = min(
            int(config.adaptive_pca_max_dim),
            int(getattr(generator, "max_dz", generator.dz)))
        maximum = max(start, maximum)
        stages = int(config.adaptive_pca_n_stages)
        dimensions = np.rint(np.linspace(start, maximum, stages)).astype(int)
        dimensions = np.maximum.accumulate(dimensions)
        return [
            {"cycle": int(cycle), "latent_dim": int(dimension)}
            for cycle, dimension in zip(
                config.adaptive_pca_expand_cycles, dimensions)
        ]

    @staticmethod
    def _target_latent_dim(cycle: int, schedule: list[dict[str, int]],
                           fallback: int) -> int:
        target = int(fallback)
        for stage in schedule:
            if int(stage["cycle"]) <= cycle:
                target = int(stage["latent_dim"])
        return target

    def _build_diagnostic_holdout(
            self, generator: nn.Module, seed: int) -> tuple[list[dict], int]:
        """Build holdout fields without changing the exact projection.

        The legacy helper floors active densities at 0.001 after decoding.
        That is harmless for an unconstrained display field, but changes the
        prescribed volume after the revision projection.
        """
        n_samples = max(0, int(self.config.diagnostic_holdout_size))
        if n_samples == 0:
            return [], 0
        random = torch.Generator().manual_seed(100_003 + int(seed))
        latent = torch.randn((n_samples, generator.dz), generator=random)
        latent.clamp_(-self.config.latent_clip, self.config.latent_clip)
        with torch.no_grad():
            density = generator.decode(latent)[
                ..., :self.problem.nely, :self.problem.nelx][:, 0]
            mask = torch.tensor(self._mask, dtype=torch.bool)
            density = torch.where(mask, density, torch.zeros_like(density))

        holdout: list[dict] = []
        for decoded in density.numpy():
            for representation, sample in (("continuous", decoded),
                    ("binary", volume_preserving_binary(decoded, self._mask, self.problem.volfrac))):
                displacement, compliance, _, _ = _fea(
                    self.problem.nelx, self.problem.nely, sample,
                    self._fixed, self._F_np, penal=self.problem.penal,
                    E0=1.0, Emin=1e-9, nu=0.30)
                holdout.append({"rho": sample.astype(np.float32),
                    "U": displacement.astype(np.float32), "compliance":float(compliance),
                    "representation":representation})
        return holdout, len(holdout)

    def _latent_prior(self, latent: torch.Tensor) -> torch.Tensor:
        """Dimension-neutral Gaussian prior anchored at the original d=16.

        At the reference dimension this is identical to the submitted
        ``lambda_z * ||z||²`` term.  For a nested expansion the factor
        ``d_ref / d`` prevents the regularisation strength from increasing
        merely because more PCA coordinates have become available.
        """
        reference = float(getattr(
            self.config, "latent_prior_reference_dim",
            self.config.latent_dim))
        dimension = max(int(latent.shape[-1]), 1)
        return (self.config.lam_z_prior * reference / dimension
                * latent.pow(2).sum(dim=-1))

    def _latent_optimize(
            self, generator: nn.Module, model: nn.Module,
            force: torch.Tensor, mask: torch.Tensor,
            initial_latent: np.ndarray | None = None,
            ) -> tuple[
                np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """Optimize in latent space with warm starts and fresh exploration.

        Up to the configured fraction of starts is copied from the preceding
        cycle.  New PCA coordinates are padded with zeros, which preserves the
        decoded design exactly for a nested additive PCA basis.  Remaining
        starts stay random so dimensional expansion does not suppress global
        exploration.
        """
        config = self.config
        problem = self.problem
        latent = torch.randn(config.M, generator.dz)
        if initial_latent is not None:
            previous = np.asarray(initial_latent, dtype=np.float32)
            if previous.ndim != 2:
                raise ValueError("initial_latent must be a two-dimensional array")
            fraction = float(getattr(
                config, "adaptive_pca_warm_start_fraction", 0.0))
            n_warm = min(
                len(previous), config.M,
                int(math.ceil(config.M * fraction)))
            n_dimensions = min(previous.shape[1], generator.dz)
            if n_warm > 0 and n_dimensions > 0:
                with torch.no_grad():
                    latent[:n_warm, :n_dimensions] = torch.tensor(
                        previous[:n_warm, :n_dimensions])
                    if n_dimensions < generator.dz:
                        latent[:n_warm, n_dimensions:] = 0.0
        def objective(latent):
            density = generator.decode(latent)[..., :problem.nely, :problem.nelx]
            physical = density[:, 0] * mask
            return model(physical.unsqueeze(1)) @ force + self._latent_prior(latent)
        latent, _ = _packet_adam(latent, objective, steps=config.multistart_steps,
            lr=config.multistart_lr,
            project=lambda value: value.clamp(-config.latent_clip, config.latent_clip))

        with torch.no_grad():
            density, diagnostics = generator.decode_with_diagnostics(latent)
            density = density[..., :problem.nely, :problem.nelx][:, 0]
            physical = torch.where(
                mask.bool(), density, torch.zeros_like(density))
            predicted_tensor = model(physical.unsqueeze(1)) @ force
            ranking = predicted_tensor + self._latent_prior(latent)
            predicted = ranking.numpy()
        diagnostics_numpy = {
            key: value.detach().cpu().numpy()
            for key, value in diagnostics.items()
        }

        # Cheap CTF-style exploration: sample a larger local cloud around the
        # best preceding latent point, rank it with the current surrogate, and
        # send no additional candidates to FEM.  Thus the exact FEM budget
        # remains n_top per active cycle.
        pool_size = int(getattr(config, "local_candidate_pool_size", 0))
        if pool_size > 0:
            centre = np.zeros(generator.dz, dtype=np.float32)
            if initial_latent is not None and len(initial_latent):
                previous = np.asarray(initial_latent[0], dtype=np.float32)
                centre[:min(len(previous), generator.dz)] = previous[
                    :generator.dz]
            local = (torch.tensor(centre).reshape(1, -1)
                     + float(config.local_candidate_sigma)
                     * torch.randn(pool_size, generator.dz))
            local.clamp_(-config.latent_clip, config.latent_clip)
            # Retain the exact zero-padded incumbent as one member; the other
            # pool points are stochastic perturbations as in legacy CTF.
            local[0] = torch.tensor(centre)
            with torch.no_grad():
                local_density, local_diagnostics = (
                    generator.decode_with_diagnostics(local))
                local_density = local_density[
                    ..., :problem.nely, :problem.nelx][:, 0]
                local_physical = torch.where(
                    mask.bool(), local_density,
                    torch.zeros_like(local_density))
                local_prediction = (
                    model(local_physical.unsqueeze(1)) @ force
                    + self._latent_prior(local))
            physical = torch.cat(
                [torch.tensor(physical.numpy()), local_physical], dim=0)
            predicted = np.concatenate([
                predicted, local_prediction.cpu().numpy()])
            latent_output = np.concatenate([
                latent.detach().cpu().numpy(), local.cpu().numpy()], axis=0)
            diagnostics_numpy = {
                key: np.concatenate([
                    diagnostics_numpy[key],
                    value.detach().cpu().numpy()], axis=0)
                for key, value in local_diagnostics.items()
            }
        else:
            latent_output = latent.detach().cpu().numpy()
        return (
            physical.numpy(), predicted, diagnostics_numpy,
            latent_output)

    def _initial_dataset(self, seed: int
                         ) -> tuple[list[dict], int, list[tuple]]:
        """Create revision anchors; called only when no checkpoint exists."""
        problem = self.problem
        config = self.config
        if config.generator_type in {"lattice", "lattice_wavelet", "wavelet"}:
            generator, _ = self._generator_from_dataset(None, None)
            return _build_revision_generator_dataset(
                problem, generator, config.n_init, seed)
        if config.simp_free_anchors:
            anchor_generator = _make_lattice_decoder(
                problem.nelx, problem.nely,
                dz=max(13, config.latent_dim), mask=self._mask,
                volfrac=problem.volfrac)
            return _build_revision_generator_dataset(
                problem, anchor_generator, config.n_init, seed)
        return _build_revision_simp_dataset(
            problem, config.n_init, seed,
            short_simp_iters=config.short_simp_iters,
            init_amp=config.anchor_init_amp)

    def _generator_from_dataset(
            self, dataset: list[dict] | None,
            restored_state: dict[str, torch.Tensor] | None
            ) -> tuple[nn.Module, dict[str, Any]]:
        """Construct or restore the inner generator without additional FEM."""
        problem = self.problem
        config = self.config
        if config.generator_type == "lattice":
            generator = _make_lattice_decoder(
                problem.nelx, problem.nely, dz=config.latent_dim,
                mask=self._mask, volfrac=problem.volfrac)
            info = {"final_recon": 0.0}
        elif config.generator_type == "lattice_wavelet":
            generator = _make_lattice_wavelet_decoder(
                problem.nelx, problem.nely, dz=config.latent_dim,
                mask=self._mask, volfrac=problem.volfrac,
                residual_weight=config.lattice_wavelet_weight,
                wavelet_levels=config.lattice_wavelet_levels)
            info = {"final_recon": 0.0}
        elif config.generator_type == "wavelet":
            generator = _make_wavelet_decoder(
                problem.nelx, problem.nely, dz=config.latent_dim,
                levels=5, slope=6.0, mask=self._mask)
            info = {"final_recon": 0.0}
        else:
            if dataset is None:
                raise ValueError("A learned generator requires anchor fields")
            # The decoder is frozen after the initial anchor set. Active FEM
            # samples appended later must not change its architecture during
            # checkpoint reconstruction (notably PCA's available mode count).
            generator_fields = np.stack([
                sample["generator_rho"]
                for sample in dataset[:config.n_init]])
            if config.generator_type == "vae":
                generator = _make_conv_vae(dz=config.latent_dim)
                if restored_state is None:
                    fields = torch.tensor(
                        generator_fields[:, None], dtype=torch.float32)
                    info = _train_vae(
                        generator, fields, epochs=config.vae_epochs,
                        lr=config.vae_lr, beta_kl=config.vae_beta_kl)
                else:
                    info = {"final_recon": None, "restored": True}
            elif config.generator_type == "ae":
                generator = _make_conv_ae(dz=config.latent_dim)
                if restored_state is None:
                    fields = torch.tensor(
                        generator_fields[:, None], dtype=torch.float32)
                    info = _train_ae(
                        generator, fields, epochs=config.vae_epochs,
                        lr=config.vae_lr)
                else:
                    info = {"final_recon": None, "restored": True}
            else:
                generator, info = _make_pca_decoder(
                    generator_fields, dz=config.latent_dim)
        if restored_state is not None:
            generator.load_state_dict(restored_state)
        return generator, info

    def _make_revision_model(self, displacement_scale):
        return _make_field_surrogate_cnn(ndof=self._ndof, u_scale=displacement_scale, fixed_dofs=self._fixed)

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        torch.manual_seed(seed + 1)
        problem = self.problem
        config = self.config

        if config.coarse_to_fine:
            raise ValueError(
                "legacy coarse_to_fine is surrogate-free and is not part of "
                "the revision protocol; use adaptive_pca_enabled instead")

        checkpoint = load_checkpoint(self, seed) if self.cache_enabled else None
        resumed = checkpoint is not None
        if resumed:
            dataset = checkpoint["dataset"]
            fem_calls = int(checkpoint["fem_calls"])
            trajectory = list(checkpoint["trajectory"])
            generator, rebuilt_info = self._generator_from_dataset(
                dataset, checkpoint["generator_state"])
            generator_info = checkpoint.get("generator_info", rebuilt_info)
        else:
            dataset, fem_calls, trajectory = self._initial_dataset(seed)
            generator, generator_info = self._generator_from_dataset(
                dataset, None)

        anchor_trajectory = (checkpoint.get("initial_anchor_trajectory", []) if resumed
                             else list(trajectory))
        if self.binary_objective and not resumed:
            # Continuous anchors train the model but are not admissible binary
            # incumbents. The first actually evaluated binary candidate wins.
            trajectory = []

        generator = FilteredVolumeProjected(
            generator, self._mask, problem.volfrac, rmin=problem.rmin)
        adaptive_schedule = self._adaptive_pca_schedule(generator)
        if resumed:
            generator.extend(int(checkpoint.get(
                "active_latent_dim", config.latent_dim)))
        for parameter in generator.parameters():
            parameter.requires_grad_(False)

        if resumed:
            holdout = checkpoint["holdout"]
            diagnostic_fem_calls = int(checkpoint["diagnostic_fem_calls"])
        else:
            holdout, diagnostic_fem_calls = self._build_diagnostic_holdout(
                generator, seed)
        force_tensor = torch.tensor(self._F_np, dtype=torch.float32)
        displacement_scale = (
            float(checkpoint["displacement_scale"])
            if resumed else float(np.std([
                np.linalg.norm(sample["U"]) for sample in dataset])
                / math.sqrt(self._ndof)))
        model = self._make_revision_model(displacement_scale)
        if resumed:
            model.load_state_dict(checkpoint["model_state"])
            for name, value in checkpoint.get("training_state", {}).items():
                setattr(model, name, value)
        free_indices = torch.tensor(
            np.setdiff1d(np.arange(self._ndof), self._fixed),
            dtype=torch.long)
        force_norm = float(np.linalg.norm(self._F_np))
        mask_tensor = torch.tensor(self._mask, dtype=torch.float32)

        if resumed:
            best = checkpoint["best"]
            stagnation = int(checkpoint["stagnation"])
            actual_cycles = int(checkpoint["actual_cycles"])
            stop_reason = str(checkpoint["stop_reason"])
            cycle_metrics = list(checkpoint["cycle_metrics"])
            training_info = checkpoint["training_info"]
            timing = dict(checkpoint["timing"])
            decoder_minima = list(checkpoint["decoder_minima"])
            decoder_maxima = list(checkpoint["decoder_maxima"])
            clipped_fractions = list(checkpoint["clipped_fractions"])
            projected_volumes = list(checkpoint["projected_volumes"])
            latent_warm_start = checkpoint.get("latent_warm_start")
            dimension_history = list(checkpoint.get("dimension_history", []))
            start_cycle = int(checkpoint["completed_cycle"]) + 1
            if stop_reason == "convergence":
                start_cycle = config.T_cycles + 1
            elapsed_before = float(checkpoint.get("elapsed_s", 0.0))
            torch.set_rng_state(checkpoint["torch_rng_state"])
        else:
            best = (None if self.binary_objective else
                    min(dataset, key=lambda sample: sample["compliance"]))
            stagnation = 0
            actual_cycles = config.T_cycles
            stop_reason = "T_cycles_cap"
            cycle_metrics: list[dict] = []
            training_info = None
            timing = {
                "surrogate_training_s": 0.0,
                "latent_optimisation_s": 0.0,
                "active_fem_s": 0.0,
            }
            decoder_minima: list[float] = []
            decoder_maxima: list[float] = []
            clipped_fractions: list[float] = []
            projected_volumes: list[float] = []
            latent_warm_start: np.ndarray | None = None
            dimension_history: list[dict[str, Any]] = []
            start_cycle = 1
            elapsed_before = 0.0

        def persist_checkpoint(completed_cycle: int) -> None:
            if not self.cache_enabled:
                return
            save_checkpoint(self, seed, {
                "completed_cycle": int(completed_cycle),
                "dataset": dataset,
                "fem_calls": int(fem_calls),
                "trajectory": list(trajectory),
                "initial_anchor_trajectory": anchor_trajectory,
                "generator_state": {
                    key: value.detach().cpu()
                    for key, value in generator.inner.state_dict().items()
                },
                "generator_info": generator_info,
                "training_state": {name:getattr(model,name) for name in
                    ("_r2_optimizer", "_r2_updates", "_r2_validation") if hasattr(model,name)},
                "holdout": holdout,
                "diagnostic_fem_calls": int(diagnostic_fem_calls),
                "displacement_scale": float(displacement_scale),
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "best": best,
                "stagnation": int(stagnation),
                "actual_cycles": int(actual_cycles),
                "stop_reason": stop_reason,
                "cycle_metrics": list(cycle_metrics),
                "training_info": training_info,
                "timing": dict(timing),
                "decoder_minima": list(decoder_minima),
                "decoder_maxima": list(decoder_maxima),
                "clipped_fractions": list(clipped_fractions),
                "projected_volumes": list(projected_volumes),
                "latent_warm_start": (
                    None if latent_warm_start is None
                    else np.asarray(latent_warm_start, dtype=np.float32)),
                "active_latent_dim": int(generator.dz),
                "adaptive_pca_schedule": list(adaptive_schedule),
                "dimension_history": list(dimension_history),
                "torch_rng_state": torch.get_rng_state(),
                "elapsed_s": (
                    elapsed_before + time.perf_counter() - started),
            })

        if not resumed:
            # The expensive anchor FEM evaluations and trained generator are
            # durable before the first active cycle starts.
            persist_checkpoint(0)

        for cycle in range(start_cycle, config.T_cycles + 1):
            _progress_cycle(cycle, config.T_cycles)
            compliance_before = best["compliance"] if best is not None else math.inf
            latent_dim_before = int(generator.dz)
            target_latent_dim = self._target_latent_dim(
                cycle, adaptive_schedule, generator.dz)
            generator.extend(target_latent_dim)
            dimension_expanded = int(generator.dz) > latent_dim_before
            if config.active_retraining or training_info is None:
                tick = time.perf_counter()
                training_info = self._train_surrogate(
                    model, dataset, force_tensor, free_indices, force_norm)
                timing["surrogate_training_s"] += time.perf_counter() - tick

            tick = time.perf_counter()
            (density_candidates, predicted, decoder_diagnostics,
             optimized_latent) = (
                self._latent_optimize(
                    generator, model, force_tensor, mask_tensor,
                    initial_latent=latent_warm_start))
            timing["latent_optimisation_s"] += time.perf_counter() - tick
            latent_warm_start = optimized_latent[np.argsort(predicted)]
            dimension_history.append({
                "cycle": int(cycle),
                "latent_dim": int(generator.dz),
                "expanded": bool(dimension_expanded),
            })
            decoder_minima.extend(decoder_diagnostics["raw_min"].tolist())
            decoder_maxima.extend(decoder_diagnostics["raw_max"].tolist())
            clipped_fractions.extend(
                decoder_diagnostics["clipped_fraction"].tolist())
            projected_volumes.extend(
                decoder_diagnostics["projected_volume"].tolist())

            for candidate_index in np.argsort(predicted)[:config.n_top]:
                rho = density_candidates[candidate_index]
                with torch.no_grad():
                    predicted_compliance = float((model(torch.tensor(rho[None,None], dtype=torch.float32)) @ force_tensor)[0])
                tick = time.perf_counter()
                displacement, compliance, _, element_stiffness = _fea(
                    problem.nelx, problem.nely, rho,
                    self._fixed, self._F_np, penal=problem.penal,
                    E0=1.0, Emin=1e-9, nu=0.30)
                timing["active_fem_s"] += time.perf_counter() - tick
                _record_prediction(predicted_compliance, compliance, cycle, rho)
                fem_calls += 1
                raw_design = np.asarray(
                    decoder_diagnostics["bounded_raw_design"][
                        candidate_index, 0, :problem.nely, :problem.nelx],
                    dtype=np.float32)
                entry = {
                    "rho": rho.astype(np.float32),
                    "generator_rho": raw_design,
                    "raw_design": raw_design,
                    "U": displacement.astype(np.float32),
                    "compliance": float(compliance),
                    "Eelem": element_stiffness.astype(np.float32),
                    "decoder_raw_min": float(
                        decoder_diagnostics["raw_min"][candidate_index]),
                    "decoder_raw_max": float(
                        decoder_diagnostics["raw_max"][candidate_index]),
                    "decoder_clipped_fraction": float(
                        decoder_diagnostics["clipped_fraction"][candidate_index]),
                    "decoder_projected_volume": float(
                        decoder_diagnostics["projected_volume"][candidate_index]),
                }
                dataset.append(entry)
                if best is None or compliance < best["compliance"]:
                    best = entry
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))

            diagnostic = self._diagnose_surrogate(
                model, holdout, force_tensor, free_indices, force_norm)
            cycle_metrics.append({
                "cycle": cycle,
                "n_dataset": len(dataset),
                "fem_calls": fem_calls,
                "best_compliance": float(best["compliance"]),
                "latent_dim": int(generator.dz),
                "dimension_expanded": bool(dimension_expanded),
                "surrogate_rel_err": training_info["surrogate_rel_err"],
                "surrogate_loss_data": training_info["final_data"],
                "surrogate_loss_phys": training_info["final_phys"],
                "active_retraining": bool(config.active_retraining),
                **diagnostic,
            })

            patience = config.convergence_patience
            should_stop = False
            convergence_warmup = int(config.convergence_warmup)
            if (adaptive_schedule and getattr(
                    config,
                    "adaptive_pca_convergence_after_final_expansion",
                    True)):
                convergence_warmup = max(
                    convergence_warmup,
                    int(adaptive_schedule[-1]["cycle"]))
            if (patience is not None and patience > 0
                    and cycle > convergence_warmup):
                tolerance = config.convergence_tol_rel * max(
                    compliance_before, 1e-12)
                improved = (not math.isfinite(compliance_before) or
                    (compliance_before - best["compliance"]) > tolerance)
                stagnation = 0 if improved else stagnation + 1
                if stagnation >= patience:
                    stop_reason = "convergence"
                    actual_cycles = cycle
                    should_stop = True
            # At most the currently running cycle is repeated after a hard
            # interruption. The checkpoint contains arrays, model weights,
            # diagnostics, RNG state and the complete FEM trajectory.
            persist_checkpoint(cycle)
            if should_stop:
                break

        if best is None:
            raise RuntimeError("No evaluated candidate is available for the requested objective")
        full_dimension = int(self._mask.sum())
        print(
            f"[PISCO revision] full={full_dimension} | latent={generator.dz} | "
            f"reduction={full_dimension / max(generator.dz, 1):.0f}x | "
            f"filter rmin={problem.rmin}", flush=True)
        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=np.asarray(best["rho"], dtype=np.float32),
            fem_calls_total=int(fem_calls), trajectory=list(trajectory),
            seed=int(seed), wall_time_s=float(
                elapsed_before + time.perf_counter() - started),
            extra={
                "revision_protocol": REVISION_PROTOCOL,
                "initial_anchor_trajectory": anchor_trajectory,
                "trajectory_representation": ("binary_active_search" if self.binary_objective
                                               else "continuous_initial_and_active"),
                "resumed_from_cycle_checkpoint": bool(resumed),
                "common_filter_in_optimisation": True,
                "common_filter_rmin": float(problem.rmin),
                "filter_order": (
                    "decoder_clip -> cone_filter -> volume_projection"),
                "generator_type": config.generator_type,
                "latent_dim": generator.dz,
                "adaptive_pca": bool(adaptive_schedule),
                "adaptive_pca_schedule": list(adaptive_schedule),
                "adaptive_pca_dimension_history": list(dimension_history),
                "latent_warm_start_fraction": float(getattr(
                    config, "adaptive_pca_warm_start_fraction", 0.0)),
                "latent_prior_reference_dim": int(getattr(
                    config, "latent_prior_reference_dim", generator.dz)),
                "local_candidate_pool_size": int(getattr(
                    config, "local_candidate_pool_size", 0)),
                "local_candidate_sigma": float(getattr(
                    config, "local_candidate_sigma", 0.0)),
                "generator_recon": generator_info.get("final_recon"),
                "best_U": np.asarray(best["U"], dtype=np.float32),
                "best_Eelem": (
                    np.asarray(best.get("Eelem"), dtype=np.float32)
                    if best.get("Eelem") is not None else None),
                "raw_design": np.asarray(
                    best.get("raw_design", best["rho"]), dtype=np.float32),
                "dataset_size": len(dataset),
                "full_design_dim": full_dimension,
                "latent_dim_final": int(generator.dz),
                "actual_T": actual_cycles,
                "stop_reason": stop_reason,
                "cycle_metrics": cycle_metrics,
                "diagnostic_holdout_size": len(holdout),
                "diagnostic_fem_calls_excluded": diagnostic_fem_calls,
                "timing_breakdown_s": timing,
                "best_decoder_raw_min": best.get("decoder_raw_min"),
                "best_decoder_raw_max": best.get("decoder_raw_max"),
                "best_decoder_clipped_fraction": best.get(
                    "decoder_clipped_fraction"),
                "best_decoder_projected_volume": best.get(
                    "decoder_projected_volume"),
                "decoder_diagnostics_all_candidates": {
                    "n": len(decoder_minima),
                    "raw_min": (float(np.min(decoder_minima))
                                if decoder_minima else None),
                    "raw_max": (float(np.max(decoder_maxima))
                                if decoder_maxima else None),
                    "clipped_fraction_mean": (
                        float(np.mean(clipped_fractions))
                        if clipped_fractions else None),
                    "clipped_fraction_max": (
                        float(np.max(clipped_fractions))
                        if clipped_fractions else None),
                    "projected_volume_max_abs_error": (
                        float(np.max(np.abs(
                            np.asarray(projected_volumes) - problem.volfrac)))
                        if projected_volumes else None),
                },
            },
        )


__all__ = [
    "FilteredVolumeProjected",
    "REVISION_PROTOCOL",
    "RevisionPISCOSolverTopology",
]


# =====================================================================
# 16. COMPONENT-CONTROL SOLVERS
# =====================================================================

"""Ablation-Solver für Phase B3 der EAAI-Revision.

Drei Ablationen, die die Architekturentscheidungen von PISCO isolieren:

1. ScalarSurrogateSolver  — skalares MLP statt Feld-CNN
2. NoRevalSolver          — budgetgleich, Surrogat nach Zyklus 1 eingefroren
3. LatentFEMSolver        — PCA-Latent + exaktes FEM/Adjoint (kein Surrogat)
"""
import sys, time, math, hashlib, json, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn



# =====================================================================
# 1. Scalar Surrogate — MLP/CNN, das nur J vorhersagt (nicht U)
# =====================================================================

def _make_scalar_surrogate_cnn() -> nn.Module:
    """CNN, das rho (1,36,60) → skalare Compliance J abbildet.
    Gleiche Encoder-Architektur wie der Feld-Surrogat, aber nur 1 Ausgang."""
    class _ScalarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            feat = 32 * 9 * 15
            self.head = nn.Sequential(
                nn.Linear(feat, 256), nn.Tanh(),
                nn.Linear(256, 128), nn.Tanh(),
                nn.Linear(128, 1))
        def forward(self, rho):
            return self.head(self.enc(rho)).squeeze(-1)
    return _ScalarCNN()


class ScalarSurrogateSolver(RevisionPISCOSolverTopology):
    """PISCO mit skalarem Surrogat (nur J-Vorhersage, kein Verschiebungsfeld).

    Ablation R1.6, R3: Zeigt den Mehrwert des feldbasierten Ansatzes."""

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        torch.manual_seed(seed + 1)
        prob = self.problem
        cfg = self.config

        dataset, fem_calls, init_traj = _build_revision_simp_dataset(
            prob, cfg.n_init, seed,
            short_simp_iters=cfg.short_simp_iters,
            init_amp=cfg.anchor_init_amp)
        fields_np = np.stack([s["generator_rho"] for s in dataset])
        gen, gen_info = _make_pca_decoder(fields_np, dz=cfg.latent_dim)
        gen = FilteredVolumeProjected(
            gen, self._mask, prob.volfrac, rmin=prob.rmin)
        trajectory = list(init_traj)
        for p in gen.parameters():
            p.requires_grad_(False)

        mask_t = torch.tensor(self._mask, dtype=torch.float32)
        model = _make_scalar_surrogate_cnn()

        best = min(dataset, key=lambda s: s["compliance"])
        for t in range(1, cfg.T_cycles + 1):
            _progress_cycle(t, cfg.T_cycles)
            self._train_scalar_surrogate(model, dataset)
            rho_cands, J_cands, raw_cands = self._latent_optimize_scalar(
                gen, model, mask_t)
            order = np.argsort(J_cands)
            top = [rho_cands[i] for i in order[:cfg.n_top]]
            for candidate_index, rho in zip(order[:cfg.n_top], top):
                U, comp, _, Eelem = _fea(
                    prob.nelx, prob.nely, rho, self._fixed, self._F_np,
                    penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
                fem_calls += 1
                entry = {"rho": rho.astype(np.float32),
                         "raw_design": raw_cands[candidate_index].astype(
                             np.float32),
                         "U": U.astype(np.float32),
                         "compliance": float(comp),
                         "Eelem": Eelem.astype(np.float32)}
                dataset.append(entry)
                if comp < best["compliance"]:
                    best = entry
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))

        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=np.asarray(best["rho"], dtype=np.float32),
            fem_calls_total=int(fem_calls),
            trajectory=trajectory, seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={"ablation": "scalar_surrogate",
                   "revision_protocol": REVISION_PROTOCOL,
                   "common_filter_in_optimisation": True,
                   "raw_design": np.asarray(
                       best.get("raw_design", best["rho"]), dtype=np.float32),
                   "generator_type": "pca",
                   "dataset_size": len(dataset)})

    def _train_scalar_surrogate(self, model, dataset):
        cfg = self.config
        optim = torch.optim.Adam(model.parameters(), lr=cfg.surrogate_lr)
        Rho = torch.tensor(np.stack([s["rho"] for s in dataset])[:, None, :, :],
                           dtype=torch.float32)
        J = torch.tensor([s["compliance"] for s in dataset], dtype=torch.float32)
        for _ in range(cfg.surrogate_steps):
            optim.zero_grad()
            J_hat = model(Rho)
            loss = ((J_hat - J) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()

    def _latent_optimize_scalar(self, gen, model, mask_t):
        cfg = self.config
        prob = self.problem
        Z = torch.randn(cfg.M, gen.dz, requires_grad=True)
        def objective(Z):
            rho = gen.decode(Z)[..., :prob.nely, :prob.nelx][:, 0] * mask_t
            return model(rho.unsqueeze(1)) + cfg.lam_z_prior * Z.pow(2).sum(dim=-1)
        Z, _ = _packet_adam(Z, objective, steps=cfg.multistart_steps,
            lr=cfg.multistart_lr, project=lambda value: value.clamp(-cfg.latent_clip, cfg.latent_clip))
        with torch.no_grad():
            rho_final, diagnostics = gen.decode_with_diagnostics(Z)
            rho_final = rho_final[..., :prob.nely, :prob.nelx][:, 0]
            rho_final = torch.where(
                mask_t.bool(), rho_final, torch.zeros_like(rho_final))
            J_pred = model(rho_final.unsqueeze(1)).numpy()
            raw = diagnostics["bounded_raw_design"][
                :, 0, :prob.nely, :prob.nelx]
        return rho_final.numpy(), J_pred, raw.numpy()


# =====================================================================
# 2. Ohne aktive Re-Evaluation — nur 1 Zyklus
# =====================================================================

class NoRevalSolver(RevisionPISCOSolverTopology):
    """Budgetgleiche Einmaltrainings-Ablation ohne aktives Modellupdate.

    Das Surrogat wird im ersten Zyklus trainiert und danach eingefroren.
    Alle ``T_cycles * n_top`` FEM-Prüfungen werden weiterhin ausgeführt,
    fließen aber nicht ins Modell zurück. Damit unterscheidet sich diese
    Ablation von PISCO nur durch das fehlende aktive Retraining und besitzt
    dasselbe Optimierungsbudget."""

    def __init__(self, problem, config, **kwargs):
        import dataclasses
        cfg_fixed = dataclasses.replace(
            config, active_retraining=False, convergence_patience=None)
        super().__init__(problem, cfg_fixed, **kwargs)


# =====================================================================
# 3. PCA-Latent + exaktes FEM (kein Surrogat)
# =====================================================================

class LatentFEMSolver(RevisionPISCOSolverTopology):
    """PCA-Latentoptimierung mit exakter FEM und Adjoint-Gradienten.

    Ablation R1.6: Klärt, ob der Gewinn vom Surrogat-Critic, vom
    Active Loop oder vom niedrigdimensionalen Suchraum kommt.

    Die Compliance-Sensitivität
    ``dc/drho = -p (E0-Emin) rho**(p-1) ce`` wird aus derselben FEM-Lösung
    gewonnen und durch den differenzierbaren Decoder nach ``z``
    zurückpropagiert. Das aktive FEM-Budget entspricht PISCO."""

    def _compute(self, *, seed: int) -> SolverResult:
        t0 = time.perf_counter()
        torch.manual_seed(seed + 1)
        prob = self.problem
        cfg = self.config

        dataset, fem_calls, init_traj = _build_revision_simp_dataset(
            prob, cfg.n_init, seed,
            short_simp_iters=cfg.short_simp_iters,
            init_amp=cfg.anchor_init_amp)
        fields_np = np.stack([s["generator_rho"] for s in dataset])
        gen, gen_info = _make_pca_decoder(fields_np, dz=cfg.latent_dim)
        gen = FilteredVolumeProjected(
            gen, self._mask, prob.volfrac, rmin=prob.rmin)
        for p in gen.parameters():
            p.requires_grad_(False)

        trajectory = list(init_traj)
        best = min(dataset, key=lambda s: s["compliance"])
        mask = np.asarray(self._mask, bool)
        mask_t = torch.tensor(mask, dtype=torch.bool)

        # Identisches aktives Budget wie PISCO: standardmäßig 12*6 = 72.
        remaining_budget = cfg.T_cycles * cfg.n_top  # typically 12*6 = 72
        batch_size = min(max(1, cfg.M), remaining_budget)
        Z = torch.randn(batch_size, gen.dz, requires_grad=True)
        opt_z = torch.optim.Adam([Z], lr=cfg.multistart_lr)
        evaluated = 0
        adjoint_steps = 0
        while evaluated < remaining_budget:
            current = min(batch_size, remaining_budget - evaluated)
            opt_z.zero_grad()
            rho_decoded, diagnostics = gen.decode_with_diagnostics(Z)
            rho_all = rho_decoded[:current, 0, :prob.nely, :prob.nelx]
            raw_all = diagnostics["bounded_raw_design"][
                :current, 0, :prob.nely, :prob.nelx]
            rho_all = torch.where(mask_t, rho_all,
                                  torch.zeros_like(rho_all))
            dc_batch = []
            for i in range(current):
                rho = rho_all[i].detach().cpu().numpy()
                U, comp, ce, _ = _fea(
                    prob.nelx, prob.nely, rho, self._fixed, self._F_np,
                    penal=prob.penal, E0=1.0, Emin=1e-9, nu=0.30)
                dc = (-prob.penal * (1.0 - 1e-9)
                      * np.power(np.maximum(rho, 1e-9), prob.penal - 1.0)
                      * ce)
                dc[~mask] = 0.0
                dc_batch.append(dc.astype(np.float32))
                fem_calls += 1
                evaluated += 1
                if comp < best["compliance"]:
                    best = {"rho": rho.astype(np.float32),
                            "raw_design": raw_all[i].detach().cpu().numpy(
                                ).astype(np.float32),
                            "U": U.astype(np.float32),
                            "compliance": float(comp)}
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))
            torch.autograd.backward(
                rho_all, torch.tensor(np.stack(dc_batch), dtype=torch.float32))
            if cfg.lam_z_prior:
                (cfg.lam_z_prior * Z.pow(2).sum()).backward()
            opt_z.step()
            with torch.no_grad():
                Z.clamp_(-cfg.latent_clip, cfg.latent_clip)
            adjoint_steps += 1

        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=np.asarray(best["rho"], dtype=np.float32),
            fem_calls_total=int(fem_calls),
            trajectory=trajectory, seed=int(seed),
            wall_time_s=float(time.perf_counter() - t0),
            extra={"ablation": "latent_fem_adjoint",
                   "revision_protocol": REVISION_PROTOCOL,
                   "common_filter_in_optimisation": True,
                   "raw_design": np.asarray(
                       best.get("raw_design", best["rho"]), dtype=np.float32),
                   "generator_type": "pca",
                   "n_candidates": evaluated,
                   "adjoint_steps": adjoint_steps,
                   "batch_size": batch_size,
                   "dataset_size": len(dataset)})


# =====================================================================
# Convenience: alle Ablationen als Dict
# =====================================================================

ABLATION_SOLVERS = {
    "scalar_surrogate": ScalarSurrogateSolver,
    "no_reeval": NoRevalSolver,
    "latent_fem": LatentFEMSolver,
}


# =====================================================================
# 17. MANUFACTURING-AWARE TOPOLOGY SOLVERS
# =====================================================================

"""Surrogate-guided hybrid PISCO with manufacturing-projected candidates.

The direct-FEM manufacturing control showed that the exact-cardinality binary
projection carries most of the legacy coarse-to-fine gain once the common
cone filter and volume projection are enforced.  This solver moves that
projection into the surrogate-guided main loop:

    gradient search   : continuous filtered decode (unchanged, differentiable)
    candidate ranking : decoder clip -> exact-cardinality binary projection
                        -> common cone filter -> exact volume projection
                        -> surrogate prediction
    FEM validation    : the same manufacturing-projected physical field

The non-differentiable projection is confined to the no-grad ranking and
validation path, so the FEM budget stays at ``n_top`` per active cycle and
the method remains budget-identical to the revised hybrid main method.

Its protocol and source hash enter the cache payload, so results from a
different candidate-evaluation rule cannot be mistaken for this one.
"""

import hashlib
from pathlib import Path
from typing import Any

import torch



MANUFACTURING_SOLVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MANUFACTURING_HYBRID_PROTOCOL = "surrogate_hybrid_manufacturing_projection_v1"


def manufacturing_binarize(bounded: torch.Tensor, mask2d: torch.Tensor,
                           volfrac: float) -> torch.Tensor:
    """Exact-cardinality top-k binarisation on the active design mask.

    Identical to the projection of the direct-FEM manufacturing control:
    exactly ``round(volfrac * n_active)`` elements become solid, everything
    outside the mask stays void.
    """
    batch = bounded.shape[0]
    mask_flat = mask2d.reshape(-1).to(device=bounded.device)
    active = bounded.reshape(batch, -1)[:, mask_flat]
    n_solid = int(round(volfrac * active.shape[1]))
    binary_active = torch.zeros_like(active)
    if n_solid > 0:
        indices = torch.topk(
            active, min(n_solid, active.shape[1]), dim=1).indices
        binary_active.scatter_(1, indices, 1.0)
    binary = torch.zeros_like(bounded).reshape(batch, -1)
    binary[:, mask_flat] = binary_active
    return binary.reshape_as(bounded)


class _ManufacturingCandidateView:
    """Delegating generator view used only inside the latent optimisation.

    ``decode`` keeps the continuous filtered path so the gradient-based
    multi-start search is unchanged; ``decode_with_diagnostics`` — used for
    the no-grad candidate ranking, the local candidate pool and the FEM
    hand-off — applies the manufacturing projection before the common filter
    and exact volume projection.
    """

    def __init__(self, wrapped: FilteredVolumeProjected):
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._wrapped.decode(z)

    def decode_with_diagnostics(self, z: torch.Tensor
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wrapped = self._wrapped
        raw = wrapped._crop(wrapped.decode_raw(z))
        bounded = raw.clamp(0.0, 1.0)
        binary = manufacturing_binarize(bounded, wrapped.mask2d, wrapped.vf)
        filtered = wrapped._filter(binary)
        projected = wrapped._project(filtered)
        batch = raw.shape[0]
        mask_flat = wrapped.mask2d.reshape(-1).to(device=raw.device)
        raw_active = raw.reshape(batch, -1)[:, mask_flat]
        filtered_active = filtered.reshape(batch, -1)[:, mask_flat]
        projected_active = projected.reshape(batch, -1)[:, mask_flat]
        diagnostics = {
            "raw_min": raw_active.min(dim=1).values,
            "raw_max": raw_active.max(dim=1).values,
            "clipped_fraction": (
                (raw_active < 0.0) | (raw_active > 1.0)).float().mean(dim=1),
            "filtered_min": filtered_active.min(dim=1).values,
            "filtered_max": filtered_active.max(dim=1).values,
            "projected_volume": projected_active.mean(dim=1),
            "bounded_raw_design": torch.where(
                wrapped.mask2d.to(device=bounded.device), bounded,
                torch.zeros_like(bounded)),
        }
        return projected, diagnostics


class _BinaryObjectiveCandidateView(_ManufacturingCandidateView):
    """Candidates are the volume-preserving binarisation of the filtered design.

    SIMP's penalisation implicitly optimises the binary structure; its
    continuous field is only the relaxation.  This view gives the challenger
    the same target: the gradient search still uses the continuous filtered
    decode, but every candidate that is ranked by the surrogate and validated
    by FEM is the exact-cardinality binarisation of the common-filtered,
    volume-projected design — the identical quantity reported for SIMP in the
    like-for-like B1 evaluation.
    """

    def decode_with_diagnostics(self, z: torch.Tensor
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wrapped = self._wrapped
        projected, diagnostics = wrapped.decode_with_diagnostics(z)
        binary = manufacturing_binarize(projected, wrapped.mask2d, wrapped.vf)
        batch = binary.shape[0]
        mask_flat = wrapped.mask2d.reshape(-1).to(device=binary.device)
        binary_active = binary.reshape(batch, -1)[:, mask_flat]
        diagnostics = {
            **diagnostics,
            "projected_volume": binary_active.mean(dim=1),
        }
        return binary, diagnostics


class ManufacturingHybridPISCOSolver(RevisionPISCOSolverTopology):
    """Adaptive hybrid PISCO whose FEM candidates are manufacturing-projected."""

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "manufacturing_hybrid_protocol": MANUFACTURING_HYBRID_PROTOCOL,
            "manufacturing_hybrid_solver_sha256": MANUFACTURING_SOLVER_SHA256,
        })
        return payload

    def _latent_optimize(self, generator, model, force, mask,
                         initial_latent=None):
        return super()._latent_optimize(
            _ManufacturingCandidateView(generator), model, force, mask,
            initial_latent=initial_latent)

    def _compute(self, *, seed: int):
        result = super()._compute(seed=seed)
        result.extra.update({
            "manufacturing_projection_candidates": True,
            "manufacturing_hybrid_protocol": MANUFACTURING_HYBRID_PROTOCOL,
            "candidate_filter_order": (
                "decoder_clip -> exact_cardinality_binary -> cone_filter "
                "-> volume_projection"),
        })
        return result


BINARY_OBJECTIVE_PROTOCOL = "surrogate_hybrid_binary_incumbent_v2"


class BinaryObjectiveHybridPISCOSolver(RevisionPISCOSolverTopology):
    """Hybrid PISCO that optimises the like-for-like binary benchmark metric.

    The FEM-validated objective equals the B1 evaluation applied to SIMP:
    common cone filter -> exact volume projection -> volume-preserving rank
    binarisation.  The FEM budget stays at ``n_top`` per active cycle.
    """

    binary_objective = True

    def __init__(self, problem, config, **kwargs):
        if config.T_cycles < 1 or config.n_top < 1:
            raise ValueError("Binary search requires at least one active FEM candidate")
        super().__init__(problem, config, **kwargs)

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "binary_objective_protocol": BINARY_OBJECTIVE_PROTOCOL,
            "binary_objective_solver_sha256": MANUFACTURING_SOLVER_SHA256,
        })
        return payload

    def _latent_optimize(self, generator, model, force, mask,
                         initial_latent=None):
        return super()._latent_optimize(
            _BinaryObjectiveCandidateView(generator), model, force, mask,
            initial_latent=initial_latent)

    def _compute(self, *, seed: int):
        result = super()._compute(seed=seed)
        result.extra.update({
            "binary_objective_candidates": True,
            "binary_objective_protocol": BINARY_OBJECTIVE_PROTOCOL,
            "candidate_filter_order": (
                "decoder_clip -> cone_filter -> volume_projection "
                "-> volume_preserving_rank_binarisation"),
        })
        return result


__all__ = [
    "BINARY_OBJECTIVE_PROTOCOL",
    "BinaryObjectiveHybridPISCOSolver",
    "MANUFACTURING_HYBRID_PROTOCOL",
    "ManufacturingHybridPISCOSolver",
    "manufacturing_binarize",
]


# =====================================================================
# 18. DYNAMIC-ANCHOR SOLVER
# =====================================================================

"""Dynamic-anchor extension of the binary-objective hybrid.

The frozen anchor PCA basis restricts which binary designs the generator can
express. This solver keeps the predeclared nested
schedule for the initial basis and, once that schedule is complete, appends
orthonormal residual directions of the FEM-validated binary candidates to
the mode bank ("dynamic anchors").  Appending never changes existing modes,
so zero-padded warm starts still decode to identical designs, and no
additional FEM evaluations are spent: the appended fields were already paid
for by the regular ``n_top`` validations.

Restore path: the dynamic bank is a deterministic function of the dataset
order, so on checkpoint resume it is replayed from the stored dataset
instead of being loaded from tensors.  This keeps fresh and resumed runs
bit-identical.
"""

import hashlib
from pathlib import Path
from typing import Any

import math

import numpy as np
import torch



DYNAMIC_SOLVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DYNAMIC_ANCHOR_PROTOCOL = "surrogate_hybrid_dynamic_anchor_v1"


class DynamicAnchorBinaryHybridSolver(BinaryObjectiveHybridPISCOSolver):
    """Binary-objective hybrid whose PCA mode bank grows with validated designs."""

    dynamic_mode_cap = 16
    residual_min_rel = 0.05

    def __init__(self, problem, config, **kwargs):
        if config.generator_type != "pca":
            raise ValueError("dynamic anchors require generator_type='pca'")
        super().__init__(problem, config, **kwargs)
        self._initial_rank: int | None = None
        self._dynamic_cursor: int | None = None
        self._current_dataset: list[dict] | None = None

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "dynamic_anchor_protocol": DYNAMIC_ANCHOR_PROTOCOL,
            "dynamic_anchor_solver_sha256": DYNAMIC_SOLVER_SHA256,
            "dynamic_mode_cap": int(self.dynamic_mode_cap),
            "residual_min_rel": float(self.residual_min_rel),
        })
        return payload

    # ---- deterministic mode bank -------------------------------------

    def _append_dynamic_modes(self, inner: torch.nn.Module,
                              dataset: list[dict]) -> int:
        """Append residual directions of new dataset entries to the bank.

        Returns the number of modes added in this call.  The bank grows even
        before the nested schedule completes; the active latent dimension is
        raised separately, so early growth never bypasses the predeclared
        unlocking order.
        """
        if self._dynamic_cursor is None:
            self._dynamic_cursor = min(len(dataset), int(self.config.n_init))
        added = 0
        mean = inner.mean.reshape(-1)
        for entry in dataset[self._dynamic_cursor:]:
            components = inner.components
            n_dynamic = components.shape[0] - int(self._initial_rank)
            if n_dynamic >= int(self.dynamic_mode_cap):
                break
            field = torch.tensor(
                np.asarray(entry["rho"], dtype=np.float32).reshape(-1))
            centred = field - mean
            residual = centred - components.T @ (components @ centred)
            norm = float(torch.linalg.vector_norm(residual))
            reference = float(torch.linalg.vector_norm(centred))
            if norm > self.residual_min_rel * max(reference, 1e-12):
                direction = (residual / norm).to(components.dtype)
                inner._buffers["components"] = torch.cat(
                    [components, direction[None, :]], dim=0)
                inner._buffers["scales"] = torch.cat([
                    inner.scales,
                    torch.tensor([norm], dtype=inner.scales.dtype)])
                added += 1
            self._dynamic_cursor += 1
        if added:
            inner.max_dz = int(inner.components.shape[0])
        return added

    # ---- solver hooks ------------------------------------------------

    def _generator_from_dataset(self, dataset, restored_state):
        # The PCA construction from the first n_init anchors is deterministic
        # and the dynamic bank is replayed from the stored dataset, so the
        # potentially larger restored tensors are never loaded directly.
        generator, info = super()._generator_from_dataset(dataset, None)
        self._initial_rank = int(generator.components.shape[0])
        self._dynamic_cursor = None
        if dataset is not None and len(dataset) > int(self.config.n_init):
            self._append_dynamic_modes(generator, dataset)
        generator.max_dz = int(generator.components.shape[0])
        return generator, info

    def _train_surrogate(self, model, dataset, force, free_indices, force_norm):
        self._current_dataset = dataset
        return super()._train_surrogate(
            model, dataset, force, free_indices, force_norm)

    def _latent_optimize(self, generator, model, force, mask,
                         initial_latent=None):
        inner = generator.inner
        if self._current_dataset is not None:
            self._append_dynamic_modes(inner, self._current_dataset)
            generator.max_dz = int(inner.max_dz)
            # Dynamic modes become searchable only after the predeclared
            # nested schedule has unlocked the full initial rank.
            if (self._initial_rank is not None
                    and generator.dz >= int(self._initial_rank)):
                generator.extend(int(inner.components.shape[0]))
        return super()._latent_optimize(
            generator, model, force, mask, initial_latent=initial_latent)

    def _compute(self, *, seed: int):
        result = super()._compute(seed=seed)
        n_dynamic = 0
        if self._initial_rank is not None:
            n_dynamic = int(result.extra["latent_dim_final"]) - int(
                self._initial_rank)
        result.extra.update({
            "dynamic_anchor_protocol": DYNAMIC_ANCHOR_PROTOCOL,
            "dynamic_anchor_modes_active": max(0, n_dynamic),
            "dynamic_anchor_initial_rank": self._initial_rank,
            "dynamic_mode_cap": int(self.dynamic_mode_cap),
            "residual_min_rel": float(self.residual_min_rel),
        })
        return result


GUARDED_DYNAMIC_ANCHOR_PROTOCOL = "surrogate_hybrid_dynamic_anchor_guarded_v1"


class GuardedDynamicAnchorSolver(DynamicAnchorBinaryHybridSolver):
    """Dynamic anchors reachable only through warm starts and gradients.

    The unguarded variant showed that sampling fresh multi-starts with full
    N(0,1) amplitude across the large-scale residual modes degrades candidate
    quality.  Here fresh random starts and the local candidate pool keep
    their dynamic coordinates fixed (zero, respectively the incumbent's
    values); only zero-padded warm starts and the gradient steps can activate
    the appended directions.  This mirrors the copied latent optimiser of
    ``RevisionPISCOSolverTopology`` with exactly that one guard added.
    """

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "guarded_dynamic_anchor_protocol":
                GUARDED_DYNAMIC_ANCHOR_PROTOCOL,
        })
        return payload

    def _latent_optimize(self, generator, model, force, mask,
                         initial_latent=None):
        inner = generator.inner
        if self._current_dataset is not None:
            self._append_dynamic_modes(inner, self._current_dataset)
            generator.max_dz = int(inner.max_dz)
            if (self._initial_rank is not None
                    and generator.dz >= int(self._initial_rank)):
                generator.extend(int(inner.components.shape[0]))
        view = _BinaryObjectiveCandidateView(generator)
        return self._guarded_latent_optimize(
            view, model, force, mask, initial_latent)

    def _guarded_latent_optimize(self, generator, model, force, mask,
                                 initial_latent=None):
        config = self.config
        problem = self.problem
        boundary = int(self._initial_rank or generator.dz)
        latent = torch.randn(config.M, generator.dz)
        n_warm = 0
        if initial_latent is not None:
            previous = np.asarray(initial_latent, dtype=np.float32)
            if previous.ndim != 2:
                raise ValueError(
                    "initial_latent must be a two-dimensional array")
            fraction = float(getattr(
                config, "adaptive_pca_warm_start_fraction", 0.0))
            n_warm = min(
                len(previous), config.M,
                int(math.ceil(config.M * fraction)))
            n_dimensions = min(previous.shape[1], generator.dz)
            if n_warm > 0 and n_dimensions > 0:
                with torch.no_grad():
                    latent[:n_warm, :n_dimensions] = torch.tensor(
                        previous[:n_warm, :n_dimensions])
                    if n_dimensions < generator.dz:
                        latent[:n_warm, n_dimensions:] = 0.0
        if generator.dz > boundary:
            # Guard: fresh exploration never samples the appended modes.
            with torch.no_grad():
                latent[n_warm:, boundary:] = 0.0
        latent.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=config.multistart_lr)
        for search_step in range(config.multistart_steps):
            _progress_step("candidate search", search_step, config.multistart_steps)
            optimizer.zero_grad()
            density = generator.decode(latent)[
                ..., :problem.nely, :problem.nelx]
            physical = density[:, 0] * mask
            displacement = model(physical.unsqueeze(1))
            predicted = displacement @ force
            prior = self._latent_prior(latent)
            (predicted + prior).sum().backward()
            optimizer.step()
            with torch.no_grad():
                latent.clamp_(-config.latent_clip, config.latent_clip)

        with torch.no_grad():
            density, diagnostics = generator.decode_with_diagnostics(latent)
            density = density[..., :problem.nely, :problem.nelx][:, 0]
            physical = torch.where(
                mask.bool(), density, torch.zeros_like(density))
            predicted_tensor = model(physical.unsqueeze(1)) @ force
            ranking = predicted_tensor + self._latent_prior(latent)
            predicted = ranking.numpy()
        diagnostics_numpy = {
            key: value.detach().cpu().numpy()
            for key, value in diagnostics.items()
        }

        pool_size = int(getattr(config, "local_candidate_pool_size", 0))
        if pool_size > 0:
            centre = np.zeros(generator.dz, dtype=np.float32)
            if initial_latent is not None and len(initial_latent):
                previous = np.asarray(initial_latent[0], dtype=np.float32)
                centre[:min(len(previous), generator.dz)] = previous[
                    :generator.dz]
            local = (torch.tensor(centre).reshape(1, -1)
                     + float(config.local_candidate_sigma)
                     * torch.randn(pool_size, generator.dz))
            local.clamp_(-config.latent_clip, config.latent_clip)
            local[0] = torch.tensor(centre)
            if generator.dz > boundary:
                # Guard: the pool perturbs only the predeclared modes.
                local[:, boundary:] = torch.tensor(centre[boundary:])
            with torch.no_grad():
                local_density, local_diagnostics = (
                    generator.decode_with_diagnostics(local))
                local_density = local_density[
                    ..., :problem.nely, :problem.nelx][:, 0]
                local_physical = torch.where(
                    mask.bool(), local_density,
                    torch.zeros_like(local_density))
                local_prediction = (
                    model(local_physical.unsqueeze(1)) @ force
                    + self._latent_prior(local))
            physical = torch.cat(
                [torch.tensor(physical.numpy()), local_physical], dim=0)
            predicted = np.concatenate([
                predicted, local_prediction.cpu().numpy()])
            latent_output = np.concatenate([
                latent.detach().cpu().numpy(), local.cpu().numpy()], axis=0)
            diagnostics_numpy = {
                key: np.concatenate([
                    diagnostics_numpy[key],
                    value.detach().cpu().numpy()], axis=0)
                for key, value in local_diagnostics.items()
            }
        else:
            latent_output = latent.detach().cpu().numpy()
        return (
            physical.numpy(), predicted, diagnostics_numpy,
            latent_output)

    def _compute(self, *, seed: int):
        result = super()._compute(seed=seed)
        result.extra["guarded_dynamic_anchor_protocol"] = (
            GUARDED_DYNAMIC_ANCHOR_PROTOCOL)
        return result


__all__ = [
    "DYNAMIC_ANCHOR_PROTOCOL",
    "DynamicAnchorBinaryHybridSolver",
    "GUARDED_DYNAMIC_ANCHOR_PROTOCOL",
    "GuardedDynamicAnchorSolver",
]


# =====================================================================
# 19. FILTERED DIRECT-FEM CONTROL
# =====================================================================

"""Filtered nested-PCA direct-FEM control for the major revision.

This is deliberately reported as a control, not as surrogate-based PISCO.
It isolates the useful part of the legacy coarse-to-fine experiment while
removing its protocol mismatch: every decoded continuous density passes the
same cone filter and exact volume projection as revised PISCO before FEM.
All high-fidelity evaluations are counted.
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch



FILTERED_CTF_SOLVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONTROL_PROTOCOL = "filtered_nested_pca_direct_fem_v1"


class FilteredNestedPCAFEMSolver(RevisionPISCOSolverTopology):
    """Gradient-free nested-PCA search against the filtered FEM objective."""

    def __init__(self, problem, config, **kwargs):
        if config.generator_type != "pca":
            raise ValueError("FilteredNestedPCAFEMSolver requires PCA")
        super().__init__(problem, config, **kwargs)

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "control_protocol": CONTROL_PROTOCOL,
            "control_solver_sha256": FILTERED_CTF_SOLVER_SHA256,
        })
        return payload

    def _evaluate_latent(self, generator, latent: np.ndarray
                         ) -> tuple[dict[str, Any], dict[str, float]]:
        problem = self.problem
        with torch.no_grad():
            density, diagnostics = generator.decode_with_diagnostics(
                torch.tensor(latent, dtype=torch.float32).reshape(1, -1))
        rho = density[0, 0, :problem.nely, :problem.nelx].numpy()
        displacement, compliance, _, element_stiffness = _fea(
            problem.nelx, problem.nely, rho, self._fixed, self._F_np,
            penal=problem.penal, E0=1.0, Emin=1e-9, nu=0.30)
        raw = diagnostics["bounded_raw_design"][
            0, 0, :problem.nely, :problem.nelx].numpy()
        entry = {
            "rho": rho.astype(np.float32),
            "generator_rho": raw.astype(np.float32),
            "raw_design": raw.astype(np.float32),
            "U": displacement.astype(np.float32),
            "compliance": float(compliance),
            "Eelem": element_stiffness.astype(np.float32),
            "decoder_raw_min": float(diagnostics["raw_min"][0]),
            "decoder_raw_max": float(diagnostics["raw_max"][0]),
            "decoder_clipped_fraction": float(
                diagnostics["clipped_fraction"][0]),
            "decoder_projected_volume": float(
                diagnostics["projected_volume"][0]),
        }
        diag = {
            "projected_volume": entry["decoder_projected_volume"],
            "clipped_fraction": entry["decoder_clipped_fraction"],
        }
        return entry, diag

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        config = self.config
        rng = np.random.default_rng(seed)
        checkpoint = load_checkpoint(self, seed) if self.cache_enabled else None
        resumed = checkpoint is not None

        if resumed:
            dataset = checkpoint["dataset"]
            fem_calls = int(checkpoint["fem_calls"])
            trajectory = list(checkpoint["trajectory"])
            inner, rebuilt_info = self._generator_from_dataset(
                dataset, checkpoint["generator_state"])
            generator_info = checkpoint.get("generator_info", rebuilt_info)
        else:
            dataset, fem_calls, trajectory = self._initial_dataset(seed)
            inner, generator_info = self._generator_from_dataset(dataset, None)

        generator = FilteredVolumeProjected(
            inner, self._mask, self.problem.volfrac,
            rmin=self.problem.rmin)
        schedule = self._adaptive_pca_schedule(generator)
        if not schedule:
            # A fixed six-stage schedule is still meaningful if the canonical
            # main-method switch was disabled explicitly for this control.
            maximum = min(int(config.ctf_max_dz), int(generator.max_dz))
            dimensions = np.rint(np.linspace(
                int(config.latent_dim), maximum,
                int(config.ctf_n_steps))).astype(int)
            schedule = [
                {"cycle": index + 1, "latent_dim": int(dimension)}
                for index, dimension in enumerate(dimensions)
            ]
        for parameter in generator.parameters():
            parameter.requires_grad_(False)

        if resumed:
            best = checkpoint["best"]
            search_best = checkpoint["search_best"]
            best_latent = np.asarray(
                checkpoint["best_latent"], dtype=np.float32)
            stage_index = int(checkpoint["stage_index"])
            iteration = int(checkpoint["iteration"])
            baseline_done = bool(checkpoint["baseline_done"])
            dimension_history = list(checkpoint["dimension_history"])
            projected_volumes = list(checkpoint["projected_volumes"])
            clipped_fractions = list(checkpoint["clipped_fractions"])
            elapsed_before = float(checkpoint.get("elapsed_s", 0.0))
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
        else:
            best = min(dataset, key=lambda sample: sample["compliance"])
            search_best = None
            best_latent = np.zeros(
                int(config.latent_dim), dtype=np.float32)
            stage_index = 0
            iteration = 0
            baseline_done = False
            dimension_history: list[dict[str, int]] = []
            projected_volumes: list[float] = []
            clipped_fractions: list[float] = []
            elapsed_before = 0.0

        def persist() -> None:
            if not self.cache_enabled:
                return
            save_checkpoint(self, seed, {
                "dataset": dataset,
                "fem_calls": int(fem_calls),
                "trajectory": list(trajectory),
                "generator_state": {
                    key: value.detach().cpu()
                    for key, value in generator.inner.state_dict().items()
                },
                "generator_info": generator_info,
                "best": best,
                "search_best": search_best,
                "best_latent": np.asarray(best_latent, dtype=np.float32),
                "stage_index": int(stage_index),
                "iteration": int(iteration),
                "baseline_done": bool(baseline_done),
                "active_latent_dim": int(generator.dz),
                "schedule": list(schedule),
                "dimension_history": list(dimension_history),
                "projected_volumes": list(projected_volumes),
                "clipped_fractions": list(clipped_fractions),
                "numpy_rng_state": rng.bit_generator.state,
                "elapsed_s": elapsed_before + time.perf_counter() - started,
            })

        if resumed:
            generator.extend(int(checkpoint.get(
                "active_latent_dim", config.latent_dim)))
        else:
            persist()

        while stage_index < len(schedule):
            target = int(schedule[stage_index]["latent_dim"])
            previous_dimension = int(generator.dz)
            generator.extend(target)
            if best_latent.size < generator.dz:
                best_latent = np.pad(
                    best_latent, (0, generator.dz - best_latent.size))
            if (not dimension_history
                    or dimension_history[-1]["stage"] != stage_index + 1):
                dimension_history.append({
                    "stage": stage_index + 1,
                    "latent_dim": int(generator.dz),
                    "expanded": int(generator.dz) > previous_dimension,
                })

            if not baseline_done:
                entry, diagnostics = self._evaluate_latent(
                    generator, best_latent)
                fem_calls += 1
                search_best = entry
                if entry["compliance"] < best["compliance"]:
                    best = entry
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))
                projected_volumes.append(diagnostics["projected_volume"])
                clipped_fractions.append(diagnostics["clipped_fraction"])
                baseline_done = True
                persist()

            while iteration < int(config.ctf_iters):
                candidate_latent = best_latent + float(
                    config.ctf_sigma) * rng.standard_normal(generator.dz)
                candidate_latent = np.clip(
                    candidate_latent, -config.latent_clip,
                    config.latent_clip).astype(np.float32)
                entry, diagnostics = self._evaluate_latent(
                    generator, candidate_latent)
                fem_calls += 1
                iteration += 1
                if entry["compliance"] < search_best["compliance"]:
                    search_best = entry
                    best_latent = candidate_latent
                if entry["compliance"] < best["compliance"]:
                    best = entry
                _append_trajectory(trajectory, (fem_calls, float(best["compliance"])))
                projected_volumes.append(diagnostics["projected_volume"])
                clipped_fractions.append(diagnostics["clipped_fraction"])
                persist()

            stage_index += 1
            iteration = 0
            persist()

        full_dimension = int(self._mask.sum())
        return SolverResult(
            best_compliance=float(best["compliance"]),
            best_design=np.asarray(best["rho"], dtype=np.float32),
            fem_calls_total=int(fem_calls), trajectory=list(trajectory),
            seed=int(seed), wall_time_s=float(
                elapsed_before + time.perf_counter() - started),
            extra={
                "revision_protocol": REVISION_PROTOCOL,
                "control_protocol": CONTROL_PROTOCOL,
                "ablation": "filtered_nested_pca_direct_fem",
                "uses_surrogate": False,
                "common_filter_in_optimisation": True,
                "filter_order": (
                    "decoder_clip -> cone_filter -> volume_projection -> FEM"),
                "generator_type": "pca",
                "adaptive_pca_schedule": list(schedule),
                "adaptive_pca_dimension_history": list(dimension_history),
                "latent_dim_final": int(generator.dz),
                "full_design_dim": full_dimension,
                "raw_design": np.asarray(
                    best.get("raw_design", best["rho"]), dtype=np.float32),
                "best_U": np.asarray(best["U"], dtype=np.float32),
                "best_Eelem": np.asarray(
                    best.get("Eelem"), dtype=np.float32)
                    if best.get("Eelem") is not None else None,
                "n_direct_search_fem": 1 + len(schedule) * int(
                    config.ctf_iters),
                "projected_volume_max_abs_error": float(np.max(np.abs(
                    np.asarray(projected_volumes) - self.problem.volfrac))),
                "clipped_fraction_mean": float(np.mean(clipped_fractions)),
                "resumed_from_checkpoint": bool(resumed),
            },
        )


__all__ = ["CONTROL_PROTOCOL", "FilteredNestedPCAFEMSolver"]


# =====================================================================
# 20. FILTERED BINARY DIRECT-FEM CONTROL
# =====================================================================

"""Manufacturing-oriented nested-PCA direct-FEM control.

Pipeline for every candidate:

    decoder clip -> exact-cardinality binary projection
    -> common cone filter -> exact volume projection -> FEM

The non-differentiable rank projection makes this a direct-FEM control, not
the surrogate-gradient PISCO main method.  It tests whether the strong legacy
coarse-to-fine numbers came from nested PCA or primarily from binarisation.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch



BINARY_CONTROL_PROTOCOL = "binary_then_filter_nested_pca_direct_fem_v1"
FILTERED_BINARY_SOLVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class FilteredBinaryNestedPCAFEMSolver(FilteredNestedPCAFEMSolver):
    """Nested PCA search against a filtered manufacturing projection."""

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "binary_control_protocol": BINARY_CONTROL_PROTOCOL,
            "binary_control_solver_sha256": FILTERED_BINARY_SOLVER_SHA256,
        })
        return payload

    def _evaluate_latent(self, generator, latent: np.ndarray
                         ) -> tuple[dict[str, Any], dict[str, float]]:
        problem = self.problem
        latent_tensor = torch.tensor(
            latent, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            raw = generator._crop(generator.decode_raw(latent_tensor))
            bounded = raw.clamp(0.0, 1.0)
            mask_flat = generator.mask2d.reshape(-1)
            active = bounded.reshape(1, -1)[:, mask_flat]
            n_solid = int(round(problem.volfrac * active.shape[1]))
            binary_active = torch.zeros_like(active)
            if n_solid > 0:
                indices = torch.topk(
                    active, min(n_solid, active.shape[1]), dim=1).indices
                binary_active.scatter_(1, indices, 1.0)
            binary = torch.zeros_like(bounded).reshape(1, -1)
            binary[:, mask_flat] = binary_active
            binary = binary.reshape_as(bounded)
            filtered = generator._filter(binary)
            density = generator._project(filtered)

        rho = density[0, 0, :problem.nely, :problem.nelx].numpy()
        displacement, compliance, _, element_stiffness = _fea(
            problem.nelx, problem.nely, rho, self._fixed, self._F_np,
            penal=problem.penal, E0=1.0, Emin=1e-9, nu=0.30)
        raw_active = raw.reshape(1, -1)[:, mask_flat]
        projected_active = density.reshape(1, -1)[:, mask_flat]
        clipped_fraction = float((
            (raw_active < 0.0) | (raw_active > 1.0)
        ).float().mean())
        projected_volume = float(projected_active.mean())
        binary_numpy = binary[0, 0].numpy().astype(np.float32)
        entry = {
            "rho": rho.astype(np.float32),
            "generator_rho": bounded[0, 0].numpy().astype(np.float32),
            "raw_design": binary_numpy,
            "U": displacement.astype(np.float32),
            "compliance": float(compliance),
            "Eelem": element_stiffness.astype(np.float32),
            "decoder_raw_min": float(raw_active.min()),
            "decoder_raw_max": float(raw_active.max()),
            "decoder_clipped_fraction": clipped_fraction,
            "decoder_projected_volume": projected_volume,
        }
        return entry, {
            "projected_volume": projected_volume,
            "clipped_fraction": clipped_fraction,
        }

    def _compute(self, *, seed: int) -> SolverResult:
        result = super()._compute(seed=seed)
        result.extra.update({
            "control_protocol": BINARY_CONTROL_PROTOCOL,
            "ablation": "filtered_binary_nested_pca_direct_fem",
            "filter_order": (
                "decoder_clip -> exact_cardinality_binary_projection -> "
                "cone_filter -> volume_projection -> FEM"),
            "manufacturing_projection": True,
        })
        return result


__all__ = [
    "BINARY_CONTROL_PROTOCOL", "FilteredBinaryNestedPCAFEMSolver"]


# =====================================================================
# 21. EXAMPLE 3 — FIN BENCHMARK WITHOUT MODULE-LEVEL MONKEYPATCHING
# =====================================================================

@dataclass(frozen=True)
class FinProblem:
    """Two-dimensional fin with a fixed skin and a designable interior.

    The geometry, pivot support and transverse load are the ones used for
    Example 3.  Keeping them in a problem object avoids the global monkeypatch
    used by the exploratory scripts from which the final benchmark evolved.
    """

    nelx: int = 60
    nely: int = 36
    length: float = 2.0
    height: float = 1.0
    taper: float = 0.12
    skin_elements: int = 2
    volfrac: float = 0.40
    penal: float = 3.0

    def geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, x_center, y_center = _structured_grid(
            self.nelx, self.nely, self.length, self.height)
        half_height = 0.5 * self.height * (
            1.0 - (1.0 - self.taper) * x_center / self.length)
        edge_distance = half_height - np.abs(y_center - 0.5 * self.height)
        domain = edge_distance >= 0.0
        skin_thickness = self.skin_elements * (self.height / self.nely)
        near_edge = edge_distance < skin_thickness
        near_tip = x_center > (
            self.length - self.skin_elements * self.length / self.nelx)
        skin = domain & (near_edge | near_tip)
        return domain, skin, domain & ~skin

    def load_and_support(self) -> tuple[np.ndarray, np.ndarray]:
        ndof, node_numbers, element_dofs, *_ = _prepare_fem(
            self.nelx, self.nely)
        _, skin, _ = self.geometry()
        _, _, x_center, _ = _structured_grid(
            self.nelx, self.nely, self.length, self.height)
        force = np.zeros(ndof)
        aerodynamic_center = 0.55 * self.length
        band = skin & (np.abs(x_center - aerodynamic_center)
                       < 0.05 * self.length)
        loaded_nodes: set[int] = set()
        rows, columns = np.where(band)
        for row, column in zip(rows, columns):
            element = column * self.nely + row
            loaded_nodes.update(int(node) for node in element_dofs[element] // 2)
        for node in sorted(loaded_nodes):
            force[2 * node + 1] -= 1.0 / max(1, len(loaded_nodes))

        x_nodes = np.linspace(0.0, self.length, self.nelx + 1)
        y_nodes = np.linspace(0.0, self.height, self.nely + 1)
        center_x, center_y = 0.13 * self.length, 0.5 * self.height
        radius = 0.09 * self.length
        fixed: list[int] = []
        for row, y_value in enumerate(y_nodes):
            for column, x_value in enumerate(x_nodes):
                if ((x_value - center_x) ** 2 + (y_value - center_y) ** 2
                        <= radius ** 2):
                    node = int(node_numbers[row, column])
                    fixed.extend((2 * node, 2 * node + 1))
        return force, np.unique(np.asarray(fixed, dtype=int))

    def physical_density(self, design_density: np.ndarray) -> np.ndarray:
        domain, skin, design = self.geometry()
        density = np.zeros((self.nely, self.nelx), dtype=float)
        supplied = np.asarray(design_density, dtype=float)
        if supplied.shape != density.shape:
            supplied = supplied.reshape(density.shape)
        density[design] = supplied[design]
        density[skin] = 1.0
        density[~domain] = 0.0
        return density

    def evaluate(self, design_density: np.ndarray
                 ) -> tuple[np.ndarray, float, np.ndarray]:
        force, fixed = self.load_and_support()
        physical = self.physical_density(design_density)
        displacement, compliance, _, _ = _fea(
            self.nelx, self.nely, physical, fixed, force,
            penal=self.penal, hx=self.length/self.nelx, hy=self.height/self.nely)
        return displacement, float(compliance), physical


class GaussianFinParameterization:
    """Map bounded control values to a volume-normalised fin density field."""

    GRID_BY_DIMENSION = {
        35: (11, 7),
        62: (15, 9),
        135: (21, 13),
        243: (29, 17),
    }

    def __init__(self, problem: FinProblem, dimension: int = 35):
        if dimension not in self.GRID_BY_DIMENSION:
            raise ValueError(
                f"Unsupported fin dimension {dimension}; choose "
                f"{sorted(self.GRID_BY_DIMENSION)}")
        self.problem = problem
        requested_x, requested_y = self.GRID_BY_DIMENSION[dimension]
        _, _, design = problem.geometry()
        rows, columns = np.where(design)
        pixels = np.stack([columns, rows], axis=1).astype(float)
        grid_x = np.linspace(2.0, problem.nelx - 2.5, requested_x)
        grid_y = np.linspace(3.0, problem.nely - 3.0, requested_y)
        controls = np.asarray([
            (x_value, y_value)
            for y_value in grid_y for x_value in grid_x
            if design[int(round(y_value)), int(round(x_value))]
        ], dtype=float)
        if len(controls) != dimension:
            raise RuntimeError(
                f"Fin control grid produced {len(controls)} instead of "
                f"the declared {dimension} variables")
        sigma = problem.nelx / float(requested_x)
        weights = np.exp(-np.sum(
            (pixels[:, None, :] - controls[None, :, :]) ** 2,
            axis=-1) / (2.0 * sigma ** 2))
        self.controls = controls
        self.weights = weights / weights.sum(axis=1, keepdims=True)
        self.dimension = len(controls)
        self._design = design
        self._weights_torch = torch.tensor(self.weights, dtype=torch.float32)
        self._design_indices_torch = torch.tensor(
            np.flatnonzero(design.reshape(-1)), dtype=torch.long)

    def raw_field(self, coordinates: np.ndarray) -> np.ndarray:
        field = np.zeros((self.problem.nely, self.problem.nelx), dtype=float)
        field[self._design] = self.weights @ np.asarray(coordinates, dtype=float)
        return field

    def project_volume(self, field: np.ndarray) -> np.ndarray:
        lower, upper = -1.0, 1.0
        active = np.asarray(field)[self._design]
        for _ in range(40):
            offset = 0.5 * (lower + upper)
            current = np.clip(active + offset, 0.0, 1.0).mean()
            if current > self.problem.volfrac:
                upper = offset
            else:
                lower = offset
        result = np.zeros_like(field, dtype=float)
        result[self._design] = np.clip(
            active + 0.5 * (lower + upper), 0.0, 1.0)
        return result

    def decode(self, coordinates: np.ndarray) -> np.ndarray:
        return self.project_volume(self.raw_field(coordinates))

    def decode_torch(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = coordinates @ self._weights_torch.to(coordinates).t()
        field = torch.zeros(coordinates.shape[0], self.problem.nely*self.problem.nelx,
                            dtype=coordinates.dtype, device=coordinates.device)
        field = field.index_copy(1, self._design_indices_torch.to(coordinates.device), values)
        field = field.view(coordinates.shape[0], self.problem.nely, self.problem.nelx)
        return project_volume_torch(field, self._design, self.problem.volfrac)

    def lift(self, field: np.ndarray) -> np.ndarray:
        """Sample a density field at the control locations for warm starts."""
        return np.clip(np.asarray([
            field[int(round(y_value)), int(round(x_value))]
            for x_value, y_value in self.controls
        ]), 0.0, 1.0)


@dataclass(frozen=True)
class FinSearchConfig:
    dimension: int = 35
    budget: int = 150
    n_init: int = 20
    n_top: int = 6
    surrogate_steps: int = 150
    multistart_count: int = 12
    multistart_steps: int = 120
    multistart_lr: float = 0.05
    lam_phys: float = 0.01
    progressive_dimensions: tuple[int, ...] = ()
    progressive_switch_cycles: tuple[int, ...] = (4, 8, 12)


class FinPISCOSolver(BaseSolver):
    """Field-surrogate PISCO used for the fixed and progressive fin searches."""

    IMPLEMENTATION_PROTOCOL = "fin_field_rectangular_exact_projection_v5"
    TORCH_THREADS = 1
    DETERMINISTIC_ALGORITHMS = True

    def __init__(self, problem: FinProblem, config: FinSearchConfig, **kwargs):
        super().__init__(problem, config, **kwargs)

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload["implementation_protocol"] = self.IMPLEMENTATION_PROTOCOL
        payload["torch_threads"] = self.TORCH_THREADS
        payload["deterministic_algorithms"] = self.DETERMINISTIC_ALGORITHMS
        payload["mkldnn_enabled"] = False
        return payload

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        torch.set_num_threads(self.TORCH_THREADS)
        torch.use_deterministic_algorithms(self.DETERMINISTIC_ALGORITHMS)
        torch.backends.mkldnn.enabled = False
        rng = self._seed_everything(seed)
        problem, config = self.problem, self.config
        dimensions = (config.progressive_dimensions
                      or (config.dimension,))
        parameterizations = [
            GaussianFinParameterization(problem, dimension)
            for dimension in dimensions]
        current_stage = 0
        parameterization = parameterizations[0]
        force, fixed = problem.load_and_support()
        force_torch = torch.tensor(force, dtype=torch.float32)
        free = torch.tensor(
            np.setdiff1d(np.arange(force.size), fixed), dtype=torch.long)
        force_norm = float(np.linalg.norm(force))
        dataset: list[dict[str, Any]] = []
        trajectory: list[tuple[int, float]] = []
        best = math.inf
        best_coordinates: np.ndarray | None = None
        best_field: np.ndarray | None = None

        def evaluate(coordinates: np.ndarray) -> None:
            nonlocal best, best_coordinates, best_field
            field = parameterization.decode(coordinates)
            displacement, compliance, physical = problem.evaluate(field)
            dataset.append({
                "rho": field.astype(np.float32),
                "physical_rho": physical.astype(np.float32),
                "U": displacement.astype(np.float32),
                "compliance": compliance,
                "coordinates": np.asarray(coordinates, dtype=np.float32),
            })
            if compliance < best:
                best = compliance
                best_coordinates = np.asarray(coordinates).copy()
                best_field = field.copy()
            _append_trajectory(trajectory, (len(dataset), float(best)))

        with audit_scope("initial_data"):
            for _ in range(config.n_init):
                evaluate(rng.uniform(0.0, 1.0, parameterization.dimension))

        displacement_scale = float(np.std([
            np.linalg.norm(item["U"]) for item in dataset
        ]) / np.sqrt(force.size))
        model = _make_field_surrogate_cnn(
            force.size, displacement_scale, fixed_dofs=fixed)
        cycle = 0
        dimension_history: list[int] = []
        while len(dataset) < config.budget:
            cycle += 1
            _progress_cycle(cycle, math.ceil((config.budget-config.n_init)/config.n_top))
            while (current_stage < len(parameterizations) - 1
                   and current_stage < len(config.progressive_switch_cycles)
                   and cycle > config.progressive_switch_cycles[current_stage]):
                current_stage += 1
                parameterization = parameterizations[current_stage]
            dimension_history.append(parameterization.dimension)

            with audit_scope("training_preparation"):
                density_batch = torch.tensor(np.stack([item["rho"] for item in dataset])[:,None], dtype=torch.float32)
                compliance_batch = torch.tensor([item["compliance"] for item in dataset], dtype=torch.float32)
                stiffness = ([_training_stiffness(problem.nelx, problem.nely, item["physical_rho"], problem.penal,
                               hx=problem.length/problem.nelx, hy=problem.height/problem.nely)
                              for item in dataset] if config.lam_phys else [])
            with audit_scope("surrogate_training"):
                training_report = _fit_field(model, density_batch, compliance_batch, force_torch,
                    stiffness, free, force_norm, steps=config.surrogate_steps, lr=2e-3,
                    lam_data=1.0, lam_phys=config.lam_phys, physics_batch=min(20,len(dataset)))
            audit = _ACTIVE_AUDIT.get()
            if audit is not None:
                audit.training.append({"update":cycle, **training_report})

            if best_field is None:
                raise RuntimeError("Fin search has no incumbent after initialisation")
            # Re-centre one local search at the control vector reconstructed
            # from the best solver-verified density field.  The same rule is
            # used on fixed grids and transfers the incumbent when the
            # progressive search changes resolution.
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with audit_scope("candidate_search"):
                warm_start = parameterization.lift(best_field)
                starts = [warm_start] + [
                    rng.uniform(0.0, 1.0, parameterization.dimension)
                    for _ in range(config.multistart_count - 1)]
                def objective(coordinates):
                    field = parameterization.decode_torch(coordinates.clamp(0.0, 1.0))
                    return model(field.unsqueeze(1)) @ force_torch
                coordinates_torch, prediction_tensor = _packet_adam(
                    torch.tensor(np.vstack(starts), dtype=torch.float32), objective,
                    steps=config.multistart_steps, lr=config.multistart_lr,
                    project=lambda value: value.clamp(0.0, 1.0))
                predicted = prediction_tensor.numpy()
                coordinates = coordinates_torch.numpy()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            for index in np.argsort(predicted)[:config.n_top]:
                if len(dataset) >= config.budget:
                    break
                evaluate(np.clip(coordinates[index], 0.0, 1.0))
                _record_prediction(predicted[index], dataset[-1]["compliance"], cycle, dataset[-1]["rho"])

        with audit_scope("independent_test"):
            test_rng = np.random.default_rng(seed + 900_001)
            audit = _ACTIVE_AUDIT.get()
            for _ in range(self.training_policy.independent_test_size):
                coordinates = test_rng.uniform(0.0, 1.0, parameterization.dimension)
                field = parameterization.decode(coordinates)
                tensor = torch.tensor(field[None,None], dtype=torch.float32)
                with torch.no_grad():
                    prediction = float((model(tensor) @ force_torch)[0])
                _, actual, physical = problem.evaluate(field)
                if audit is not None:
                    audit.test_predictions.append({"predicted":prediction,"actual":float(actual),
                        "representation":"continuous_fin", "dimension":parameterization.dimension,
                        "sampling":"independent uniform coordinates on final parameterization", "coordinates":coordinates.tolist()})
        return SolverResult(
            best_compliance=float(best),
            best_design=problem.physical_density(best_field),
            fem_calls_total=len(dataset), trajectory=trajectory, seed=seed,
            wall_time_s=time.perf_counter() - started,
            extra={
                "protocol": PUBLICATION_PROTOCOL,
                "benchmark": "fin",
                "dimension": config.dimension,
                "dimension_history": dimension_history,
                "best_coordinates": best_coordinates,
                "lam_phys": config.lam_phys,
            })


class _FinDirectSolver(BaseSolver):
    """Base for budget-matched direct-search controls on the fin."""

    def __init__(self, problem: FinProblem, config: FinSearchConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self._parameterization = GaussianFinParameterization(
            problem, config.dimension)

    def _evaluate(self, coordinates: np.ndarray) -> tuple[float, np.ndarray]:
        field = self._parameterization.decode(coordinates)
        _, compliance, physical = self.problem.evaluate(field)
        return compliance, physical

    def _result(self, seed: int, started: float, values: list[float],
                designs: list[np.ndarray], extra: dict[str, Any]
                ) -> SolverResult:
        running = np.minimum.accumulate(values)
        index = int(np.argmin(values))
        return SolverResult(
            best_compliance=float(values[index]), best_design=designs[index],
            fem_calls_total=len(values),
            trajectory=[(call, float(value)) for call, value in
                        enumerate(running, start=1)],
            seed=seed, wall_time_s=time.perf_counter() - started,
            extra={"protocol": PUBLICATION_PROTOCOL, "benchmark": "fin",
                   "dimension": self.config.dimension, **extra})


class FinRandomSolver(_FinDirectSolver):
    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        rng = self._seed_everything(seed)
        values, designs = [], []
        for _ in range(self.config.budget):
            value, design = self._evaluate(
                rng.uniform(0.0, 1.0, self.config.dimension))
            values.append(value)
            designs.append(design)
        return self._result(seed, started, values, designs,
                            {"method": "random_search"})


class FinDESolver(_FinDirectSolver):
    def _compute(self, *, seed: int) -> SolverResult:
        from scipy.optimize import differential_evolution

        started = time.perf_counter()
        values: list[float] = []
        designs: list[np.ndarray] = []

        def objective(coordinates):
            if len(values) >= self.config.budget:
                return min(values) + 1e6
            value, design = self._evaluate(coordinates)
            values.append(value)
            designs.append(design)
            return value

        differential_evolution(
            objective, [(0.0, 1.0)] * self.config.dimension,
            seed=seed, maxiter=30, popsize=4, tol=-1.0,
            polish=False, init="latinhypercube")
        return self._result(seed, started, values, designs,
                            {"method": "differential_evolution"})


class FinCMAESSolver(_FinDirectSolver):
    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        rng = self._seed_everything(seed)
        dimension = self.config.dimension
        population = 4 + int(3 * math.log(dimension))
        parents = population // 2
        weights = np.log(parents + 0.5) - np.log(np.arange(1, parents + 1))
        weights /= weights.sum()
        effective = 1.0 / float(np.sum(weights ** 2))
        cs = (effective + 2.0) / (dimension + effective + 5.0)
        damping = (1.0 + 2.0 * max(
            0.0, math.sqrt((effective - 1.0) / (dimension + 1.0)) - 1.0)
            + cs)
        cc = ((4.0 + effective / dimension)
              / (dimension + 4.0 + 2.0 * effective / dimension))
        c1 = 2.0 / ((dimension + 1.3) ** 2 + effective)
        cmu = min(1.0 - c1, 2.0 * (effective - 2.0 + 1.0 / effective)
                  / ((dimension + 2.0) ** 2 + effective))
        expected_norm = math.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension)
            + 1.0 / (21.0 * dimension ** 2))
        mean = np.full(dimension, 0.5)
        sigma = 0.3
        covariance = np.eye(dimension)
        path_sigma = np.zeros(dimension)
        path_covariance = np.zeros(dimension)
        values: list[float] = []
        designs: list[np.ndarray] = []
        generation = 0
        while len(values) < self.config.budget:
            generation += 1
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            count = min(population, self.config.budget - len(values))
            normal = rng.standard_normal((count, dimension))
            candidates = np.clip(
                mean + sigma * (normal @ transform.T), 0.0, 1.0)
            fitness = []
            for candidate in candidates:
                value, design = self._evaluate(candidate)
                values.append(value)
                designs.append(design)
                fitness.append(value)
            if count < parents:
                break
            selected = candidates[np.argsort(fitness)[:parents]]
            previous_mean = mean
            mean = weights @ selected
            step = (mean - previous_mean) / sigma
            inverse_transform = (eigenvectors
                @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T)
            path_sigma = ((1.0 - cs) * path_sigma
                + math.sqrt(cs * (2.0 - cs) * effective)
                * (inverse_transform @ step))
            signal = float(
                np.linalg.norm(path_sigma)
                / math.sqrt(1.0 - (1.0 - cs) ** (2 * generation))
                < (1.4 + 2.0 / (dimension + 1.0)) * expected_norm)
            path_covariance = ((1.0 - cc) * path_covariance
                + signal * math.sqrt(cc * (2.0 - cc) * effective) * step)
            normalised = (selected - previous_mean) / sigma
            covariance = ((1.0 - c1 - cmu) * covariance
                + c1 * (np.outer(path_covariance, path_covariance)
                        + (1.0 - signal) * cc * (2.0 - cc) * covariance)
                + cmu * normalised.T @ np.diag(weights) @ normalised)
            sigma *= math.exp(
                (cs / damping)
                * (np.linalg.norm(path_sigma) / expected_norm - 1.0))
            sigma = float(np.clip(sigma, 1e-8, 1.0))
        return self._result(seed, started, values, designs,
                            {"method": "cma_es"})


class FinARDSolver(_FinDirectSolver):
    """Gaussian-process Bayesian optimisation with an ARD Matérn kernel."""

    def _compute(self, *, seed: int) -> SolverResult:
        import warnings
        from scipy.stats import norm as gaussian
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel, Matern, WhiteKernel)

        started = time.perf_counter()
        rng = self._seed_everything(seed)
        coordinates = list(rng.uniform(
            0.0, 1.0, (self.config.n_init, self.config.dimension)))
        values, designs = [], []
        for candidate in coordinates:
            value, design = self._evaluate(candidate)
            values.append(value)
            designs.append(design)
        while len(values) < self.config.budget:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(self.config.dimension),
                         length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(1e-6, (1e-10, 1e-1)))
            model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                random_state=int(rng.integers(2 ** 31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                with audit_scope("surrogate_training"):
                    model.fit(np.asarray(coordinates), np.asarray(values))
            incumbent_value = float(np.min(values))
            incumbent = np.asarray(coordinates)[int(np.argmin(values))]
            candidates = np.vstack([
                rng.uniform(0.0, 1.0, (2048, self.config.dimension)),
                np.clip(incumbent + 0.15 * rng.standard_normal(
                    (2048, self.config.dimension)), 0.0, 1.0)])
            mean, std = model.predict(candidates, return_std=True)
            std = np.maximum(std, 1e-12)
            improvement = (incumbent_value - mean) / std
            expected = std * (
                improvement * gaussian.cdf(improvement)
                + gaussian.pdf(improvement))
            choice = candidates[int(np.argmax(expected))]
            value, design = self._evaluate(choice)
            coordinates.append(choice)
            values.append(value)
            designs.append(design)
        return self._result(seed, started, values, designs,
                            {"method": "ard_bayesian_optimisation"})


# =====================================================================
# 22. PUBLIC EXPERIMENT SUITE
# =====================================================================

class PublicationExperimentSuite:
    """Direct, resumable access to every numerical benchmark family.

    This class replaces the former collection of command-line fragments.
    It returns ``SolverResult`` objects and never formats tables or figures;
    that publication-specific work belongs to ``PublicationBuilder``.
    """

    def __init__(self, output_root: Path | str = "reproduction_results", training_policy=None):
        self.output_root = Path(output_root)
        self.store = ResultStore(self.output_root / "runs", training_policy)

    def smoke(self):
        """Small real solves on publication meshes; never publication results."""
        e1 = dataclasses.replace(make_publication_pisco_e1_config(), n_init=4,
            T_cycles=2, n_top=2, M=2, M_explore=4, multistart_steps=3,
            surrogate_steps=3, surrogate_hidden=8, surrogate_depth=1)
        e2 = dataclasses.replace(make_publication_pisco_e2_config(), n_init=4,
            short_simp_iters=2, T_cycles=2, n_top=2, M=2, multistart_steps=3,
            surrogate_steps=3, latent_dim=2, adaptive_pca_max_dim=3,
            adaptive_pca_n_stages=1, adaptive_pca_expand_cycles=(1,),
            local_candidate_pool_size=4, diagnostic_holdout_size=4)
        fin = FinSearchConfig(n_init=4, budget=8, n_top=2, surrogate_steps=3,
                              multistart_count=2, multistart_steps=3)
        ellipse = make_publication_ellipse_problem()
        topology = make_publication_lbracket_problem()
        results = {"example1": {}, "example2": {}, "example3": {}, "mesh_refinement": {}}
        results["example1"]["pisco"] = self.store.solve_many("example1", "pisco", (7,8),
            lambda: PISCOSolver(ellipse, e1, generator=AnalyticEllipseGenerator(
                ellipse, make_publication_ellipse_box())))
        for name, cls, config in (("pisco_binary", BinaryObjectiveHybridPISCOSolver, e2),
                ("field_no_physics_matched", RevisionPISCOSolverTopology, dataclasses.replace(e2,lam_phys=0.)),
                ("scalar_surrogate", MatchedScalarSurrogateSolver, e2)):
            results["example2"][name] = self.store.solve_many("example2", name, (7,8),
                lambda cls=cls, config=config: cls(topology, config))
        simp = dataclasses.replace(make_publication_baseline_configs()[3], maxiter=3)
        results["example2"]["simp"] = [self.store.solve("example2", "simp", SIMPSolver(topology,simp),0)]
        for name, cls in (("pisco_d35", FinPISCOSolver), ("scalar_surrogate_d35", FinScalarPISCOSolver)):
            results["example3"][name] = self.store.solve_many("example3", name, (0,1),
                lambda cls=cls: cls(FinProblem(), fin))
        mask = _l_bracket_mask(topology.nelx, topology.nely)
        for name in ("pisco_binary", "simp"):
            results["mesh_refinement"][name] = []
            for source in results["example2"][name]:
                density = source.best_design
                if name == "simp":
                    density = common_filtered_design(source.extra["raw_design"], topology, mask)
                    density = volume_preserving_binary(density, mask, topology.volfrac)
                elif not np.all((density == 0.) | (density == 1.)):
                    raise ValueError("Binary solver returned a continuous design")
                solver = BinaryMeshRefinementSolver(topology,
                    MeshRefinementConfig(source_method=name, source_seed=source.seed), density)
                results["mesh_refinement"][name].append(self.store.solve("mesh_refinement",name,solver,source.seed))
        return results

    def example1(self, seeds: Any = PUBLICATION_SEEDS
                 ) -> dict[str, list[SolverResult]]:
        problem = make_publication_ellipse_problem()
        box = make_publication_ellipse_box()
        de_config, bo_config, cma_config, _ = make_publication_baseline_configs()
        results = {
            "pisco": self.store.solve_many(
                "example1", "pisco", seeds,
                lambda: PISCOSolver(
                    problem, make_publication_pisco_e1_config(),
                    generator=AnalyticEllipseGenerator(problem, box))),
            "de": self.store.solve_many(
                "example1", "de", seeds,
                lambda: DESolver(problem, de_config)),
            "bo": self.store.solve_many(
                "example1", "bo", seeds,
                lambda: BOSolver(problem, bo_config)),
            "cmaes": self.store.solve_many(
                "example1", "cmaes", seeds,
                lambda: CMAESSolver(problem, cma_config)),
        }
        for weight in (0.0, 0.001, 0.1):
            method = f"pisco_lambda_{weight:g}"
            results[method] = self.store.solve_many(
                "example1", method, seeds,
                lambda weight=weight: PISCOSolver(
                    problem,
                    make_publication_pisco_e1_config(lam_phys=weight),
                    generator=AnalyticEllipseGenerator(problem, box)))
        fixed_budget = 156
        results["pisco_fixed_156"] = self.store.solve_many(
            "example1", "pisco_fixed_156", seeds,
            lambda: PISCOSolver(
                problem, make_publication_pisco_e1_config(
                    convergence_patience=None),
                generator=AnalyticEllipseGenerator(problem, box)))
        results["bo_fixed_156"] = self.store.solve_many(
            "example1", "bo_fixed_156", seeds,
            lambda: BOSolver(
                problem, dataclasses.replace(
                    bo_config, convergence_patience=None)))
        de_fixed_raw = self.store.solve_many(
            "example1", "de_fixed_raw_160", seeds,
            lambda: DESolver(
                problem, dataclasses.replace(
                    de_config, maxiter=3, polish=False, tol=0.0, atol=0.0)))
        cma_fixed_raw = self.store.solve_many(
            "example1", "cmaes_fixed_raw_160", seeds,
            lambda: CMAESSolver(
                problem, dataclasses.replace(
                    cma_config, n_generations=20,
                    convergence_patience=None)))
        results["de_fixed_156"] = [
            result_at_fem_checkpoint(result, fixed_budget)
            for result in de_fixed_raw]
        results["cmaes_fixed_156"] = [
            result_at_fem_checkpoint(result, fixed_budget)
            for result in cma_fixed_raw]
        return results

    def example2(self, seeds: Any = PUBLICATION_SEEDS,
                 *, include_controls: bool = True
                 ) -> dict[str, list[SolverResult]]:
        problem = make_publication_lbracket_problem()
        _, _, _, simp_config = make_publication_baseline_configs()
        dynamic = make_publication_pisco_e2_config()
        static = dataclasses.replace(
            dynamic, adaptive_pca_enabled=False)
        dynamic_fixed = dataclasses.replace(
            dynamic, convergence_patience=None)
        static_fixed = dataclasses.replace(
            static, convergence_patience=None)
        configurations: dict[str, Callable[[], BaseSolver]] = {
            "pisco_binary": lambda: BinaryObjectiveHybridPISCOSolver(
                problem, dynamic),
            "pisco_strong_anchors":
                lambda: BinaryObjectiveHybridPISCOSolver(
                    problem, dataclasses.replace(
                        dynamic, short_simp_iters=20)),
            "pisco_dynamic_pca": lambda: RevisionPISCOSolverTopology(
                problem, dynamic),
            "pisco_static_pca": lambda: RevisionPISCOSolverTopology(
                problem, static),
            "pisco_dynamic_pca_fixed_cycles":
                lambda: RevisionPISCOSolverTopology(
                    problem, dynamic_fixed),
            "pisco_static_pca_fixed_cycles":
                lambda: RevisionPISCOSolverTopology(
                    problem, static_fixed),
            "pisco_vae": lambda: RevisionPISCOSolverTopology(
                problem, make_publication_pisco_e2_config(
                    generator_type="vae")),
            "pisco_ae": lambda: RevisionPISCOSolverTopology(
                problem, make_publication_pisco_e2_config(
                    generator_type="ae")),
            "pisco_vae_lambda_0": lambda: RevisionPISCOSolverTopology(
                problem, make_publication_pisco_e2_config(
                    generator_type="vae", lam_phys=0.0)),
            "pisco_vae_lambda_1": lambda: RevisionPISCOSolverTopology(
                problem, make_publication_pisco_e2_config(
                    generator_type="vae", lam_phys=1.0)),
            "pisco_vae_data_lambda_0":
                lambda: RevisionPISCOSolverTopology(
                    problem, make_publication_pisco_e2_config(
                        generator_type="vae", lam_data=0.0,
                        lam_phys=0.01)),
        }
        if include_controls:
            configurations.update({
                "pisco_random_basis":
                    lambda: RevisionPISCOSolverTopology(
                        problem, dataclasses.replace(
                            dynamic, simp_free_anchors=True)),
                "scalar_surrogate": lambda: MatchedScalarSurrogateSolver(
                    problem, dynamic_fixed),
                "field_no_physics_matched": lambda: RevisionPISCOSolverTopology(
                    problem, dataclasses.replace(dynamic_fixed, lam_phys=0.0)),
                "no_periodic_verification": lambda: NoRevalSolver(
                    problem, dynamic),
                "latent_fem": lambda: LatentFEMSolver(problem, dynamic),
                "latent_cmaes": lambda: LatentCMAESBinaryBaseline(
                    problem, static),
                "latent_ard_bo": lambda: LatentBOBinaryBaseline(
                    problem, static),
                "filtered_direct_fem": lambda: FilteredNestedPCAFEMSolver(
                    problem, dynamic),
                "filtered_binary_direct_fem":
                    lambda: FilteredBinaryNestedPCAFEMSolver(problem, dynamic),
            })
        results = {
            method: self.store.solve_many(
                "example2", method, seeds, factory)
            for method, factory in configurations.items()
        }
        results["simp"] = [self.store.solve(
            "example2", "simp", SIMPSolver(problem, simp_config), 0)]
        if include_controls:
            results["simp_multistart_30"] = [self.store.solve(
                "example2", "simp_multistart_30",
                SIMPSolver(problem, dataclasses.replace(
                    simp_config, n_starts=30)), 0)]
        results["pca_projection_study"] = [self.store.solve(
            "example2", "pca_projection_study",
            Example2ProjectionStudySolver(
                problem, E2ProjectionStudyConfig()), 7)]
        return results

    def example3(self, seeds: Any = FIN_SEEDS,
                 *, dimensions: Any = (35, 62, 135, 243),
                 include_controls: bool = True
                 ) -> dict[str, list[SolverResult]]:
        problem = FinProblem()
        results: dict[str, list[SolverResult]] = {}
        for dimension in [int(value) for value in dimensions]:
            config = FinSearchConfig(dimension=dimension)
            results[f"pisco_d{dimension}"] = self.store.solve_many(
                "example3", f"pisco_d{dimension}", seeds,
                lambda config=config: FinPISCOSolver(problem, config))
            if include_controls:
                for method, solver_class in (
                        ("random", FinRandomSolver),
                        ("de", FinDESolver),
                        ("cmaes", FinCMAESSolver),
                        ("ard_bo", FinARDSolver)):
                    key = f"{method}_d{dimension}"
                    results[key] = self.store.solve_many(
                        "example3", key, seeds,
                        lambda solver_class=solver_class, config=config:
                            solver_class(problem, config))
        progressive = FinSearchConfig(
            dimension=35,
            progressive_dimensions=(35, 62, 135, 243))
        results["pisco_progressive"] = self.store.solve_many(
            "example3", "pisco_progressive", seeds,
            lambda: FinPISCOSolver(problem, progressive))
        if include_controls:
            results["pisco_no_physics_d35"] = self.store.solve_many(
                "example3", "pisco_no_physics_d35", seeds,
                lambda: FinPISCOSolver(
                    problem, FinSearchConfig(dimension=35, lam_phys=0.0)))
            results["scalar_surrogate_d35"] = self.store.solve_many(
                "example3", "scalar_surrogate_d35", seeds,
                lambda: FinScalarPISCOSolver(
                    problem, FinSearchConfig(dimension=35)))
            results["ard_bo_progressive"] = self.store.solve_many(
                "example3", "ard_bo_progressive", seeds,
                lambda: FinProgressiveARDSolver(
                    problem, FinSearchConfig(dimension=35)))
            sensitivity_seeds = [int(seed) for seed in seeds][:3]
            main_by_seed = {
                int(result.seed): result for result in results["pisco_d35"]}
            results["pisco_starts_12_d35"] = [
                main_by_seed[seed] for seed in sensitivity_seeds]
            results["pisco_top_6_d35"] = [
                main_by_seed[seed] for seed in sensitivity_seeds]
            for starts in (6, 24):
                key = f"pisco_starts_{starts}_d35"
                results[key] = self.store.solve_many(
                    "example3", key, sensitivity_seeds,
                    lambda starts=starts: FinPISCOSolver(
                        problem, FinSearchConfig(
                            dimension=35, multistart_count=starts)))
            for top in (3, 12):
                key = f"pisco_top_{top}_d35"
                results[key] = self.store.solve_many(
                    "example3", key, sensitivity_seeds,
                    lambda top=top: FinPISCOSolver(
                        problem, FinSearchConfig(
                            dimension=35, n_top=top)))
        return results

    def run(self, example: str = "all", *, seeds: Any | None = None,
            include_controls: bool = True
            ) -> dict[str, dict[str, list[SolverResult]]]:
        selected = ("example1", "example2", "example3") \
            if example == "all" else (example,)
        result: dict[str, dict[str, list[SolverResult]]] = {}
        for name in selected:
            if name == "example1":
                result[name] = self.example1(
                    PUBLICATION_SEEDS if seeds is None else seeds)
            elif name == "example2":
                result[name] = self.example2(
                    PUBLICATION_SEEDS if seeds is None else seeds,
                    include_controls=include_controls)
                problem = make_publication_lbracket_problem()
                mask = _l_bracket_mask(problem.nelx, problem.nely)
                refinement = {}
                for method in ("pisco_binary", "simp"):
                    refinement[method] = []
                    for source in result[name][method]:
                        density = np.asarray(source.best_design)
                        if method == "simp":
                            density = volume_preserving_binary(common_filtered_design(
                                source.extra["raw_design"], problem, mask), mask, problem.volfrac)
                        elif not np.all((density == 0.) | (density == 1.)):
                            raise ValueError("Binary solver returned a continuous design")
                        solver = BinaryMeshRefinementSolver(problem,
                            MeshRefinementConfig(source_method=method, source_seed=source.seed), density)
                        refinement[method].append(self.store.solve("mesh_refinement", method, solver, source.seed))
                result["mesh_refinement"] = refinement
            elif name == "example3":
                result[name] = self.example3(
                    FIN_SEEDS if seeds is None else seeds,
                    include_controls=include_controls)
            else:
                raise ValueError(
                    "example must be 'all', 'example1', 'example2' or "
                    "'example3'")
        return result


# The concise public names refer to the revised topology protocol.  The
# submitted implementation remains accessible explicitly for audit purposes.
OriginalPISCOConfigTopology = PISCOConfigTopology
OriginalPISCOSolverTopology = PISCOSolverTopology
PISCOConfigTopology = RevisionPISCOConfigTopology
PISCOSolverTopology = RevisionPISCOSolverTopology

__all__ = sorted(set(globals().get("__all__", [])) | {
    "PUBLICATION_PROTOCOL", "PUBLICATION_SEEDS", "FIN_SEEDS",
    "ResultStore", "PublicationExperimentSuite",
    "make_publication_ellipse_problem", "make_publication_ellipse_box",
    "make_publication_lbracket_problem",
    "make_publication_pisco_e1_config",
    "make_publication_pisco_e2_config", "make_publication_baseline_configs",
    "RevisionPISCOConfigTopology", "RevisionPISCOSolverTopology",
    "OriginalPISCOConfigTopology", "OriginalPISCOSolverTopology",
    "PISCOConfigTopology", "PISCOSolverTopology",
    "ScalarSurrogateSolver", "NoRevalSolver", "LatentFEMSolver",
    "ManufacturingHybridPISCOSolver", "BinaryObjectiveHybridPISCOSolver",
    "DynamicAnchorBinaryHybridSolver", "GuardedDynamicAnchorSolver",
    "FilteredNestedPCAFEMSolver", "FilteredBinaryNestedPCAFEMSolver",
    "FinProblem", "GaussianFinParameterization", "FinSearchConfig",
    "FinPISCOSolver", "FinRandomSolver", "FinDESolver",
    "FinCMAESSolver", "FinARDSolver",
})


# =====================================================================
# 23. SURROGATE-FREE SEARCH IN THE EXAMPLE-2 PCA SPACE
# =====================================================================

def common_filtered_design(rho_raw: np.ndarray, problem: LBracketProblem,
                           mask: np.ndarray) -> np.ndarray:
    """Apply the common cone filter and exact volume projection."""
    filter_matrix, filter_sum = _build_filter(
        problem.nelx, problem.nely, problem.rmin)
    raw = np.asarray(rho_raw, dtype=float).reshape(
        problem.nely, problem.nelx).copy()
    raw[~mask] = 0.0
    filtered = ((filter_matrix @ raw.ravel(order="F")) / filter_sum).reshape(
        (problem.nely, problem.nelx), order="F")
    filtered[~mask] = 0.0
    values = filtered[mask]
    lower, upper = -1.0, 1.0
    for _ in range(60):
        offset = 0.5 * (lower + upper)
        if np.clip(values + offset, 0.0, 1.0).mean() < problem.volfrac:
            lower = offset
        else:
            upper = offset
    output = np.zeros_like(filtered)
    output[mask] = np.clip(
        values + 0.5 * (lower + upper), 0.0, 1.0)
    return output


def volume_preserving_binary(rho: np.ndarray, mask: np.ndarray,
                             volfrac: float) -> np.ndarray:
    """Rank-project to the nearest feasible number of solid elements."""
    values = np.asarray(rho, dtype=float)[mask]
    solid_count = int(round(volfrac * values.size))
    binary = np.zeros(values.size, dtype=float)
    if solid_count:
        chosen = np.argpartition(
            values, values.size - solid_count)[-solid_count:]
        binary[chosen] = 1.0
    output = np.zeros_like(np.asarray(rho, dtype=float))
    output[mask] = binary
    return output


LATENT_BASELINE_SOLVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LATENT_BASELINE_PROTOCOL = "latent_direct_binary_separate_anchors_v2"


class _LatentDirectBinaryBaseline(BaseSolver):
    """Common scaffold: anchors, frozen PCA, B1 evaluation, budget."""

    method_name = "abstract"

    def __init__(self, problem: LBracketProblem, config, *,
                 cache_dir: Path | None = None, cache_enabled: bool = True):
        super().__init__(problem, config,
                         cache_dir=cache_dir, cache_enabled=cache_enabled)
        self._mask = _l_bracket_mask(problem.nelx, problem.nely)
        F, fixed = _l_bracket_load(problem.nelx, problem.nely)
        self._F_np = F
        self._fixed = fixed

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload.update({
            "revision_protocol": REVISION_PROTOCOL,
            "latent_baseline_protocol": LATENT_BASELINE_PROTOCOL,
            "latent_baseline_solver_sha256": LATENT_BASELINE_SOLVER_SHA256,
            "latent_baseline_method": self.method_name,
        })
        return payload

    def _evaluate_factory(self, decoder, call_offset=0) -> tuple[Callable, list]:
        problem = self.problem
        mask = np.asarray(self._mask, dtype=bool)
        state = {"fem_calls": 0, "best": math.inf, "best_design": None}
        trajectory: list[tuple[int, float]] = []

        def evaluate(z: np.ndarray) -> float:
            with torch.no_grad():
                raw = decoder.decode(torch.tensor(
                    np.asarray(z, dtype=np.float32)).reshape(1, -1))[
                    0, 0, :problem.nely, :problem.nelx].numpy()
            physical = volume_preserving_binary(
                common_filtered_design(raw, problem, mask), mask,
                problem.volfrac)
            _, compliance, _, _ = _fea(
                problem.nelx, problem.nely, physical, self._fixed,
                self._F_np, penal=problem.penal, E0=1.0, Emin=1e-9, nu=0.30)
            state["fem_calls"] += 1
            if compliance < state["best"]:
                state["best"] = float(compliance)
                state["best_design"] = physical.astype(np.float32)
            _append_trajectory(trajectory, (call_offset + state["fem_calls"], state["best"]))
            return float(compliance)

        return evaluate, [state, trajectory]

    def _search(self, evaluate: Callable, dimension: int, budget: int,
                rng: np.random.Generator) -> None:
        raise NotImplementedError

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        config = self.config
        problem = self.problem
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed + 1)

        dataset, anchor_fem, anchor_traj = _build_revision_simp_dataset(
            problem, config.n_init, seed,
            short_simp_iters=config.short_simp_iters,
            init_amp=config.anchor_init_amp)
        fields = np.stack([
            sample["generator_rho"] for sample in dataset[:config.n_init]])
        decoder, _info = _make_pca_decoder(
            fields, dz=config.latent_dim)

        evaluate, (state, trajectory) = self._evaluate_factory(decoder, anchor_fem)
        budget = int(config.T_cycles) * int(config.n_top)
        self._search(evaluate, int(config.latent_dim), budget, rng)

        total_fem = anchor_fem + state["fem_calls"]
        full_trajectory = list(trajectory)
        return SolverResult(
            best_compliance=float(state["best"]),
            best_design=np.asarray(state["best_design"], dtype=np.float32),
            fem_calls_total=int(total_fem),
            trajectory=full_trajectory, seed=int(seed),
            wall_time_s=float(time.perf_counter() - started),
            extra={
                "revision_protocol": REVISION_PROTOCOL,
                "latent_baseline_protocol": LATENT_BASELINE_PROTOCOL,
                "initial_anchor_trajectory": list(anchor_traj),
                "trajectory_representation": "binary_active_search",
                "latent_baseline_method": self.method_name,
                "uses_surrogate": False,
                "common_filter_in_optimisation": True,
                "candidate_filter_order": (
                    "decoder_clip -> cone_filter -> volume_projection "
                    "-> volume_preserving_rank_binarisation"),
                "generator_type": "pca",
                "latent_dim": int(config.latent_dim),
                "anchor_fem_calls": int(anchor_fem),
                "direct_search_fem_calls": int(state["fem_calls"]),
                "raw_design": np.asarray(
                    state["best_design"], dtype=np.float32),
            },
        )


class LatentCMAESBinaryBaseline(_LatentDirectBinaryBaseline):
    """Standard (mu/mu_w, lambda) CMA-ES in the frozen 16D latent space."""

    method_name = "latent_cmaes"

    def _search(self, evaluate, dimension, budget, rng):
        clip = float(self.config.latent_clip)
        lam = 4 + int(3 * math.log(dimension))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / float(np.sum(weights ** 2))
        cs = (mu_eff + 2.0) / (dimension + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(
            0.0, math.sqrt((mu_eff - 1.0) / (dimension + 1.0)) - 1.0) + cs
        cc = (4.0 + mu_eff / dimension) / (
            dimension + 4.0 + 2.0 * mu_eff / dimension)
        c1 = 2.0 / ((dimension + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
                  / ((dimension + 2.0) ** 2 + mu_eff))
        chi_n = math.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension ** 2))

        mean = np.zeros(dimension)
        sigma = 1.0
        covariance = np.eye(dimension)
        p_sigma = np.zeros(dimension)
        p_c = np.zeros(dimension)
        evaluations = 0
        generation = 0
        while evaluations + lam <= budget:
            generation += 1
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            standard = rng.standard_normal((lam, dimension))
            candidates = mean + sigma * (standard @ transform.T)
            candidates = np.clip(candidates, -clip, clip)
            fitness = np.array([evaluate(c) for c in candidates])
            evaluations += lam
            order = np.argsort(fitness)
            selected = candidates[order[:mu]]
            mean_old = mean
            mean = weights @ selected
            y = (mean - mean_old) / sigma
            inv_transform = eigenvectors @ np.diag(
                1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            p_sigma = ((1.0 - cs) * p_sigma
                       + math.sqrt(cs * (2.0 - cs) * mu_eff)
                       * (inv_transform @ y))
            h_sigma = float(np.linalg.norm(p_sigma)
                            / math.sqrt(1.0 - (1.0 - cs) ** (2 * generation))
                            < (1.4 + 2.0 / (dimension + 1.0)) * chi_n)
            p_c = ((1.0 - cc) * p_c
                   + h_sigma * math.sqrt(cc * (2.0 - cc) * mu_eff) * y)
            artmp = (selected - mean_old) / sigma
            covariance = ((1.0 - c1 - cmu) * covariance
                          + c1 * (np.outer(p_c, p_c)
                                  + (1.0 - h_sigma) * cc * (2.0 - cc)
                                  * covariance)
                          + cmu * artmp.T @ np.diag(weights) @ artmp)
            sigma *= math.exp(
                (cs / ds) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            sigma = float(np.clip(sigma, 1e-8, 2.0 * clip))
        # Spend any remainder of the budget around the incumbent mean.
        while evaluations < budget:
            candidate = np.clip(
                mean + sigma * rng.standard_normal(dimension), -clip, clip)
            evaluate(candidate)
            evaluations += 1


class LatentBOBinaryBaseline(_LatentDirectBinaryBaseline):
    """GP Bayesian optimisation (Matern 5/2, expected improvement)."""

    method_name = "latent_bo"
    n_random_init = 20
    n_acquisition_candidates = 4096

    def _search(self, evaluate, dimension, budget, rng):
        from scipy.stats import norm as gaussian
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel, Matern, WhiteKernel)
        import warnings

        clip = float(self.config.latent_clip)
        n_init = min(self.n_random_init, budget)
        points = list(rng.uniform(-clip, clip, size=(n_init, dimension)))
        values = [evaluate(p) for p in points]
        while len(values) < budget:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=np.ones(dimension),
                               length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(1e-6, (1e-10, 1e-1)))
            model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True,
                random_state=int(rng.integers(2 ** 31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                with audit_scope("surrogate_training"):
                    model.fit(np.asarray(points), np.asarray(values))
            best = float(np.min(values))
            uniform = rng.uniform(
                -clip, clip,
                size=(self.n_acquisition_candidates // 2, dimension))
            incumbent = np.asarray(points)[int(np.argmin(values))]
            local = np.clip(
                incumbent + 0.3 * rng.standard_normal(
                    (self.n_acquisition_candidates // 2, dimension)),
                -clip, clip)
            candidates = np.vstack([uniform, local])
            mean, std = model.predict(candidates, return_std=True)
            std = np.maximum(std, 1e-12)
            gamma = (best - mean) / std
            expected_improvement = std * (
                gamma * gaussian.cdf(gamma) + gaussian.pdf(gamma))
            choice = candidates[int(np.argmax(expected_improvement))]
            points.append(choice)
            values.append(evaluate(choice))


__all__ = sorted(set(__all__) | {
    "LATENT_BASELINE_PROTOCOL", "LatentBOBinaryBaseline",
    "LatentCMAESBinaryBaseline", "common_filtered_design",
    "volume_preserving_binary",
})


# =====================================================================
# 24. EXAMPLE-3 ARCHITECTURE AND PROGRESSIVE-SEARCH CONTROLS
# =====================================================================

class _ScalarFinCNN(nn.Module):
    """Same convolutional trunk as the fin field model, with scalar output."""

    def __init__(self, compliance_scale: float):
        super().__init__()
        self.compliance_scale = float(compliance_scale)
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 9 * 15, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        return self.body(density).squeeze(-1) * self.compliance_scale


class FinScalarPISCOSolver(BaseSolver):
    """Matched scalar-output control for the 35-variable fin benchmark."""

    IMPLEMENTATION_PROTOCOL = "fin_scalar_rectangular_exact_projection_v4"
    TORCH_THREADS = 1
    DETERMINISTIC_ALGORITHMS = True

    def __init__(self, problem: FinProblem, config: FinSearchConfig, **kwargs):
        if config.dimension != 35 or config.progressive_dimensions:
            raise ValueError("The reported scalar control uses fixed d=35")
        super().__init__(problem, config, **kwargs)

    def _cache_payload(self, seed: int) -> dict[str, Any]:
        payload = super()._cache_payload(seed)
        payload["implementation_protocol"] = self.IMPLEMENTATION_PROTOCOL
        payload["torch_threads"] = self.TORCH_THREADS
        payload["deterministic_algorithms"] = self.DETERMINISTIC_ALGORITHMS
        payload["mkldnn_enabled"] = False
        return payload

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        torch.set_num_threads(self.TORCH_THREADS)
        torch.use_deterministic_algorithms(self.DETERMINISTIC_ALGORITHMS)
        torch.backends.mkldnn.enabled = False
        rng = self._seed_everything(seed)
        problem, config = self.problem, self.config
        parameterization = GaussianFinParameterization(problem, 35)
        dataset: list[dict[str, Any]] = []
        trajectory: list[tuple[int, float]] = []
        best = math.inf
        best_design = None

        def evaluate(coordinates: np.ndarray) -> None:
            nonlocal best, best_design
            field = parameterization.decode(coordinates)
            _, compliance, physical = problem.evaluate(field)
            dataset.append({"rho": field.astype(np.float32),
                            "compliance": compliance,
                            "coordinates": np.asarray(coordinates).copy()})
            if compliance < best:
                best, best_design = compliance, physical
            _append_trajectory(trajectory, (len(dataset), float(best)))

        for _ in range(config.n_init):
            evaluate(rng.uniform(0.0, 1.0, 35))
        model = _ScalarFinCNN(float(np.mean([
            item["compliance"] for item in dataset])))
        while len(dataset) < config.budget:
            _progress_cycle(getattr(model,"_r2_updates",0)+1, math.ceil((config.budget-config.n_init)/config.n_top))
            with audit_scope("surrogate_training"):
                densities = torch.tensor(np.stack([item["rho"] for item in dataset])[:,None], dtype=torch.float32)
                targets = torch.tensor([item["compliance"] for item in dataset], dtype=torch.float32)
                report = _fit_field(model, densities, targets, None, [], None, 1.0,
                    steps=config.surrogate_steps, lr=2e-3, lam_data=1.0, lam_phys=0.0, scalar=True)
                audit = _ACTIVE_AUDIT.get()
                if audit is not None:
                    audit.training.append({"update":len(audit.training)+1, **report})
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            with audit_scope("candidate_search"):
                incumbent = parameterization.lift(min(dataset, key=lambda item: item["compliance"])["rho"])
                coordinates = torch.tensor(np.vstack([
                    incumbent,
                    *[rng.uniform(0.0, 1.0, 35)
                      for _ in range(config.multistart_count - 1)],
                ]), dtype=torch.float32, requires_grad=True)
                def objective(values):
                    fields = parameterization.decode_torch(values.clamp(0.0, 1.0))
                    return model(fields.unsqueeze(1))
                coordinates, scores = _packet_adam(coordinates, objective,
                    steps=config.multistart_steps, lr=config.multistart_lr,
                    project=lambda value: value.clamp(0.0, 1.0))
                prediction = scores.numpy()
                candidates = coordinates.numpy()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            for index in np.argsort(prediction)[:config.n_top]:
                if len(dataset) >= config.budget:
                    break
                evaluate(candidates[index])
                _record_prediction(prediction[index], dataset[-1]["compliance"], model._r2_updates, dataset[-1]["rho"])
        with audit_scope("independent_test"):
            test_rng = np.random.default_rng(seed + 900_001)
            audit = _ACTIVE_AUDIT.get()
            for _ in range(self.training_policy.independent_test_size):
                coordinates = test_rng.uniform(0.0, 1.0, parameterization.dimension)
                field = parameterization.decode(coordinates)
                tensor = torch.tensor(field[None,None], dtype=torch.float32)
                with torch.no_grad():
                    prediction = float(model(tensor)[0])
                _, actual, physical = problem.evaluate(field)
                if audit is not None:
                    audit.test_predictions.append({"predicted":prediction,"actual":float(actual),
                        "representation":"continuous_fin", "dimension":parameterization.dimension,
                        "sampling":"independent uniform coordinates on final parameterization", "coordinates":coordinates.tolist()})
        return SolverResult(
            best_compliance=float(best), best_design=best_design,
            fem_calls_total=len(dataset), trajectory=trajectory, seed=seed,
            wall_time_s=time.perf_counter() - started,
            extra={"protocol": "scalar_matched_field_cnn_v3",
                   "benchmark": "fin", "dimension": 35,
                   "uses_displacement_field": False})


class FinProgressiveARDSolver(_FinDirectSolver):
    """ARD Bayesian optimisation on the reported 35→62→135→243 schedule."""

    STAGE_EVALUATIONS = (24, 24, 24, 58)

    def _compute(self, *, seed: int) -> SolverResult:
        import warnings
        from scipy.stats import norm as gaussian
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel, Matern, WhiteKernel)

        started = time.perf_counter()
        rng = self._seed_everything(seed)
        parameterizations = [
            GaussianFinParameterization(self.problem, dimension)
            for dimension in (35, 62, 135, 243)]
        current = parameterizations[0]
        coordinates = list(rng.uniform(
            0.0, 1.0, (self.config.n_init, current.dimension)))
        values: list[float] = []
        designs: list[np.ndarray] = []
        transfer_diagnostics = []

        def evaluate(candidate: np.ndarray) -> None:
            field = current.decode(candidate)
            _, value, physical = self.problem.evaluate(field)
            values.append(value)
            designs.append(physical)

        for candidate in coordinates:
            evaluate(candidate)
        for stage_index, stage_budget in enumerate(self.STAGE_EVALUATIONS):
            if stage_index:
                previous = parameterizations[stage_index - 1]
                current = parameterizations[stage_index]
                coordinates = [current.lift(previous.raw_field(candidate)) for candidate in coordinates]
                errors = [float(np.linalg.norm(self.problem.physical_density(current.decode(c))-d)
                          /max(np.linalg.norm(d),1e-12)) for c,d in zip(coordinates,designs)]
                transfer_diagnostics.append({"stage":stage_index,"relative_field_errors":errors,
                    "label_semantics":"historical FEM values at original fields; approximate labels after transfer",
                    "re_evaluated":False})
            for _ in range(stage_budget):
                kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern(length_scale=np.ones(current.dimension),
                             length_scale_bounds=(1e-2, 1e2), nu=2.5)
                    + WhiteKernel(1e-6, (1e-10, 1e-1)))
                model = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    random_state=int(rng.integers(2 ** 31)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    with audit_scope("surrogate_training"):
                        model.fit(np.asarray(coordinates), np.asarray(values))
                incumbent_value = float(np.min(values))
                incumbent = np.asarray(coordinates)[int(np.argmin(values))]
                candidates = np.vstack([
                    rng.uniform(0.0, 1.0, (2048, current.dimension)),
                    np.clip(incumbent + 0.15 * rng.standard_normal(
                        (2048, current.dimension)), 0.0, 1.0)])
                mean, std = model.predict(candidates, return_std=True)
                std = np.maximum(std, 1e-12)
                improvement = (incumbent_value - mean) / std
                acquisition = std * (
                    improvement * gaussian.cdf(improvement)
                    + gaussian.pdf(improvement))
                choice = candidates[int(np.argmax(acquisition))]
                coordinates.append(choice)
                evaluate(choice)
        return self._result(
            seed, started, values, designs,
            {"method": "progressive_ard_bayesian_optimisation",
             "dimension_history": [35, 62, 135, 243],
             "transfer_diagnostics":transfer_diagnostics,
             "exploratory":True,
             "coordinate_transfer":
                 "sample_coarse_field_at_fine_controls"})


__all__ = sorted(set(__all__) | {
    "FinScalarPISCOSolver", "FinProgressiveARDSolver",
})


# =====================================================================
# 25. EXAMPLE-2 PCA REPRESENTATION DIAGNOSTIC
# =====================================================================

@dataclass(frozen=True)
class E2ProjectionStudyConfig:
    """Anchor configurations used for the published PCA projection study."""

    latent_dim: int = 16
    anchor_initialisation_amplitude: float = 0.20
    anchor_grid: tuple[tuple[int, int], ...] = (
        (40, 6), (60, 6), (80, 6), (120, 6),
        (40, 12), (80, 12), (40, 20))


class Example2ProjectionStudySolver(BaseSolver):
    """Project the converged SIMP topology into each reported anchor basis.

    The projected field is evaluated with the same filter, exact volume
    projection, rank binarisation and finite-element model as the Example-2
    method comparison.  This is a representation diagnostic, not a lower
    bound on every topology reachable by a PCA search.
    """

    def __init__(self, problem: LBracketProblem,
                 config: E2ProjectionStudyConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self._mask = _l_bracket_mask(
            problem.nelx, problem.nely).astype(bool)
        self._force, self._fixed = _l_bracket_load(
            problem.nelx, problem.nely)

    def _evaluate(self, raw: np.ndarray) -> tuple[float, np.ndarray]:
        filtered = common_filtered_design(raw, self.problem, self._mask)
        binary = volume_preserving_binary(
            filtered, self._mask, self.problem.volfrac)
        _, compliance, _, _ = _fea(
            self.problem.nelx, self.problem.nely, binary,
            self._fixed, self._force, penal=self.problem.penal,
            E0=1.0, Emin=1e-9, nu=0.30)
        return float(compliance), binary

    def _compute(self, *, seed: int) -> SolverResult:
        started = time.perf_counter()
        *_, simp_config = make_publication_baseline_configs()
        simp = SIMPSolver(
            self.problem, simp_config, cache_enabled=False).solve(seed=0)
        simp_raw = np.asarray(simp.extra["raw_design"], dtype=np.float64)
        simp_binary_compliance, simp_binary = self._evaluate(simp_raw)
        projections: dict[str, dict[str, Any]] = {}
        best_compliance = math.inf
        best_design = None
        fem_calls = int(simp.fem_calls_total) + 1
        trajectory = [(fem_calls, simp_binary_compliance)]
        for anchor_count, simp_iterations in self.config.anchor_grid:
            dataset, anchor_fem, _ = _build_revision_simp_dataset(
                self.problem, anchor_count, seed,
                short_simp_iters=simp_iterations,
                init_amp=self.config.anchor_initialisation_amplitude)
            fields = np.stack([
                sample["generator_rho"] for sample in dataset])
            decoder, _ = _make_pca_decoder(
                fields, dz=self.config.latent_dim)
            mean = decoder.mean.numpy().reshape(-1)
            components = decoder.components.numpy()
            scales = decoder.scales.numpy()
            centred = simp_raw.reshape(-1) - mean
            coordinates = ((centred @ components.T)
                           / np.maximum(scales, 1e-12))
            reconstruction = np.clip(
                ((coordinates * scales) @ components + mean).reshape(
                    simp_raw.shape), 0.0, 1.0)
            compliance, binary = self._evaluate(reconstruction)
            fem_calls += int(anchor_fem) + 1
            if compliance < best_compliance:
                best_compliance, best_design = compliance, binary
            _append_trajectory(trajectory, (fem_calls, min(
                best_compliance, simp_binary_compliance)))
            key = f"anchors{anchor_count}_iters{simp_iterations}"
            projections[key] = {
                "projected_simp_binary_compliance": compliance,
                "anchor_fem_calls": int(anchor_fem),
                "basis_rank": int(components.shape[0]),
            }
            print(
                f"example2 projection {key}: compliance={compliance:.3f}, "
                f"anchor FEM={anchor_fem}, rank={components.shape[0]}",
                flush=True)
        return SolverResult(
            best_compliance=float(best_compliance),
            best_design=np.asarray(best_design, dtype=np.float32),
            fem_calls_total=fem_calls, trajectory=trajectory, seed=seed,
            wall_time_s=time.perf_counter() - started,
            extra={
                "protocol": "e2_pca_projection_diagnostic_v1",
                "simp_direct_binary_compliance": simp_binary_compliance,
                "simp_direct_binary_design": simp_binary.astype(np.float32),
                "projections": projections,
                "interpretation": "representation diagnostic, not a bound",
            })


__all__ = sorted(set(__all__) | {
    "E2ProjectionStudyConfig", "Example2ProjectionStudySolver",
})

class MatchedScalarSurrogateSolver(RevisionPISCOSolverTopology):
    """Scalar compliance control sharing the field solver's entire search loop.

    The scalar head is lifted along F solely to reuse the analytic score F.T U.
    This is NOT a learned displacement field and no physics loss is allowed.
    """
    def __init__(self, problem, config, **kwargs):
        super().__init__(problem, dataclasses.replace(config, lam_phys=0.0), **kwargs)

    def _make_revision_model(self, displacement_scale):
        force = torch.tensor(self._F_np, dtype=torch.float32)
        class ScalarScoreAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar_head = _make_scalar_surrogate_cnn()
                self.register_buffer("score_lift", force/force.square().sum())
            def forward(self, density):
                return self.scalar_head(density)[:,None]*self.score_lift[None,:]
        return ScalarScoreAdapter()


@dataclass(frozen=True)
class MeshRefinementConfig:
    factors: tuple[int, ...] = (1, 2)
    source_method: str = "unspecified"
    source_seed: int = 0


class BinaryMeshRefinementSolver(BaseSolver):
    """Re-evaluate a fixed binary pixel geometry without filtering/optimisation.

    Uniform subdivision retains the original element aspect ratio and unit
    thickness. The existing Q4 plane-stress matrix is scale invariant under
    uniform 2D refinement. Left-edge support and the point load remain at the
    same physical locations. A corner point load can limit mesh convergence;
    this is a sensitivity check, not proof of a continuum error bound.
    """
    def __init__(self, problem, config, density, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.density = np.array(density, dtype=float, copy=True)
        if self.density.shape != (problem.nely, problem.nelx):
            raise ValueError("Binary refinement expects the original density grid")
        if not np.all((self.density == 0) | (self.density == 1)):
            raise ValueError("Refinement requires an exactly binary design")
        if not config.factors or config.factors[0] != 1 or any(int(f)!=f or f<1 for f in config.factors):
            raise ValueError("Refinement factors must be positive integers beginning with 1")

    def _cache_payload(self, seed):
        return {**super()._cache_payload(seed),
                "design_sha256":hashlib.sha256(self.density.tobytes()).hexdigest()}

    def _compute(self, *, seed):
        started = time.perf_counter()
        p = self.problem
        mask = _l_bracket_mask(p.nelx,p.nely)
        if np.any(self.density[~mask]):
            raise ValueError("Material outside original L-bracket domain")
        rows=[]
        for factor in self.config.factors:
            rho = self.density.repeat(factor,axis=0).repeat(factor,axis=1)
            transferred_mask = mask.repeat(factor,axis=0).repeat(factor,axis=1)
            force, fixed = _l_bracket_load(p.nelx*factor,p.nely*factor)
            with audit_scope("mesh_validation"):
                _, compliance, _, _ = _fea(p.nelx*factor,p.nely*factor,rho,fixed,force,penal=p.penal)
            rows.append({"factor":factor,"mesh":[p.nelx*factor,p.nely*factor],
                         "compliance":compliance,"volume_fraction":float(rho[transferred_mask].mean()),
                         "geometry_changed":False,"binary":True,"total_vertical_force":float(force[1::2].sum()),
                         "fixed_dof_count":len(fixed),"relative_change":compliance/rows[0]['compliance']-1 if rows else 0.0})
        return SolverResult(rows[-1]['compliance'],self.density,len(rows),
            [(i+1,row['compliance']) for i,row in enumerate(rows)],seed,
            time.perf_counter()-started,extra={"mesh_refinement":rows,
              "source_method":self.config.source_method,"source_seed":self.config.source_seed,
              "transfer":"uniform element subdivision; no refiltering or optimisation",
              "material":{"penal":p.penal,"E0":1.0,"Emin":1e-9,"nu":0.3,"thickness":1.0},
              "limitation":"point-loaded corner; refinement sensitivity is not a continuum error bound"})


def expected_publication_experiments(seeds=None, include_controls=True, examples=None):
    """Declarative completeness contract, independent of observed results."""
    plan={}
    e1=list(PUBLICATION_SEEDS if seeds is None else seeds)
    e3=list(FIN_SEEDS if seeds is None else seeds)
    for name in ('pisco','de','bo','cmaes','pisco_lambda_0','pisco_lambda_0.001','pisco_lambda_0.1',
                 'pisco_fixed_156','de_fixed_156','bo_fixed_156','cmaes_fixed_156'):
        plan['example1/'+name]=e1
    for name in ('pisco_binary','pisco_strong_anchors','pisco_dynamic_pca','pisco_static_pca',
                 'pisco_dynamic_pca_fixed_cycles','pisco_static_pca_fixed_cycles','pisco_vae','pisco_ae',
                 'pisco_vae_lambda_0','pisco_vae_lambda_1','pisco_vae_data_lambda_0'):
        plan['example2/'+name]=e1
    plan['example2/simp']=[0]
    plan['example2/pca_projection_study']=[7]
    if include_controls:
        for name in ('pisco_random_basis','scalar_surrogate','field_no_physics_matched','no_periodic_verification',
                     'latent_fem','latent_cmaes','latent_ard_bo','filtered_direct_fem','filtered_binary_direct_fem'):
            plan['example2/'+name]=e1
        plan['example2/simp_multistart_30']=[0]
    for dim in (35,62,135,243):
        for method in ('pisco','random','de','cmaes','ard_bo') if include_controls else ('pisco',):
            plan[f'example3/{method}_d{dim}']=e3
    plan['example3/pisco_progressive']=e3
    if include_controls:
        for name in ('pisco_no_physics_d35','scalar_surrogate_d35','ard_bo_progressive'):
            plan['example3/'+name]=e3
        for name in ('pisco_starts_6_d35','pisco_starts_12_d35','pisco_starts_24_d35',
                     'pisco_top_3_d35','pisco_top_6_d35','pisco_top_12_d35'):
            plan['example3/'+name]=e3[:3]
    plan['mesh_refinement/pisco_binary']=e1
    plan['mesh_refinement/simp']=[0]
    if examples is not None:
        allowed=set(examples)
        if 'example2' in allowed:allowed.add('mesh_refinement')
        plan={k:v for k,v in plan.items() if k.split('/')[0] in allowed}
    return plan

# Free-standing anchor builders share the same exclusive phase accounting.
_build_revision_simp_dataset = _audited_method("initial_data", _build_revision_simp_dataset)
