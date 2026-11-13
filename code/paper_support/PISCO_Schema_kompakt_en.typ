// Compact overview — four phases with central dataset D.
// Follows the color coding from PISCO_Schema_en.typ:
//   Blue = surrogate S_omega · Green = score B · Purple = critic C_omega
//   Red  = generator G        · Warm orange = hat(J) and final result
//   Everything else stays neutral (white / gray / charcoal).

#import "@preview/cetz:0.3.4": canvas, draw

#canvas(length: 0.85cm, {
  import draw: *

  // Same palette as the full schema
  let cb      = rgb("#1a56a0")
  let cfb     = rgb("#e9f1fb")
  let cpde    = rgb("#2a7b39")
  let cfpde   = rgb("#eaf5ec")
  let ckri    = rgb("#7c3aed")
  let cfkr    = rgb("#f1ebff")
  let cr      = rgb("#c0392b")
  let cfr     = rgb("#fbecea")
  let cg      = rgb("#555555")
  let cdark   = rgb("#1f2937")
  let cstep   = rgb("#fafbfc")
  let cstep-s = rgb("#cbd2d9") + 0.5pt
  let cwarm   = rgb("#d97706")

  let step-box(x0, y0, x1, y1, num, title) = {
    rect((x0, y0), (x1, y1), fill: cstep, stroke: cstep-s, radius: 0.16)
    circle((x0 + 0.42, y1 - 0.42), radius: 0.27, fill: cdark, stroke: none)
    content((x0 + 0.42, y1 - 0.42),
      text(size: 6.5pt, weight: "bold", fill: white)[#num])
    content((x0 + 0.85, y1 - 0.42), anchor: "west",
      text(size: 7.5pt, weight: "bold")[#title])
  }

  let pill(x, y, w, label, fg, bg) = {
    rect((x, y - 0.22), (x + w, y + 0.22),
         fill: bg, stroke: fg + 0.7pt, radius: 0.10)
    content((x + w/2, y),
            text(size: 5.6pt, weight: "bold", fill: fg)[#label])
  }

  // === 2x2 layout with central data box ===
  let lx = 0.3
  let pw = 5.0
  let gap_h = 2.6
  let ph = 3.9
  let by = 1.6
  let lx2 = lx + pw
  let rx  = lx2 + gap_h
  let rx2 = rx + pw
  let by2 = by + ph
  let gap_v = 0.55
  let ty  = by2 + gap_v
  let ty2 = ty + ph
  let tmid = ty + ph/2
  let bmid = by + ph/2
  let cx = (lx2 + rx) / 2
  let cy = (by2 + ty) / 2

  step-box(lx, ty, lx2, ty2, "1", [Initial dataset + generator choice])
  step-box(rx, ty, rx2, ty2, "2", [Train surrogate])
  step-box(rx, by, rx2, by2, "3", [Optimize (latent)])
  step-box(lx, by, lx2, by2, "4", [Revalidate])

  // Central data box D — neutral (data is not an entity);
  // the only warm accent refers to hat(J) convergence.
  let dbw = gap_h - 0.4
  let dbh = 1.9
  rect((cx - dbw/2, cy - dbh/2), (cx + dbw/2, cy + dbh/2),
        fill: white, stroke: cdark + 1.0pt, radius: 0.14)
  content((cx, cy + 0.65),
    text(size: 9pt, weight: "bold")[Dataset $D$])
  content((cx, cy + 0.20),
    text(size: 5.8pt, fill: cdark)[$\{(d_i, J_i, psi_i)\}_(i=1)^N$])
  content((cx, cy - 0.20),
    text(size: 5.4pt, fill: cg)[grows per cycle])
  content((cx, cy - 0.50),
    text(size: 5.4pt, fill: cg)[by $n_"top"$ entries])
  content((cx, cy - 0.85),
    text(size: 5.2pt, weight: "bold", fill: cwarm)[
      Loop until $abs(Delta hat(J)) < epsilon$])

  // Data-flow arrows from / to D — neutral gray
  line((lx2 + 0.05, tmid - 0.5), (cx - dbw/2 - 0.05, cy + 0.4),
        mark: (end: ">", size: 0.13), stroke: cg + 1.0pt)
  line((cx + dbw/2 + 0.05, cy + 0.4), (rx - 0.05, tmid - 0.5),
        mark: (end: ">", size: 0.13), stroke: cg + 1.0pt)
  line((lx2 + 0.05, bmid + 0.5), (cx - dbw/2 - 0.05, cy - 0.4),
        mark: (end: ">", size: 0.13), stroke: cg + 1.0pt)

  // Workflow arrows (charcoal)
  let rmid = rx + pw/2
  line((rmid, ty - 0.05), (rmid, by2 + 0.05),
        mark: (end: ">", size: 0.16), stroke: cdark + 1.4pt)
  line((rx - 0.05, bmid), (lx2 + 0.05, bmid),
        mark: (end: ">", size: 0.16), stroke: cdark + 1.4pt)

  // ════════════════════════════════════════════════════════════════════
  //  PANEL 1 — Initial dataset + generator
  // ════════════════════════════════════════════════════════════════════
  let f1cx = lx + pw/2
  let f1y = ty2 - 1.05
  pill(lx + 0.40, f1y, 4.2, [$N$ designs (random / short-SIMP) + FEM], cdark, white)
  content((f1cx, f1y - 0.55), text(size: 5.4pt, fill: cg)[
    one FEM call per design $arrow.r$ initial $D$])
  pill(lx + 0.40, f1y - 1.30, 4.2, [Choose generator $G$ once], cr, cfr)
  content((f1cx, f1y - 1.95), text(size: 5.4pt, fill: cg)[
    Case 1: analytic $G_"param"(a)$])
  content((f1cx, f1y - 2.30), text(size: 5.4pt, fill: cg)[
    Case 2: $G$ from $D$ as PCA or VAE decoder, fixed])

  // ════════════════════════════════════════════════════════════════════
  //  PANEL 2 — Train surrogate
  //  Two symmetric, neutral loss pills (data loss + physics residual);
  //  the score color (green) is NOT used here, since L_phys is not a
  //  scoring term but a surrogate loss.
  // ════════════════════════════════════════════════════════════════════
  let f2cx = rx + pw/2
  let f2y = ty2 - 1.05
  pill(rx + 0.40, f2y, 4.2, [Data loss $cal(L)_"data"$], cdark, white)
  content((f2cx, f2y - 0.50), text(size: 5.4pt, fill: cg)[
    surrogate fields against FEM solutions in $D$])
  pill(rx + 0.40, f2y - 1.10, 4.2, [Physics residual $cal(L)_"phys"$], cdark, white)
  content((f2cx, f2y - 1.65), text(size: 5.4pt, fill: cg)[
    equilibrium at collocation points])
  content((f2cx, f2y - 2.25), text(size: 5.4pt, weight: "bold", fill: cb)[
    surrogate $S_omega$ learns $hat(u), hat(sigma)$])
  content((f2cx, f2y - 2.55), text(size: 5.2pt, fill: cg)[
    then $cal(B)$ yields the scalar $hat(J)$ via quadrature])

  // ════════════════════════════════════════════════════════════════════
  //  PANEL 3 — Optimize in latent space
  //  z (neutral) → G(z) (red) → C_omega (purple) — strict legend match.
  // ════════════════════════════════════════════════════════════════════
  let f3cx = rx + pw/2
  let f3y = by2 - 1.05
  pill(rx + 0.30, f3y, 1.4, [$z$], cdark, white)
  line((rx + 1.75, f3y), (rx + 2.00, f3y),
    mark: (end: ">", size: 0.09), stroke: cdark + 0.8pt)
  pill(rx + 2.00, f3y, 1.4, [$G(z)$], cr, cfr)
  line((rx + 3.45, f3y), (rx + 3.70, f3y),
    mark: (end: ">", size: 0.09), stroke: cdark + 0.8pt)
  pill(rx + 3.70, f3y, 1.6, [$C_omega$], ckri, cfkr)

  content((f3cx, f3y - 0.65), text(size: 5.4pt, fill: cg)[
    multi-start against the frozen])
  content((f3cx, f3y - 1.00), text(size: 5.4pt, fill: cg)[
    chain $C_omega compose G$])

  line((rx + 4.7, f3y - 1.50), (rx + 0.6, f3y - 1.50),
    mark: (end: ">", size: 0.13),
    stroke: (paint: cdark, thickness: 1.0pt, dash: "dashed"))
  content((f3cx, f3y - 1.85), text(size: 5.4pt, weight: "bold", fill: cdark)[
    gradient: $partial cal(L) \/ partial z$])
  content((f3cx, f3y - 2.40), text(size: 5.4pt, fill: cg)[
    top-$n_"top"$ candidates to FEM])

  // ════════════════════════════════════════════════════════════════════
  //  PANEL 4 — Revalidate
  //  All pills neutral: candidate / FEM / true (J,psi) are data,
  //  not hat(J) — therefore no warm orange.
  // ════════════════════════════════════════════════════════════════════
  let f4cx = lx + pw/2
  let f4y = by2 - 1.05
  pill(lx + 0.30, f4y, 1.5, [candidate $hat(d)$], cdark, white)
  line((lx + 1.85, f4y), (lx + 2.10, f4y),
    mark: (end: ">", size: 0.09), stroke: cdark + 0.8pt)
  pill(lx + 2.10, f4y, 1.3, [FEM], cdark, white)
  line((lx + 3.45, f4y), (lx + 3.70, f4y),
    mark: (end: ">", size: 0.09), stroke: cdark + 0.8pt)
  pill(lx + 3.70, f4y, 1.6, [true $(J, psi)$], cdark, white)

  content((f4cx, f4y - 0.65), text(size: 5.4pt, fill: cg)[
    constraints met + improvement?])
  content((f4cx, f4y - 1.00), text(size: 5.4pt, weight: "bold", fill: cdark)[
    update best-so-far])
  content((f4cx, f4y - 1.50), text(size: 5.2pt, fill: cb)[
    extends $D$ $arrow.r$ surrogate retrains])
  content((f4cx, f4y - 1.95), text(size: 5.2pt, weight: "bold", fill: cr)[
    generator stays fixed])

  // ════════════════════════════════════════════════════════════════════
  //  Case-2 banner — refers to the generator choice (red)
  // ════════════════════════════════════════════════════════════════════
  rect((lx, by - 1.30), (rx2, by - 0.25),
        fill: cfr, stroke: cr + 0.9pt, radius: 0.10)
  content(((lx + rx2)/2, by - 0.55),
          text(size: 6.5pt, weight: "bold", fill: cr)[
            Case 2 — general design space (e.g., density field):])
  content(((lx + rx2)/2, by - 0.92),
          text(size: 5.8pt, fill: cdark)[
            in step 1 a low-dimensional decoder is derived from $D$
            (PCA by default, alternatively a pre-trained VAE); the
            decoder is the fixed generator $G$ (no retraining)])
})
