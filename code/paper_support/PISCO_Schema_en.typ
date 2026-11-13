// Detailed workflow schema — quiet, simplified variant.
// Five vertically stacked phases, four entity colors, generous whitespace.
// Surrogate outputs are explicitly wired to the score-function inputs.
//
//   0  Initialization        (dataset + generator)
//   1  Train surrogate
//   2  Build critic           (score ∘ surrogate)
//   3  Multi-start optimization in latent space
//   4  FEM revalidation + convergence test  (active loop back to 1)

#import "@preview/cetz:0.3.4": canvas, draw

#canvas(length: 0.85cm, {
  import draw: *

  // ────────────────────────────────────────────────────────────────────
  //  Four entity colors + two neutral tones. That's it.
  // ────────────────────────────────────────────────────────────────────
  let cb     = rgb("#1a56a0")        // surrogate $S_omega$
  let cfb    = rgb("#e9f1fb")
  let cpde   = rgb("#2a7b39")        // score $cal(B)$
  let cfpde  = rgb("#eaf5ec")
  let ckri   = rgb("#7c3aed")        // critic $C_omega = cal(B) compose S_omega$
  let cfkr   = rgb("#f1ebff")
  let cr     = rgb("#c0392b")        // generator $G$
  let cfr    = rgb("#fbecea")
  let cg     = rgb("#555555")        // helper lines, generic text
  let cdark  = rgb("#1f2937")        // structural lines, loop arrow
  let cstep  = rgb("#fafbfc")        // step background (very light)
  let cstep-s = rgb("#cbd2d9") + 0.5pt
  let cwarm  = rgb("#d97706")        // single accent: final output $hat(J)$ + result

  // ────────────────────────────────────────────────────────────────────
  //  Mini helpers
  // ────────────────────────────────────────────────────────────────────
  let nn-xs(cx, bw, n) = range(n).map(i => cx - bw/2 + bw * i / (n - 1))
  let nn-ys(cy, bh, ni) = if ni == 1 { (cy,) }
                          else { range(ni).map(j => cy - bh/2 + bh * j / (ni - 1)) }

  let nn-icon(cx, cy, bw, bh, layers, nr, col) = {
    let n = layers.len()
    let xs = nn-xs(cx, bw, n)
    let yss = layers.map(ni => nn-ys(cy, bh, ni))
    for k in range(n - 1) {
      for ya in yss.at(k) {
        for yb in yss.at(k + 1) {
          line((xs.at(k), ya), (xs.at(k+1), yb),
               stroke: (paint: col, thickness: 0.28pt))
        }
      }
    }
    for i in range(n) {
      for yi in yss.at(i) {
        circle((xs.at(i), yi), radius: nr, fill: col, stroke: none)
      }
    }
  }

  let step-num(x, y, num) = {
    circle((x, y), radius: 0.30, fill: cdark, stroke: none)
    content((x, y), text(size: 7pt, weight: "bold", fill: white)[#num])
  }

  let badge(x, y, w, label, fg, bg) = {
    rect((x, y - 0.18), (x + w, y + 0.18),
         fill: bg, stroke: fg + 0.5pt, radius: 0.07)
    content((x + w/2, y), text(size: 6pt, weight: "bold", fill: fg)[#label])
  }

  // ────────────────────────────────────────────────────────────────────
  //  Y layout — wide vertical gaps between steps on purpose.
  // ────────────────────────────────────────────────────────────────────
  let y0_top = 17.10
  let y0_bot = 14.50
  let y1_top = 13.60
  let y1_bot = 10.20
  let y2_top =  9.30
  let y2_bot =  5.60
  let y3_top =  4.70
  let y3_bot =  1.30
  let y4_top =  0.40
  let y4_bot = -3.30

  // ────────────────────────────────────────────────────────────────────
  //  Compact legend — four entity colors + warm accent for final output
  // ────────────────────────────────────────────────────────────────────
  badge( 1.5, 17.85, 2.0, [Surrogate $S_omega$],         cb,    cfb)
  badge( 4.0, 17.85, 2.0, [Score $cal(B)$],              cpde,  cfpde)
  badge( 6.5, 17.85, 2.0, [Critic $C_omega$],            ckri,  cfkr)
  badge( 9.0, 17.85, 2.0, [Generator $G$],               cr,    cfr)
  badge(11.5, 17.85, 2.5, [Result / $hat(J)$],           cwarm, rgb("#fff7ed"))

  // ════════════════════════════════════════════════════════════════════
  //  STEP 0 — Initialization
  // ════════════════════════════════════════════════════════════════════
  rect((0.5, y0_bot), (16.5, y0_top), fill: cstep, stroke: cstep-s, radius: 0.18)
  step-num(1.1, y0_top - 0.40, "0")
  content((8.50, y0_top - 0.40), text(size: 9pt, weight: "bold")[
    Initialization — dataset and generator])

  rect((1.5, 14.95), (8.0, 16.45), fill: white, stroke: cdark + 0.6pt, radius: 0.10)
  content((4.75, 16.10), text(size: 7pt, weight: "bold")[Initial dataset $D$])
  content((4.75, 15.65), text(size: 6.3pt, fill: cg)[
    $N$ designs (random or short-SIMP) + FEM])
  content((4.75, 15.20), text(size: 6.2pt, fill: cg)[
    $D = \{(d_i, J_i)\}_(i=1)^N$])

  rect((9.0, 14.95), (15.5, 16.45), fill: cfr, stroke: cr + 0.8pt, radius: 0.10)
  content((12.25, 16.10), text(size: 7pt, weight: "bold", fill: cr)[
    Choose generator $G$ once])
  content((12.25, 15.65), text(size: 6.2pt, fill: cg)[
    Case 1: $G_"param"(a) = H_beta(phi_a(x))$])
  content((12.25, 15.25), text(size: 6.2pt, fill: cg)[
    Case 2: derive $G$ from $D$ as PCA or VAE decoder, fixed])

  // Arrow 0 → 1
  line((8.5, y0_bot), (8.5, y1_top),
       mark: (end: ">", size: 0.16), stroke: cdark + 1.2pt)

  // ════════════════════════════════════════════════════════════════════
  //  STEP 1 — Train surrogate
  // ════════════════════════════════════════════════════════════════════
  rect((0.5, y1_bot), (16.5, y1_top), fill: cstep, stroke: cstep-s, radius: 0.18)
  step-num(1.1, y1_top - 0.40, "1")
  content((8.50, y1_top - 0.40), text(size: 9pt, weight: "bold")[
    Train physics-informed surrogate $S_omega$])

  // Left column: two loss blocks
  rect((1.3, 11.85), (7.4, 12.75), fill: white, stroke: cg + 0.6pt, radius: 0.08)
  content((4.35, 12.50), text(size: 6.6pt, weight: "bold")[
    Data loss $cal(L)_"data"$])
  content((4.35, 12.10), text(size: 6.0pt, fill: cg)[
    $1/abs(D) sum_i (hat(J)(d_i) - J_i)^2$  (via FEM quadrature)])

  rect((1.3, 10.55), (7.4, 11.55), fill: white, stroke: cg + 0.6pt, radius: 0.08)
  content((4.35, 11.30), text(size: 6.6pt, weight: "bold")[
    Physics residual $cal(L)_"phys"$])
  content((4.35, 10.85), text(size: 6.0pt, fill: cg)[
    $norm(nabla dot hat(sigma) + b)^2 + norm(hat(sigma) - sigma(hat(u)))^2$  at collocation points])

  // Right column: the surrogate itself
  rect((8.4, 10.55), (15.5, 12.95), fill: cfb, stroke: cb + 1.2pt, radius: 0.10)
  content((11.95, 12.65), text(size: 7.5pt, weight: "bold", fill: cb)[
    Surrogate $S_omega$ (PINN)])
  // Same dimensions as the surrogate icon in step 2 — same network, same look.
  nn-icon(11.95, 11.70, 3.2, 0.95, (3, 5, 5, 4), 0.06, cb)
  content((11.95, 10.85), text(size: 6.2pt, fill: cb)[
    $(d, x) arrow.r (hat(u)(x), hat(sigma)(x))$])

  // Loss arrows into the surrogate
  line((7.4, 12.30), (8.4, 12.10),
       mark: (end: ">", size: 0.13), stroke: cg + 1.0pt)
  line((7.4, 11.05), (8.4, 11.20),
       mark: (end: ">", size: 0.13), stroke: cg + 1.0pt)

  // Arrow 1 → 2
  line((8.5, y1_bot), (8.5, y2_top),
       mark: (end: ">", size: 0.16), stroke: cdark + 1.2pt)
  content((9.5, (y1_bot + y2_top)/2), anchor: "west",
    text(size: 6.3pt, fill: cb)[$omega^*$ frozen])

  // ════════════════════════════════════════════════════════════════════
  //  STEP 2 — Build critic (surrogate → score, purple container)
  //  Surrogate outputs are EXPLICITLY wired to score inputs — node by
  //  node.
  // ════════════════════════════════════════════════════════════════════
  rect((0.5, y2_bot), (16.5, y2_top), fill: cstep, stroke: cstep-s, radius: 0.18)
  step-num(1.1, y2_top - 0.40, "2")
  content((8.50, y2_top - 0.40), text(size: 9pt, weight: "bold")[
    Build critic: score $compose$ surrogate $arrow.r C_omega$])

  // Critic container (purple, solid) — surrounds the two inner boxes
  rect((1.5, 5.95), (13.5, 8.55), fill: cfkr, stroke: ckri + 1.4pt, radius: 0.14)
  content((7.5, 8.30), text(size: 6.5pt, weight: "bold", fill: ckri)[
    Critic $C_omega = cal(B) compose S_(omega^*)$])

  // Surrogate box (within the critic container, left)
  let surr_cx = 4.10
  let surr_cy = 7.05
  let surr_bw = 3.2
  let surr_bh = 0.95
  rect((surr_cx - 1.85, 6.20), (surr_cx + 1.85, 8.00),
       fill: cfb, stroke: cb + 1.2pt, radius: 0.08)
  content((surr_cx, 7.72), text(size: 6.8pt, weight: "bold", fill: cb)[
    $S_(omega^*)$])
  // Same layer structure as the surrogate icon in step 1 — same network!
  nn-icon(surr_cx, surr_cy, surr_bw, surr_bh, (3, 5, 5, 4), 0.06, cb)
  let surr_out_x = nn-xs(surr_cx, surr_bw, 4).last()
  let surr_out_ys = nn-ys(surr_cy, surr_bh, 4)
  content((surr_cx, 6.40), text(size: 5.6pt, fill: cb)[
    $hat(u), hat(sigma)$])

  // Score box (within the critic container, right)
  let bew_cx = 9.20
  let bew_cy = 7.05
  let bew_bw = 2.4
  let bew_bh = 0.95
  rect((bew_cx - 1.65, 6.20), (bew_cx + 1.65, 8.00),
       fill: cfpde, stroke: cpde + 1.2pt, radius: 0.08)
  content((bew_cx, 7.72), text(size: 6.8pt, weight: "bold", fill: cpde)[
    $cal(B)$])
  nn-icon(bew_cx, bew_cy, bew_bw, bew_bh, (4, 1), 0.06, cpde)
  let bew_in_x = nn-xs(bew_cx, bew_bw, 2).first()
  let bew_in_ys = nn-ys(bew_cy, bew_bh, 4)
  let bew_out_x = nn-xs(bew_cx, bew_bw, 2).last()
  content((bew_cx, 6.40), text(size: 5.6pt, fill: cpde)[
    fixed $w_k$])

  // Node-by-node connections from surrogate output to score input
  // (all four explicit, identical y each)
  for i in range(4) {
    line((surr_out_x, surr_out_ys.at(i)),
         (bew_in_x, bew_in_ys.at(i)),
         stroke: cdark + 0.6pt)
  }

  // Score output → J-hat (outside the critic container)
  line((bew_out_x, bew_cy), (12.40, bew_cy),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.8pt)
  rect((12.40, bew_cy - 0.35), (13.30, bew_cy + 0.35),
       fill: rgb("#fff7ed"), stroke: cwarm + 0.8pt, radius: 0.06)
  content((12.85, bew_cy), text(size: 7.5pt, weight: "bold", fill: cwarm)[
    $hat(J)$])

  // Annotation outside the container (right)
  content((14.00, bew_cy + 0.20), anchor: "west",
    text(size: 6.0pt, fill: cg)[
      scalar score])
  content((14.00, bew_cy - 0.20), anchor: "west",
    text(size: 6.0pt, fill: cg)[
      $hat(J) = C_omega(d)$])

  // Arrow 2 → 3
  line((8.5, y2_bot), (8.5, y3_top),
       mark: (end: ">", size: 0.16), stroke: cdark + 1.2pt)
  content((9.5, (y2_bot + y3_top)/2), anchor: "west",
    text(size: 6.3pt, fill: ckri)[$C_omega$ frozen])

  // ════════════════════════════════════════════════════════════════════
  //  STEP 3 — Multi-start optimization in latent space
  // ════════════════════════════════════════════════════════════════════
  rect((0.5, y3_bot), (16.5, y3_top), fill: cstep, stroke: cstep-s, radius: 0.18)
  step-num(1.1, y3_top - 0.40, "3")
  content((8.50, y3_top - 0.40), text(size: 9pt, weight: "bold")[
    Multi-start optimization in latent space])

  // Optimization chain: z → G → d̂ → C_omega → J̃
  // Slightly taller box so the caption is clearly separated from the inner boxes
  rect((1.4, 2.05), (13.6, 4.05),
       fill: white,
       stroke: (paint: cdark, thickness: 0.7pt, dash: "dashed"), radius: 0.10)
  content((7.5, 3.85), text(size: 6.3pt, fill: cg)[
    frozen chain — only the search vector $z$ is updated by gradient])

  // z on the left
  content((1.95, 2.95), text(size: 7pt, weight: "bold")[$z$])
  line((2.20, 2.95), (3.20, 2.95),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.8pt)

  // Generator box
  rect((3.20, 2.30), (6.10, 3.50), fill: cfr, stroke: cr + 1.0pt, radius: 0.08)
  content((4.65, 3.20), text(size: 6.8pt, weight: "bold", fill: cr)[
    Generator $G$])
  content((4.65, 2.55), text(size: 6.0pt, fill: cg)[$hat(d) = G(z)$])

  // d̂
  line((6.10, 2.95), (7.30, 2.95),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.8pt)
  content((6.70, 3.20), text(size: 6pt, fill: cg)[$hat(d)$])

  // Critic box (purple)
  rect((7.30, 2.30), (10.40, 3.50), fill: cfkr, stroke: ckri + 1.0pt, radius: 0.08)
  content((8.85, 3.20), text(size: 6.8pt, weight: "bold", fill: ckri)[
    Critic $C_(omega^*)$])
  content((8.85, 2.55), text(size: 6.0pt, fill: cg)[
    $cal(B) compose S_(omega^*)$])

  // J̃
  line((10.40, 2.95), (11.40, 2.95),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.8pt)
  content((11.85, 2.95), text(size: 7pt, weight: "bold", fill: cwarm)[$hat(J)$])

  // Gradient backwards (dashed, dark gray) — clearly under the chain
  line((12.10, 2.95), (12.85, 2.95),
       stroke: cdark + 0.8pt)
  line((12.85, 2.95), (12.85, 1.70),
       stroke: cdark + 0.8pt)
  line((12.85, 1.70), (1.95, 1.70),
       stroke: (paint: cdark, thickness: 0.8pt, dash: "dashed"))
  line((1.95, 1.70), (1.95, 2.65),
       mark: (end: ">", size: 0.16),
       stroke: (paint: cdark, thickness: 0.8pt, dash: "dashed"))
  content((7.5, 1.50), text(size: 6.0pt, fill: cdark)[
    $partial cal(L) \/ partial z$  — gradient backward through both frozen networks])

  // Arrow 3 → 4: the top-n best candidates go to revalidation
  line((8.5, y3_bot), (8.5, y4_top),
       mark: (end: ">", size: 0.16), stroke: cdark + 1.2pt)
  content((9.5, (y3_bot + y4_top)/2), anchor: "west",
    text(size: 6.3pt, fill: cdark)[Top-$n_"top"$ candidates $hat(d)^((1)) , dots, hat(d)^((n_"top"))$])

  // ════════════════════════════════════════════════════════════════════
  //  STEP 4 — FEM revalidation + convergence test
  // ════════════════════════════════════════════════════════════════════
  rect((0.5, y4_bot), (16.5, y4_top), fill: cstep, stroke: cstep-s, radius: 0.18)
  step-num(1.1, y4_top - 0.40, "4")
  content((8.50, y4_top - 0.40), text(size: 9pt, weight: "bold")[
    FEM revalidation and convergence test])

  // Revalidation chain as a horizontal flow in its own box
  rect((1.4, -1.30), (13.6, -0.40), fill: white, stroke: cdark + 0.5pt, radius: 0.10)

  // 1) Input: top-n candidates (small box)
  rect((1.65, -1.10), (3.40, -0.60), fill: cstep, stroke: cdark + 0.6pt, radius: 0.06)
  content((2.525, -0.85), text(size: 6.2pt, weight: "bold")[
    Top-$n_"top"$ $hat(d)$])

  line((3.40, -0.85), (4.05, -0.85),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.7pt)

  // 2) FEM solver
  rect((4.05, -1.10), (5.55, -0.60), fill: cstep, stroke: cdark + 0.6pt, radius: 0.06)
  content((4.80, -0.85), text(size: 6.2pt, weight: "bold")[
    FEM $cal(S)$])

  line((5.55, -0.85), (6.20, -0.85),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.7pt)

  // 3) True (J, ψ)
  rect((6.20, -1.10), (8.10, -0.60), fill: white,
       stroke: cg + 0.6pt, radius: 0.06)
  content((7.15, -0.85), text(size: 6.2pt, weight: "bold")[
    true $(J, psi)$])

  line((8.10, -0.85), (8.75, -0.85),
       mark: (end: ">", size: 0.13), stroke: cdark + 0.7pt)

  // 4) Dataset grows
  rect((8.75, -1.10), (13.35, -0.60), fill: white, stroke: cg + 0.6pt, radius: 0.06)
  content((11.05, -0.85), text(size: 6.2pt, weight: "bold")[
    $D arrow.l D union {(hat(d), J)}$])

  // Arrow from revalidation to the convergence decision
  line((8.5, -1.30), (8.5, -1.75),
       mark: (end: ">", size: 0.14), stroke: cdark + 0.9pt)

  // ── Convergence decision (diamond) ──────────────────────────────────
  let kx = 8.5
  let ky = -2.50
  let khw = 2.2
  let khh = 0.70
  line((kx, ky + khh), (kx + khw, ky), (kx, ky - khh), (kx - khw, ky), close: true,
       fill: white, stroke: cdark + 1.0pt)
  content((kx, ky + 0.20), text(size: 6.6pt, weight: "bold", fill: cdark)[
    Convergence test])
  content((kx, ky - 0.22), text(size: 5.8pt, fill: cdark)[
    $abs(Delta hat(J)) < epsilon$  or  $t = T_max$ ?])

  // ── "no" → loop back INTO step 1 (surrogate is retrained) ────────────
  // The arrow ends clearly INSIDE step 1, not in the gap to step 2
  let y1_mid = (y1_top + y1_bot) / 2   // = 11.90, well within step 1
  line((kx - khw, ky), (0.20, ky),
       stroke: (paint: cdark, thickness: 1.2pt, dash: "dashed"))
  line((0.20, ky), (0.20, y1_mid),
       stroke: (paint: cdark, thickness: 1.2pt, dash: "dashed"))
  line((0.20, y1_mid), (1.20, y1_mid),
       mark: (end: ">", size: 0.18),
       stroke: (paint: cdark, thickness: 1.2pt, dash: "dashed"))
  content((kx - khw - 0.05, ky + 0.30), anchor: "east",
    text(size: 6.4pt, weight: "bold", fill: cdark)[no])
  content((-0.05, (y1_mid + ky)/2), anchor: "east",
    text(size: 6.0pt, weight: "bold", fill: cdark)[
      Loop:\ $D$ has been\ extended\ $arrow$ surrogate\ retrains on $D$])

  // ── "yes" → on to the result ────────────────────────────────────────
  content((kx + 0.25, ky - khh - 0.15), anchor: "west",
    text(size: 6.4pt, weight: "bold", fill: cwarm)[yes])
  line((kx, ky - khh), (kx, -4.00),
       mark: (end: ">", size: 0.16), stroke: cwarm + 1.2pt)

  // ════════════════════════════════════════════════════════════════════
  //  RESULT
  // ════════════════════════════════════════════════════════════════════
  rect((4.0, -5.00), (13.0, -4.00),
       fill: rgb("#fff7ed"), stroke: cwarm + 1.2pt, radius: 0.12)
  circle((4.55, -4.50), radius: 0.32, fill: cwarm, stroke: none)
  content((4.55, -4.50), text(size: 8pt, fill: white, weight: "bold")[✓])
  content((8.85, -4.35), text(size: 7.5pt, weight: "bold", fill: cwarm)[
    Optimal feasible design $d^* = G(z^*)$])
  content((8.85, -4.75), text(size: 6.0pt, fill: cg)[
    after $T$ loop cycles, $approx N + T dot n_"top"$ FEM calls])
})
