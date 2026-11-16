// Wiring in the candidate-search step — R2 content correction of the R1 layout:
// Generator → Surrogate → Score — all three frozen (upper chain),
// followed by the FEM feedback path that closes the active loop (lower row).
// Layout, colours and geometry of the upper chain are those of
// code/paper_support/Schema_Verschaltung_Extrakt_en.typ; only labels that
// contradicted the R2 method description are changed.
// Surrogate + score are explicitly grouped into a critic container.
//
// Follows the color coding from PISCO_Schema_en.typ:
//   Blue = Surrogate S_omega · Green = Score B · Purple = Critic C_omega
//   Red  = Generator G       · Warm orange = hat(J)
//   Data-flow arrows, structure lines, and the gradient stay neutral.
//
// Drawing order: container → inner boxes → nn icons → arrows.
// Arrows are drawn last so their heads are NOT covered by the input
// nodes of the downstream boxes, and they end ~0.10 before the target
// node, keeping a clean gap visible.

#import "@preview/cetz:0.3.4": canvas, draw

#canvas(length: 0.85cm, {
  import draw: *

  // Same palette as the full schema
  let cb     = rgb("#1a56a0")
  let cfb    = rgb("#e9f1fb")
  let cpde   = rgb("#2a7b39")
  let cfpde  = rgb("#eaf5ec")
  let ckri   = rgb("#7c3aed")
  let cfkr   = rgb("#f1ebff")
  let cr     = rgb("#c0392b")
  let cfr    = rgb("#fbecea")
  let cg     = rgb("#555555")
  let cdark  = rgb("#1f2937")
  let cwarm  = rgb("#d97706")

  let nn-icon(cx, cy, bw, bh, layers, nr, col) = {
    let n = layers.len()
    let xs = range(n).map(i => cx - bw/2 + bw * i / (n - 1))
    let yss = layers.map(ni =>
      if ni == 1 { (cy,) }
      else { range(ni).map(j => cy - bh/2 + bh * j / (ni - 1)) })
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

  let badge(x, y, w, label, fg, bg) = {
    rect((x, y - 0.18), (x + w, y + 0.18),
         fill: bg, stroke: fg + 0.6pt, radius: 0.07)
    content((x + w/2, y), text(size: 6.8pt, weight: "bold", fill: fg)[#label])
  }

  // ════════════════════════════════════════════════════════════════════
  //  Geometry centrally defined — all arrow calculations rely on this
  // ════════════════════════════════════════════════════════════════════
  // Vertical axis: all networks centered at cy = 4.10 with bh = 1.6
  let cy_net = 4.10
  let bh_net = 1.6
  let nr     = 0.07           // node radius
  let gap    = 0.10           // gap between arrowhead and target node

  // Generator geometry
  let g_cx = 3.30
  let g_bw = 3.4
  // (2,4,5,4,3): output layer has 3 nodes
  let g_out_x = g_cx + g_bw/2                              // 5.00
  let g_out_ys = (cy_net - bh_net/2,                       // 3.30
                  cy_net,                                  // 4.10
                  cy_net + bh_net/2)                       // 4.90

  // Surrogate geometry
  let s_cx = 8.30
  let s_bw = 3.4
  // (3,5,5,4): input layer has 3 nodes, output layer has 4 nodes
  let s_in_x = s_cx - s_bw/2                               // 6.60
  let s_in_ys = g_out_ys                                   // same y values as G output
  let s_out_x = s_cx + s_bw/2                              // 10.00
  let s_out_ys = (cy_net - bh_net/2 + 0 * bh_net/3,        // 3.30
                  cy_net - bh_net/2 + 1 * bh_net/3,        // 3.833
                  cy_net - bh_net/2 + 2 * bh_net/3,        // 4.367
                  cy_net - bh_net/2 + 3 * bh_net/3)        // 4.90

  // Score geometry
  let b_cx = 13.00
  let b_bw = 2.5
  // (4,1): input layer has 4 nodes, output layer has 1 node
  let b_in_x = b_cx - b_bw/2                               // 11.75
  let b_in_ys = s_out_ys                                   // same y values as S output
  let b_out_x = b_cx + b_bw/2                              // 14.25

  // ════════════════════════════════════════════════════════════════════
  //  Title
  // ════════════════════════════════════════════════════════════════════
  content((8.50, 7.30), text(size: 9pt, weight: "bold")[
    Candidate search — wiring Generator $arrow.r$
    Surrogate $arrow.r$ Score])
  content((8.50, 6.95), text(size: 6.8pt, fill: cg)[
    Fixed parameters; $R(z) = lambda_z norm(z)^2$ in case 2, $0$ otherwise])

  // ════════════════════════════════════════════════════════════════════
  //  Input z (far left)
  // ════════════════════════════════════════════════════════════════════
  content((-0.50, 4.30), text(size: 9pt, weight: "bold")[$z$])
  content((-0.50, 3.95), text(size: 6.8pt, fill: cg)[search vector])

  // ════════════════════════════════════════════════════════════════════
  //  CRITIC CONTAINER (purple, encloses surrogate + score)
  //  drawn first — as background behind the inner boxes
  // ════════════════════════════════════════════════════════════════════
  rect((6.10, 1.85), (14.90, 6.40),
       fill: cfkr, stroke: ckri + 1.4pt, radius: 0.12)
  content((10.50, 6.18), text(size: 7pt, weight: "bold", fill: ckri)[
    Critic $C_omega = cal(B) compose S_(omega^*)$ (frozen)])

  // ════════════════════════════════════════════════════════════════════
  //  GENERATOR BOX (red)
  // ════════════════════════════════════════════════════════════════════
  rect((1.30, 1.85), (5.30, 6.40),
       fill: cfr, stroke: cr + 1.6pt, radius: 0.10)
  content((g_cx, 6.10), text(size: 7.5pt, weight: "bold", fill: cr)[
    Generator $G$])
  content((g_cx, 5.65), text(size: 6.8pt, fill: cg)[
    Case 1: fixed analytical map])
  content((g_cx, 5.30), text(size: 6.8pt, fill: cg)[
    Case 2: PCA / VAE decoder])
  nn-icon(g_cx, cy_net, g_bw, bh_net, (2, 4, 5, 4, 3), nr, cr)
  content((g_cx, 2.85), text(size: 7pt, fill: cr)[$d = G(z)$])
  badge(1.55, 2.20, 1.5, [$G$ frozen], cr, cfr)

  // ════════════════════════════════════════════════════════════════════
  //  SURROGATE BOX (left inner, blue)
  //  Generous height — formula just below the network, badge with margin.
  // ════════════════════════════════════════════════════════════════════
  rect((6.30, 2.00), (10.30, 5.90),
       fill: cfb, stroke: cb + 1.4pt, radius: 0.10)
  content((s_cx, 5.60), text(size: 7.5pt, weight: "bold", fill: cb)[
    Surrogate $S_(omega^*)$])
  content((s_cx, 5.25), text(size: 6.8pt, fill: cg)[
    nodal displacements])
  nn-icon(s_cx, cy_net, s_bw, bh_net, (3, 5, 5, 4), nr, cb)
  content((s_cx, 2.85), text(size: 6.8pt, fill: cb)[
    $hat(U) = S_(omega^*) (d)$])
  badge(6.50, 2.30, 1.4, [$omega^*$ fixed], cb, cfb)

  // ════════════════════════════════════════════════════════════════════
  //  SCORE BOX (right inner, green)
  // ════════════════════════════════════════════════════════════════════
  rect((11.30, 2.00), (14.70, 5.90),
       fill: cfpde, stroke: cpde + 1.4pt, radius: 0.10)
  content((b_cx, 5.60), text(size: 7.5pt, weight: "bold", fill: cpde)[
    Score $cal(B)$])
  content((b_cx, 5.25), text(size: 6.8pt, fill: cg)[
    analytic — external work])
  nn-icon(b_cx, cy_net, b_bw, bh_net, (4, 1), nr, cpde)
  content((b_cx, 2.85), text(size: 6.8pt, fill: cpde)[
    $hat(J) = hat(U)^top F$])
  badge(11.55, 2.30, 2.75, [fixed load vector $F$], cpde, cfpde)

  // ════════════════════════════════════════════════════════════════════
  //  OUTPUT BOX hat(J) (warm) — far right
  // ════════════════════════════════════════════════════════════════════
  rect((15.70, 3.70), (16.85, 4.50),
       fill: rgb("#fff7ed"), stroke: cwarm + 1.0pt, radius: 0.06)
  content((16.275, 4.10), text(size: 9pt, weight: "bold", fill: cwarm)[
    $hat(J)$])

  // ════════════════════════════════════════════════════════════════════
  //  CONNECTING ARROWS — drawn last so the heads are not covered
  //  by node circles painted afterwards. They start just outside the
  //  source node (+gap) and end just before the target node (-gap),
  //  so the heads stand free.
  // ════════════════════════════════════════════════════════════════════

  // Input arrow z → generator (1st node of input layer at x = g_cx - g_bw/2)
  let g_in_x = g_cx - g_bw/2                               // 1.60
  line((-0.10, 4.10), (g_in_x - gap, 4.10),
       mark: (end: ">", size: 0.18), stroke: cdark + 1.4pt)

  // Generator → surrogate (3 parallel connections)
  for sy in g_out_ys {
    line((g_out_x + gap, sy), (s_in_x - gap, sy),
         mark: (end: ">", size: 0.14), stroke: cdark + 1.1pt)
  }

  // Surrogate → score (4 parallel connections)
  for sy in s_out_ys {
    line((s_out_x + gap, sy), (b_in_x - gap, sy),
         mark: (end: ">", size: 0.14), stroke: cdark + 1.1pt)
  }

  // Score → hat(J) box (1 connection from the output node)
  line((b_out_x + gap, cy_net), (15.70 - 0.05, cy_net),
       mark: (end: ">", size: 0.18), stroke: cdark + 1.4pt)

  // ════════════════════════════════════════════════════════════════════
  //  Gradient backwards (dashed, neutral) — structural information
  // ════════════════════════════════════════════════════════════════════
  line((14.70, 1.20), (3.30, 1.20),
       mark: (end: ">", size: 0.22),
       stroke: (paint: cdark, thickness: 1.2pt, dash: "dashed"))
  content((8.50, 0.80), text(size: 6.8pt, weight: "bold", fill: cdark)[
    $nabla_z (hat(J) + R(z))$ — update search variables through the fixed components])

  // ════════════════════════════════════════════════════════════════════
  //  LOWER PATH — FEM feedback after the search (not differentiated).
  //  Only the selected candidates are evaluated by the FEM, appended to
  //  the dataset, and used to retrain the surrogate for the next cycle.
  // ════════════════════════════════════════════════════════════════════
  let cstep   = rgb("#fafbfc")
  let cstep-s = rgb("#cbd2d9") + 0.5pt
  rect((1.30, -2.35), (14.90, 0.25), fill: cstep, stroke: cstep-s, radius: 0.12)
  content((8.10, -0.05), text(size: 6.8pt, weight: "bold", fill: cdark)[
    After the search — FEM feedback closes the active loop])

  // 1) Select and verify
  rect((1.60, -2.05), (5.30, -0.45), fill: white, stroke: cdark + 0.8pt, radius: 0.08)
  content((3.45, -0.80), text(size: 7pt, weight: "bold", fill: cdark)[
    Select and verify])
  content((3.45, -1.25), text(size: 6.8pt, fill: cg)[
    top-$n_"top"$ by $hat(J) + R(z)$])
  content((3.45, -1.65), text(size: 6.8pt, fill: cg)[
    FEM $cal(S)$ returns the true $J$])

  // 2) Extend the dataset
  rect((6.30, -2.05), (10.30, -0.45), fill: white, stroke: cdark + 0.8pt, radius: 0.08)
  content((8.30, -0.80), text(size: 7pt, weight: "bold", fill: cdark)[
    Extend the dataset])
  content((8.30, -1.25), text(size: 6.8pt, fill: cg)[
    $D arrow.l D union {(d, J)}$])
  content((8.30, -1.65), text(size: 6.8pt, fill: cg)[
    update best verified design])

  // 3) Retrain the surrogate (blue = surrogate)
  rect((11.30, -2.05), (14.60, -0.45), fill: cfb, stroke: cb + 1.0pt, radius: 0.08)
  content((12.95, -0.80), text(size: 7pt, weight: "bold", fill: cb)[
    Retrain the surrogate])
  content((12.95, -1.25), text(size: 6.8pt, fill: cg)[
    update $omega$ using $D$])
  content((12.95, -1.65), text(size: 6.8pt, fill: cg)[
    for the next search cycle])

  line((5.30 + 0.05, -1.25), (6.30 - 0.05, -1.25),
       mark: (end: ">", size: 0.14), stroke: cdark + 1.1pt)
  line((10.30 + 0.05, -1.25), (11.30 - 0.05, -1.25),
       mark: (end: ">", size: 0.14), stroke: cdark + 1.1pt)
})
