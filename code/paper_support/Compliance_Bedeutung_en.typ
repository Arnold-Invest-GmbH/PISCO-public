// Figure: physical meaning of compliance — two cantilever beams
// Included via #include.

#import "@preview/cetz:0.3.4": canvas, draw

#canvas(length: 1cm, {
  import draw: *

  let cg   = rgb("#555555")
  let cblue = rgb("#1a56a0")
  let cred  = rgb("#c0392b")
  let cwall = rgb("#888888")
  let cbeam = rgb("#d0d0d0")

  // === Title ===
  content((6.0, 5.8), text(size: 10pt, weight: "bold")[
    $J = integral_(Gamma_N) overline(t) dot u thin d Gamma$])
  content((6.0, 5.2), text(size: 8pt, fill: cg)[
    Small compliance $arrow.r$ stiff structure; large compliance $arrow.r$ soft structure.])

  // ============================================================
  // LEFT BEAM — stiff (low compliance)
  // ============================================================

  // Fixed support (hatched)
  rect((0.3, 2.4), (0.6, 4.2), fill: cwall, stroke: cwall + 0.5pt)
  for i in range(7) {
    let y = 2.5 + i * 0.25
    line((0.15, y), (0.55, y + 0.25), stroke: cg + 0.4pt)
  }

  // Beam (undeformed reference, dashed)
  rect((0.6, 3.0), (4.2, 3.35), fill: cbeam, stroke: luma(120) + 0.8pt)

  // Deformed shape (slightly bent)
  bezier((0.6, 3.18), (4.2, 3.05), (2.0, 3.17), (3.5, 3.08),
         stroke: cblue + 1.8pt)

  // Load arrow
  line((4.0, 4.0), (4.0, 3.42), mark: (end: ">", size: 0.15),
       stroke: luma(30) + 1.2pt)
  content((4.0, 4.2), text(size: 7pt, weight: "bold")[$overline(t)$])

  // Annotation
  content((2.4, 2.1), text(size: 7.5pt, fill: cblue, weight: "bold")[
    small deflection])
  content((2.4, 1.7), text(size: 7pt, fill: cg)[
    $arrow.r$ small compliance])
  content((2.4, 1.35), text(size: 7pt, fill: cg)[
    $arrow.r$ stiff structure])

  // ============================================================
  // RIGHT BEAM — soft (high compliance)
  // ============================================================

  // Fixed support (hatched)
  rect((7.3, 2.4), (7.6, 4.2), fill: cwall, stroke: cwall + 0.5pt)
  for i in range(7) {
    let y = 2.5 + i * 0.25
    line((7.15, y), (7.55, y + 0.25), stroke: cg + 0.4pt)
  }

  // Beam (undeformed reference, dashed)
  rect((7.6, 3.0), (11.2, 3.35), fill: cbeam, stroke: luma(120) + 0.8pt)

  // Deformed shape (strongly bent)
  bezier((7.6, 3.18), (11.2, 2.40), (9.0, 3.12), (10.5, 2.55),
         stroke: cred + 1.8pt)

  // Load arrow
  line((11.0, 4.0), (11.0, 3.42), mark: (end: ">", size: 0.15),
       stroke: luma(30) + 1.2pt)
  content((11.0, 4.2), text(size: 7pt, weight: "bold")[$overline(t)$])

  // Annotation
  content((9.4, 2.1), text(size: 7.5pt, fill: cred, weight: "bold")[
    large deflection])
  content((9.4, 1.7), text(size: 7pt, fill: cg)[
    $arrow.r$ large compliance])
  content((9.4, 1.35), text(size: 7pt, fill: cg)[
    $arrow.r$ soft structure])

  // === Divider ===
  line((5.85, 1.2), (5.85, 4.6), stroke: (paint: luma(180), thickness: 0.5pt, dash: "dashed"))

  // === Labels left/right ===
  content((2.4, 4.6), text(size: 8pt, fill: cblue, weight: "bold")[stiff])
  content((9.4, 4.6), text(size: 8pt, fill: cred, weight: "bold")[soft])
})
