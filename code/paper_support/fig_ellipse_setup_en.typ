// Figure: setup of benchmark A — 2D cantilever plate with elliptical cutout
// Domain 120 mm × 60 mm, fixed support on the left, line load on the right (y = 25–35 mm)
// Ellipse: xc = 60, yc = 30, rx = 12, ry = 8, θ = 25°
// Included via #include.

#import "@preview/cetz:0.3.4": canvas, draw

#canvas(length: 1cm, {
  import draw: *

  let cg      = luma(80)
  let cdomain = luma(242)

  // Scaling: 1 mm → 0.1 canvas units
  let x0 = 1.8; let y0 = 0.9
  let px(mm) = x0 + mm / 10
  let py(mm) = y0 + mm / 10

  // ── Ellipse geometry ─────────────────────────────────────────────
  let ex = px(60); let ey = py(30)
  let erx = 1.2; let ery = 0.8
  let k = 0.5523
  let th  = 25.0 * calc.pi / 180.0
  let ca  = calc.cos(th); let sa = calc.sin(th)
  let rp(x, y) = (ex + x * ca - y * sa, ey + x * sa + y * ca)

  let ep0 = rp(erx, 0);   let ep1 = rp(0, ery)
  let ep2 = rp(-erx, 0);  let ep3 = rp(0, -ery)
  let ec00 = rp(erx, k*ery);   let ec01 = rp(k*erx, ery)
  let ec10 = rp(-k*erx, ery);  let ec11 = rp(-erx, k*ery)
  let ec20 = rp(-erx, -k*ery); let ec21 = rp(-k*erx, -ery)
  let ec30 = rp(k*erx, -ery);  let ec31 = rp(erx, -k*ery)

  // ── Domain ──────────────────────────────────────────────────────
  rect((px(0), py(0)), (px(120), py(60)),
       fill: cdomain, stroke: luma(40) + 1.5pt)

  // ── Fixed support ────────────────────────────────────────────────
  for i in range(13) {
    let y = i * 5.0
    line((px(-5), py(y)), (px(0), py(y + 3)), stroke: cg + 0.6pt)
  }
  line((px(0), py(0)), (px(0), py(60)), stroke: luma(40) + 2pt)

  // ── Line load ────────────────────────────────────────────────────
  line((px(120), py(25)), (px(120), py(35)), stroke: luma(40) + 2.5pt)
  for i in range(5) {
    let y = 25.0 + i * 2.5
    line((px(120)+0.4, py(y)+0.45), (px(120)+0.4, py(y)),
         mark: (end: ">", size: 0.13), stroke: luma(30) + 1.2pt)
  }

  // ── Ellipse fill (white) ─────────────────────────────────────────
  merge-path({
    bezier(ep0, ep1, ec00, ec01)
    bezier(ep1, ep2, ec10, ec11)
    bezier(ep2, ep3, ec20, ec21)
    bezier(ep3, ep0, ec30, ec31)
  }, close: true, fill: white, stroke: luma(30) + 1.5pt)

  // ── Center ──────────────────────────────────────────────────────
  circle((ex, ey), radius: 0.065, fill: luma(30), stroke: none)

  // ── r_x arrow (global +x, length = erx) ─────────────────────────
  // Tip lies on the ellipse boundary (ep0); label just outside
  line((ex, ey), ep0, mark: (end: ">", size: 0.11), stroke: luma(30) + 1.1pt)
  content((ep0.at(0) + ca*0.18, ep0.at(1) + sa*0.18),
    text(size: 7.5pt)[$r_x$])

  // ── r_y arrow (global +y, length = ery) ─────────────────────────
  // Tip lies on the ellipse boundary (ep1); label just outside
  line((ex, ey), ep1, mark: (end: ">", size: 0.11), stroke: luma(30) + 1.1pt)
  content((ep1.at(0) + (-sa)*0.18, ep1.at(1) + ca*0.18),
    text(size: 7.5pt)[$r_y$])

  // ── θ arc (from x direction up to 25°) ──────────────────────────
  // Place the label above the r_x arrow (originally: xc+6, yc+4 mm)
  arc((ex, ey), start: 0deg, stop: 25deg, radius: 0.48,
      stroke: luma(30) + 0.9pt)
  content((ex + 0.60, ey + 0.40), text(size: 7.5pt)[$theta$])

  // ── (x_c, y_c) label — centered below the point ─────────────────
  content((ex, ey - 0.10), anchor: "north",
    text(size: 7.5pt)[$(x_c, y_c)$])

  // ── Text labels (content in cetz is always on top) ───────────────
  content((px(4), py(60)+0.32), anchor: "west",
    text(size: 7.5pt)[fixed support])
  content((px(114), py(43)), anchor: "east",
    text(size: 7.5pt)[line load $overline(t)$])

  // ── Dimensions ────────────────────────────────────────────────────
  let dy = py(-8)
  line((px(0), py(0)), (px(0), dy), stroke: cg + 0.4pt)
  line((px(120), py(0)), (px(120), dy), stroke: cg + 0.4pt)
  line((px(0), dy), (px(120), dy),
       mark: (start: ">", end: ">", size: 0.1), stroke: cg + 0.5pt)
  content((px(60), dy - 0.28), anchor: "north", text(size: 7pt)[120 mm])

  let dx = px(-9)
  line((px(0), py(0)), (dx, py(0)), stroke: cg + 0.4pt)
  line((px(0), py(60)), (dx, py(60)), stroke: cg + 0.4pt)
  line((dx, py(0)), (dx, py(60)),
       mark: (start: ">", end: ">", size: 0.1), stroke: cg + 0.5pt)
  content((dx - 0.28, py(30)), anchor: "east", text(size: 7pt)[60 mm])
})
