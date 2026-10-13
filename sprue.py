"""Remove casting leftovers (sprue, clay or wax blob, loose bits) from the piece mask
WITHOUT cutting into the ring itself.

Runs on the foreground mask that engine.py already makes. No TensorFlow in here, so it
can be tested on its own.

How the sprue is removed:
  1. Fit an ellipse to the ring's inner hole (RANSAC, so the clay or sprue touching the
     hole edge cannot pull the fit off).
  2. Look outward from the hole in 360 directions and record where the metal ends. That is
     the ring's outer edge. A sprue shows up as a run of directions where the edge jumps
     far out.
  3. Only the run that sits next to a red cut mark counts as sprue. Heads, bezels and
     other designs are left alone.
  4. Across that run, draw the outer edge as a smooth bridge between the ring edge on
     each side. Everything inside the bridge is ring and stays. Only the sprue metal
     outside it is removed, so the bottom keeps its real thickness and shape.
  5. If the clay hid part of the ring bottom, that stretch is rebuilt from the metal
     next to it (a smear of real ring texture along the circle, not invented detail).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIDE = 2048        # geometry runs at this size (never on the photo's pixels), then masks are scaled back up
N_THETA = 360           # one ray per degree around the hole


@dataclass
class CleanResult:
    mask: np.ndarray               # bool, cleaned piece mask (full resolution)
    sprue: np.ndarray              # bool, pixels removed as sprue
    clay: np.ndarray               # bool, pixels removed as clay / wax / grey blob
    loose: np.ndarray              # bool, pixels removed as loose fragments
    marks: np.ndarray              # bool, red cut marks that were found
    fill: np.ndarray               # bool, hidden ring pixels that were rebuilt
    before: np.ndarray             # bool, the piece mask as it was before any cleanup
    rgb_filled: np.ndarray | None  # RGB with the rebuilt pixels painted in (None if nothing rebuilt)
    band_width: float              # estimated band width of the piece, full-resolution pixels
    note: str                      # one line saying what happened, for the UI


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _to_u8(m: np.ndarray) -> np.ndarray:
    return m.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def find_cut_marks(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Red marker strokes on the metal. Gold is orange-ish, so we ask for a strong, truly red hue."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 7) | (h >= 172)) & (s >= 120) & (v >= 50)
    red &= cv2.dilate(_to_u8(mask), _disk(3)) > 0
    red = cv2.morphologyEx(_to_u8(red), cv2.MORPH_OPEN, _disk(1))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    min_area = max(12, 8e-6 * mask.size)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(cc, keep)


def _band_half_width(mask: np.ndarray) -> float:
    dt = cv2.distanceTransform(_to_u8(mask), cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, _disk(2))) & (dt > 2)
    vals = dt[ridge]
    return float(np.median(vals)) if vals.size else 0.0


def _drop_small_parts(mask: np.ndarray, min_frac: float) -> np.ndarray:
    """Remove separate pieces smaller than min_frac of the largest one."""
    n, cc, stats, _ = cv2.connectedComponentsWithStats(_to_u8(mask), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_frac * areas.max()]
    return np.isin(cc, keep)


def _find_clay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dark, colourless blob stuck to a coloured metal (the grey clay in the photo).
    Skipped when the metal itself is colourless (silver, white gold), or it would eat the piece."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    if np.median(s[mask]) < 60:
        return np.zeros_like(mask)
    grey = mask & (s < 45) & (v < 150)
    grey = cv2.morphologyEx(_to_u8(grey), cv2.MORPH_OPEN, _disk(3))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(grey, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.004 * mask.sum()]
    blob = np.isin(cc, keep)
    return cv2.dilate(_to_u8(blob), _disk(4)).astype(bool) & mask if blob.any() else blob


# --------------------------------------------------------------------------- #
# Ring geometry: ellipse of the inner hole, polar profile of the outer edge
# --------------------------------------------------------------------------- #
def _ellipse_err(pts: np.ndarray, e) -> np.ndarray:
    (cx, cy), (w, h), ang = e
    a, b = w / 2.0, h / 2.0
    t = np.deg2rad(ang)
    c, s = np.cos(t), np.sin(t)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xr, yr = dx * c + dy * s, -dx * s + dy * c
    return np.abs(np.hypot(xr / a, yr / b) - 1.0) * np.sqrt(a * b)


def _ransac_ellipse(pts: np.ndarray, iters: int = 400):
    """Ellipse through the majority of the points. Outliers (clay bump, sprue, hull chord) are ignored."""
    if len(pts) > 1500:
        pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
    if len(pts) < 20:
        return None
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        try:
            e = cv2.fitEllipse(pts[rng.choice(len(pts), 8, replace=False)])
        except cv2.error:
            continue
        w, h = e[1]
        if min(w, h) < 20 or max(w, h) > 3.0 * min(w, h):
            continue
        n_in = int((_ellipse_err(pts, e) < 0.03 * np.sqrt(w * h / 4) + 2).sum())
        if n_in > best_n:
            best, best_n = e, n_in
    if best is None or best_n < 0.4 * len(pts):
        return None
    for _ in range(2):                                    # refit on the inliers
        inl = _ellipse_err(pts, best) < 0.03 * np.sqrt(best[1][0] * best[1][1] / 4) + 2
        if inl.sum() < 5:
            break
        best = cv2.fitEllipse(pts[inl])
    return best


def _hole_ellipse(m: np.ndarray):
    """Ellipse of the ring's inner opening. Uses the convex hull, so a ring with a gap
    (a C shape) still has a well-defined hole."""
    nz = cv2.findNonZero(_to_u8(m))
    if nz is None:
        return None
    hull = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(nz), 1)
    empty = (hull > 0) & ~m
    dt = cv2.distanceTransform(_to_u8(empty), cv2.DIST_L2, 5)
    seed = np.unravel_index(int(dt.argmax()), dt.shape)
    _, cc = cv2.connectedComponents(_to_u8(empty), connectivity=4)
    hole = cc == cc[seed]
    if hole.sum() < 0.1 * m.sum():
        return None
    contours, _ = cv2.findContours(_to_u8(hole), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len).reshape(-1, 2).astype(np.float32)
    return _ransac_ellipse(pts)


def _frame(e):
    (cx, cy), (w, h), ang = e
    t = np.deg2rad(ang)
    return cx, cy, w / 2.0, h / 2.0, np.cos(t), np.sin(t)


def _to_polar(xs, ys, fr):
    """Pixel -> (rho, theta) where rho = 1 on the hole edge."""
    cx, cy, a, b, c, s = fr
    dx, dy = xs - cx, ys - cy
    xr, yr = dx * c + dy * s, -dx * s + dy * c
    xn, yn = xr / a, yr / b
    return np.hypot(xn, yn), np.arctan2(yn, xn)


def _from_polar(rho, th, fr):
    cx, cy, a, b, c, s = fr
    xr, yr = a * rho * np.cos(th), b * rho * np.sin(th)
    return cx + xr * c - yr * s, cy + xr * s + yr * c


def _ring_profile(m: np.ndarray, fr, zone: np.ndarray | None = None,
                  rmax: float = 4.0, step: float = 0.01) -> np.ndarray:
    """For each direction: rho where the metal ends after leaving the hole. NaN = no metal found
    (a gap in the ring, or the clay hid it)."""
    th = np.arange(N_THETA) * (2 * np.pi / N_THETA)
    rho = np.arange(1.0, rmax, step)
    X, Y = _from_polar(rho[None, :], th[:, None], fr)
    xi, yi = np.rint(X).astype(int), np.rint(Y).astype(int)
    H, W = m.shape
    inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    hit = np.zeros(X.shape, bool)
    hit[inb] = m[yi[inb], xi[inb]]
    idx = np.arange(rho.size)[None, :]
    first = np.argmax(hit, axis=1)
    after = (~hit) & (idx >= first[:, None])
    exit_i = np.where(after.any(1), np.argmax(after, axis=1), rho.size - 1)
    unsure = np.zeros(N_THETA, bool)
    if zone is not None:                       # the edge ends right next to the clay: not the real ring edge
        ar = np.arange(N_THETA)
        unsure = zone[np.clip(yi[ar, exit_i], 0, H - 1), np.clip(xi[ar, exit_i], 0, W - 1)]
    return np.where(hit.any(1) & ~unsure, rho[exit_i], np.nan)


def _circ_fill_nan(v: np.ndarray):
    ok = ~np.isnan(v)
    if ok.sum() < 0.3 * len(v):
        return None
    n = len(v)
    x = np.arange(n)
    xx, vv, ok3 = np.concatenate([x - n, x, x + n]), np.tile(v, 3), np.tile(ok, 3)
    return np.interp(x, xx[ok3], vv[ok3])


def _circ_open(v: np.ndarray, w: int) -> np.ndarray:
    """Grey opening on a circular signal: removes bumps narrower than w samples."""
    n = len(v)
    t = np.tile(v.astype(np.float32), 3).reshape(1, -1)
    k = np.ones((1, w), np.uint8)
    return cv2.dilate(cv2.erode(t, k), k)[0, n:2 * n]


def _runs(flag: np.ndarray) -> list[np.ndarray]:
    """Contiguous True stretches of a circular boolean array, as index arrays."""
    n = len(flag)
    if flag.all() or not flag.any():
        return []
    start0 = int(np.argmin(flag))
    rot = np.roll(flag, -start0)
    out, i = [], 0
    while i < n:
        if rot[i]:
            j = i
            while j + 1 < n and rot[j + 1]:
                j += 1
            out.append((np.arange(i, j + 1) + start0) % n)
            i = j + 1
        else:
            i += 1
    return out


def _circ_dist(a_idx: np.ndarray, b_idx: np.ndarray) -> int:
    d = np.abs(a_idx[:, None] - b_idx[None, :])
    return int(np.minimum(d, N_THETA - d).min())


def _bridge(rho_f: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Straight bridge of the outer edge across `run`, from the ring edge a few degrees outside it."""
    side = np.arange(3, 9)
    left = np.median(rho_f[(run[0] - side) % N_THETA])
    right = np.median(rho_f[(run[-1] + side) % N_THETA])
    t = (np.arange(len(run)) + 1) / (len(run) + 1)
    return left + (right - left) * t


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def clean_mask(rgb: np.ndarray, mask: np.ndarray, *, remove_sprue: bool = True,
               remove_clay: bool = True, remove_loose: bool = True,
               fill_hidden: bool = True, require_marks: bool = True,
               excess: float = 0.15, open_deg: int = 140, mark_reach_deg: int = 45,
               tol: float = 0.025, min_part_frac: float = 0.15) -> CleanResult:
    """Clean `mask` (bool, same size as `rgb`).

    excess          how far (in hole radii) the outer edge must jump past the ring to count as sprue.
    open_deg        bumps narrower than this many degrees are treated as sprue candidates.
    mark_reach_deg  the sprue must start or end within this many degrees of a red mark.
    tol             extra margin kept past the bridged ring edge, in hole radii.
    require_marks   True: no red mark near the bump, no removal. Protects heads and stone settings.
    fill_hidden     rebuild the ring bottom where the clay covered it.
    """
    H, W = mask.shape
    s = min(1.0, WORK_SIDE / max(H, W))
    size = (max(1, round(W * s)), max(1, round(H * s)))
    small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    m0 = cv2.resize(_to_u8(mask), size, interpolation=cv2.INTER_NEAREST).astype(bool) if s < 1 else mask.copy()
    zero = np.zeros_like(m0)
    notes: list[str] = []

    clay = _find_clay(small_rgb, m0) if remove_clay else zero
    m = m0 & ~clay
    if clay.any():
        notes.append("grey blob removed")
    if clay.any():                              # dark shadow rim left around the clay counts as clay too
        hsv0 = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        grey_rim = (hsv0[..., 1] < 55) & (hsv0[..., 2] < 170)
        dark_rim = hsv0[..., 2] < 105                            # clay shadow, even when it picked up a brown tint
        rim = (cv2.dilate(_to_u8(clay), _disk(18)) > 0) & m & grey_rim
        rim |= (cv2.dilate(_to_u8(clay), _disk(30)) > 0) & m & dark_rim
        clay = clay | rim
        m = m & ~clay
    clay_zone = (cv2.dilate(_to_u8(clay), _disk(8)) > 0) if clay.any() else None
    marks = find_cut_marks(small_rgb, m0)

    loose = zero
    if remove_loose:
        kept = _drop_small_parts(m, min_part_frac)
        loose = m & ~kept
        m = kept
        if loose.any():
            notes.append("loose fragments removed")

    sprue, fill = zero.copy(), zero.copy()
    fill_rgb = None
    ring_ok_w, fr_saved = None, None
    m_geo = m0 & ~loose                      # with the clay still in: it hides the hole edge less than a gap does
    e = _hole_ellipse(m_geo) if (remove_sprue or (fill_hidden and clay.any())) else None
    if e is None:
        if remove_sprue:
            notes.append("ring hole not found, sprue kept")
    else:
        fr = _frame(e)
        fr_saved = fr
        rho_raw = _ring_profile(m, fr, clay_zone)
        rho_f = _circ_fill_nan(rho_raw)
        if rho_f is None:
            notes.append("ring edge not found, sprue kept")
        else:
            yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(np.float32)
            rho_pix, th_pix = _to_polar(xx, yy, fr)
            th_idx = (np.floor((th_pix % (2 * np.pi)) / (2 * np.pi) * N_THETA).astype(int)) % N_THETA
            rho_final = rho_f.copy()
            wedge_idx = None

            if remove_sprue:
                base = _circ_open(rho_f, open_deg)
                runs = [r for r in _runs(rho_f - base > excess) if 10 <= len(r) <= 170]
                mark_idx = np.array([], int)
                if marks.any():
                    my, mx = np.nonzero(marks)
                    _, mth = _to_polar(mx.astype(np.float32), my.astype(np.float32), fr)
                    mark_idx = np.unique((np.rint((mth % (2 * np.pi)) / (2 * np.pi) * N_THETA).astype(int)) % N_THETA)
                pick = None
                if runs and mark_idx.size:
                    d = [_circ_dist(r, mark_idx) for r in runs]
                    if min(d) <= mark_reach_deg:
                        pick = runs[int(np.argmin(d))]
                elif runs and not require_marks:
                    pick = max(runs, key=lambda r: float((rho_f[r] - base[r]).sum()))
                if pick is None:
                    notes.append("no sprue next to a red mark" if marks.any()
                                 else "no red cut marks found, sprue kept")
                else:
                    ext = np.arange(pick[0] - 3, pick[-1] + 4) % N_THETA
                    wide = np.arange(pick[0] - 7, pick[-1] + 8) % N_THETA     # edge next to the sprue is unreliable
                    unknown = np.isnan(rho_raw)
                    unknown[wide] = True
                    redo = _circ_fill_nan(np.where(unknown, np.nan, rho_raw))
                    if redo is not None:
                        rho_final = np.minimum(rho_f, redo) if False else redo
                    wedge_idx = np.zeros(N_THETA, bool)
                    wedge_idx[ext] = True

            rho_b = np.interp(((th_pix % (2 * np.pi)) / (2 * np.pi) * N_THETA).ravel(),
                              np.arange(N_THETA + 1), np.append(rho_final, rho_final[0])).reshape(rho_pix.shape)
            region = rho_pix <= rho_b + tol            # the ring as it should be, bottom included

            if wedge_idx is not None:
                extra = m & ~region
                n_c, cc = cv2.connectedComponents(_to_u8(extra), connectivity=8)
                counts = np.bincount(cc[wedge_idx[th_idx] & extra], minlength=n_c)
                counts[0] = 0
                k = int(counts.argmax())
                if counts[k] >= 0.01 * m.sum():
                    cand = cc == k
                    after = _drop_small_parts(m & ~cand, 0.03)
                    if cand.sum() <= 0.65 * m.sum() and after.sum() >= 0.3 * m.sum():
                        sprue, m = m & ~after, after
                        notes.append("sprue removed, ring kept whole")
                    else:
                        notes.append("sprue candidate rejected (too large a share of the piece)")
                else:
                    notes.append("no sprue metal outside the ring edge")

            if fill_hidden and clay.any():
                cand = clay & region & (rho_pix >= 0.98) & ~m
                if cand.sum() >= 30:
                    sat = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)[..., 1]
                    near_clay = cv2.dilate(_to_u8(clay), _disk(10)) > 0       # shadow and halo around the clay
                    ring_ok = m & region & ~cand & ~near_clay & (sat >= 55)    # real, coloured metal only
                    ring_ok_w = ring_ok
                    ys, xs = np.nonzero(cand)
                    rho_c, th_c = rho_pix[ys, xs], th_pix[ys, xs]
                    out_rgb = np.zeros((len(ys), 3), np.uint8)
                    done = np.zeros(len(ys), bool)
                    for d in np.deg2rad(np.arange(1, 70, 1.0)):
                        for sgn in (1, -1):
                            sx, sy = _from_polar(rho_c, th_c + sgn * d, fr)
                            sx, sy = np.rint(sx).astype(int), np.rint(sy).astype(int)
                            ok = (sx >= 0) & (sx < m.shape[1]) & (sy >= 0) & (sy < m.shape[0]) & ~done
                            ok[ok] = ring_ok[sy[ok], sx[ok]]
                            out_rgb[ok] = small_rgb[sy[ok], sx[ok]]
                            done |= ok
                    if done.any():
                        fill[ys[done], xs[done]] = True
                        m = m | fill
                        fill_work = small_rgb.copy()
                        fill_work[ys[done], xs[done]] = out_rgb[done]
                        fill_rgb = fill_work
                        notes.append("ring bottom hidden by the clay rebuilt from neighbouring metal")

    def up(a: np.ndarray) -> np.ndarray:           # blocky, fine for reporting only
        if s >= 1:
            return a
        return cv2.resize(_to_u8(a), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    def up_soft(a: np.ndarray) -> np.ndarray:      # smooth edge, used for the cut lines
        if s >= 1:
            return a
        return cv2.resize(_to_u8(a) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127

    fill_full = up_soft(fill) & mask
    rgb_out = None
    if fill_full.any() and s < 1 and ring_ok_w is not None:
        # Rebuild the hidden stretch from NATIVE photo pixels (same ring, shifted along the circle),
        # so it keeps the photo's real grain instead of a blurry upscaled patch.
        ok_full = (cv2.resize(_to_u8(ring_ok_w) * 255, (W, H), interpolation=cv2.INTER_LINEAR) > 127) & ~fill_full
        ys, xs = np.nonzero(fill_full)
        rho_c, th_c = _to_polar((xs + 0.5) * s - 0.5, (ys + 0.5) * s - 0.5, fr_saved)
        out = np.zeros((len(ys), 3), np.uint8)
        done = np.zeros(len(ys), bool)
        for d in np.deg2rad(np.arange(1, 70, 0.5)):
            for sgn in (1, -1):
                sx, sy = _from_polar(rho_c, th_c + sgn * d, fr_saved)
                sx = np.rint((sx + 0.5) / s - 0.5).astype(int)
                sy = np.rint((sy + 0.5) / s - 0.5).astype(int)
                ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H) & ~done
                ok[ok] = ok_full[sy[ok], sx[ok]]
                out[ok] = rgb[sy[ok], sx[ok]]
                done |= ok
        fill_full = np.zeros((H, W), bool)
        fill_full[ys[done], xs[done]] = True
        if done.any():
            rgb_out = rgb.copy()
            rgb_out[ys[done], xs[done]] = out[done]
    elif fill_rgb is not None and fill_full.any():
        rgb_out = rgb.copy()
        rgb_out[fill_full] = fill_rgb[fill_full]

    # Only SUBTRACT what was cut from the original full-resolution mask. Everything that was not
    # cut keeps the exact edge it had in the photo, so the ring is never eroded by the rescaling.
    cut_full = up_soft(m0 & ~(m & ~fill)) & mask      # everything cut, including the stretch we then rebuild
    final = (mask & ~cut_full) | fill_full

    hw = _band_half_width(m)
    return CleanResult(final, up_soft(sprue) & mask, up_soft(clay) & mask, up_soft(loose) & mask, up(marks),
                       fill_full, mask.copy(), rgb_out, 2 * hw / s, "; ".join(notes) or "nothing to remove")
