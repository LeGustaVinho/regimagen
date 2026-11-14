"""Preserve the widths of repeated material joints where an opaque repair meets its context."""

import cv2
import numpy as np


def _profiles(lab):
    light, a, b = np.moveaxis(lab, -1, 0)
    chroma = np.hypot(a, b)
    # Favor neutral material and retain darker joints where they are still distinct.
    material = np.maximum(0, chroma - light * 0.12)
    return cv2.GaussianBlur(material, (0, 0), sigmaX=8, sigmaY=0.75)


def _valleys(profile, radius):
    line = profile[:, None]
    upper = cv2.dilate(line, np.ones((2 * radius + 1, 1), np.uint8)).ravel()
    troughs = np.flatnonzero(
        (profile[1:-1] <= profile[:-2]) & (profile[1:-1] < profile[2:])
    ) + 1
    troughs = troughs[(troughs >= radius) & (troughs < len(profile) - radius)]
    found = []
    for y in troughs:
        left = profile[y - radius:y].max()
        right = profile[y + 1:y + radius + 1].max()
        shoulder = min(left, right)
        contrast = shoulder - profile[y]
        if contrast < 10 or upper[y] - profile[y] < 14:
            continue
        level = profile[y] + 0.62 * contrast
        before = np.flatnonzero(profile[y - radius:y] >= level)
        after = np.flatnonzero(profile[y + 1:y + radius + 1] >= level)
        if before.size and after.size:
            low = y - radius + before[-1] + 0.5
            high = y + 1 + after[0] - 0.5
            if 2 <= high - low <= radius * 1.35:
                found.append((float(y), low, high, float(contrast)))
    # Adjacent minima in the same joint describe a single band.
    result = []
    for band in found:
        if result and band[1] <= result[-1][2]:
            if band[-1] > result[-1][-1]:
                result[-1] = band
        else:
            result.append(band)
    return result


def _periodic_courses(profile, support):
    usable = support >= 0.95
    if np.count_nonzero(usable) < 64:
        return []
    courses = [band for band in _valleys(profile, 14)
               if usable[int(band[1]) - 2:int(band[2]) + 3].all()]
    if len(courses) < 4:
        return []
    spacing = np.diff([band[0] for band in courses])
    typical = np.median(spacing)
    if typical < 14 or np.count_nonzero(abs(spacing - typical) <= typical * 0.28) < max(3, len(spacing) * 0.55):
        return []
    return courses


def _neighbor_widths(anchors, left, right, peers, y):
    widths = []
    for side in (left, right):
        values = []
        for sample in side:
            match = min(anchors[sample], key=lambda band: abs(band[0] - y))
            if abs(match[0] - y) <= 10:
                values.append(match[2] - match[1])
        if values:
            widths.append(np.median(values))
    if len(widths) == 2:
        return widths

    # A repair can cover a joint end to end. Compare the next intact courses on
    # either side of that row so a paint band across the canvas still has a reference.
    neighbors = []
    for sample in peers:
        above, below = [], []
        for center, low, high, _ in anchors[sample]:
            distance = center - y
            if -112 <= distance <= -18:
                above.append(high - low)
            elif 18 <= distance <= 112:
                below.append(high - low)
        if above and below:
            neighbors.append((above[-1], below[0]))
    return np.median(neighbors, axis=0) if len(neighbors) >= 3 else ()


def _course_map(reference, candidate, selection):
    height, width = selection.shape
    source, target = _profiles(reference), _profiles(candidate)
    support = cv2.GaussianBlur((~selection).astype(np.float32), (0, 0), sigmaX=8, sigmaY=0.75)
    anchors = {}
    for x in range(12, width - 12, 8):
        bands = _periodic_courses(source[:, x], support[:, x])
        if bands:
            anchors[x] = bands
    if len(anchors) < 5:
        return None
    modified = np.zeros((height, width), np.float32)
    weight = np.zeros_like(modified)
    for x in range(12, width - 12, 8):
        if not selection[:, x].any():
            continue
        nearby = sorted((abs(s - x), s) for s in anchors if abs(s - x) <= 104)
        left = [s for distance, s in nearby if distance >= 12 and s < x][:3]
        right = [s for distance, s in nearby if distance >= 12 and s > x][:3]
        peers = [s for distance, s in nearby if distance <= 24][:5]
        for y, low, high, contrast in _valleys(target[:, x], 14):
            center = int(y)
            if not selection[center, x] or contrast < 13:
                continue
            widths = _neighbor_widths(anchors, left, right, peers, y)
            if len(widths) != 2 or max(widths) > 2.1 * min(widths):
                continue
            wanted = float(np.mean(widths))
            actual = high - low
            if abs(actual - wanted) < 1.3 or not 0.3 <= actual / wanted <= 2.3:
                continue
            center_y = (low + high) / 2
            half = wanted / 2
            outer = max(10, half + 6, actual / 2 + 3)
            start, stop = max(0, int(center_y - outer)), min(height, int(center_y + outer) + 1)
            rows = np.arange(start, stop)
            offset = rows - center_y
            limits = [-outer, -half, 0, half, outer]
            mapped = np.interp(offset, limits, [-outer, -actual / 2, 0, actual / 2, outer]) - offset
            strength = (1 - np.clip(abs(offset) / outer, 0, 1) ** 2).astype(np.float32)
            modified[start:stop, x] += mapped * strength
            weight[start:stop, x] += strength
    if not weight.any():
        return None
    # Interpolate between sampled columns and fade where neighboring profiles disagree.
    field = cv2.GaussianBlur(modified, (0, 0), sigmaX=6, sigmaY=1)
    support = cv2.GaussianBlur(weight, (0, 0), sigmaX=6, sigmaY=1)
    return np.clip(field / np.maximum(support, 0.027), -3.5, 3.5) * selection


def match_course_widths(reference, aligned, selected):
    """Warp only the approved repair when neighboring rows establish a reliable width."""
    if min(selected.shape) < 72 or not selected.any():
        return aligned
    source, candidate = (cv2.cvtColor(pixels.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
                         for pixels in (reference, aligned))
    dy = _course_map(source, candidate, selected)
    dx = _course_map(source.transpose(1, 0, 2), candidate.transpose(1, 0, 2), selected.T)
    if dx is None and dy is None:
        return aligned
    height, width = selected.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    if dx is not None:
        x += dx.T
    if dy is not None:
        y += dy
    return cv2.remap(aligned, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
