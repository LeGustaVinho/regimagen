"""Recover sprite edges after editing against a charcoal matte."""

from math import ceil

import cv2
import numpy as np
from PIL import Image

from .files import TRANSPARENT_MATTE
from .geometry import Rect


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def _sample_positions(start: int, end: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    positions = (np.arange(start, end) + 0.5) * scale - 0.5
    lower = np.floor(positions).astype(np.intp)
    return lower, (positions - lower).astype(np.float32)


def _cubic_weights(t: np.ndarray) -> tuple[np.ndarray, ...]:
    square = t * t
    cube = square * t
    return (
        -0.5 * t + square - 0.5 * cube,
        1 - 2.5 * square + 1.5 * cube,
        0.5 * t + 2 * square - 1.5 * cube,
        -0.5 * square + 0.5 * cube,
    )


class SpriteMatte:
    def __init__(self, source: Image.Image, size: tuple[int, int]):
        self.source = np.asarray(source) if source.mode == "RGBA" else None
        self.alpha = source.getchannel("A") if self.source is not None else source
        self.samples = np.asarray(self.alpha)
        self.thin_detail = None
        if self.source is not None:
            neighborhood = np.ones((3, 3), dtype=np.uint8)
            opened = cv2.morphologyEx(self.samples, cv2.MORPH_OPEN, neighborhood)
            detail = cv2.subtract(self.samples, opened)
            detail_peak = cv2.dilate(detail, neighborhood)
            self.thin_detail = detail * (1 - _smoothstep(125, 195, detail_peak))
        self.scale_x = self.alpha.width / size[0]
        self.scale_y = self.alpha.height / size[1]
        scale = 1 / max(self.scale_x, self.scale_y)
        self.radius = min(64, max(2, ceil(2.6 * scale)))
        self.ridge_radius = max(1, ceil(self.radius / 4))
        self.padding = self.radius + 2

    def _box(self, area: Rect) -> tuple[float, float, float, float]:
        return (
            area.left * self.scale_x, area.top * self.scale_y,
            area.right * self.scale_x, area.bottom * self.scale_y,
        )

    def coverage(self, area: Rect) -> np.ndarray:
        if max(self.scale_x, self.scale_y) > 1:
            return np.array(self.alpha.resize(
                (area.width, area.height), Image.Resampling.BILINEAR, box=self._box(area),
            ))

        # Global sample coordinates keep tile and stripe renders consistent.
        x, tx = _sample_positions(area.left, area.right, self.scale_x)
        y, ty = _sample_positions(area.top, area.bottom, self.scale_y)
        xs = [np.clip(x + shift, 0, self.alpha.width - 1) for shift in (-1, 0, 1, 2)]
        ys = [np.clip(y + shift, 0, self.alpha.height - 1) for shift in (-1, 0, 1, 2)]
        top, bottom = ys[0][0], ys[-1][-1] + 1
        horizontal = sum(self.samples[top:bottom, indices] * weight
                         for indices, weight in zip(xs, _cubic_weights(tx), strict=True))
        alpha = sum(horizontal[indices - top, :] * weight[:, None]
                    for indices, weight in zip(ys, _cubic_weights(ty), strict=True))
        # Clamp the cubic samples to local bounds to avoid halos or new opacity.
        a, b = self.samples[ys[1][:, None], xs[1]], self.samples[ys[1][:, None], xs[2]]
        c, d = self.samples[ys[2][:, None], xs[1]], self.samples[ys[2][:, None], xs[2]]
        lower = np.minimum(np.minimum(a, b), np.minimum(c, d))
        upper = np.maximum(np.maximum(a, b), np.maximum(c, d))
        return np.rint(np.clip(alpha, lower, upper)).astype(np.uint8)

    def _foreground(self, area: Rect, edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.source is not None and self.thin_detail is not None
        rows, columns = np.nonzero(edge)
        x, tx = _sample_positions(area.left, area.right, self.scale_x)
        y, ty = _sample_positions(area.top, area.bottom, self.scale_y)
        x, tx = x[columns], tx[columns]
        y, ty = y[rows], ty[rows]
        color = np.zeros((len(rows), 3), dtype=np.float32)
        alpha = np.zeros(len(rows), dtype=np.float32)
        detail = np.zeros(len(rows), dtype=np.float32)
        # Interpolating premultiplied RGB in floating point excludes fully transparent colors.
        for dy, wy in enumerate((1 - ty, ty)):
            for dx, wx in enumerate((1 - tx, tx)):
                coordinates = (
                    np.clip(y + dy, 0, self.alpha.height - 1),
                    np.clip(x + dx, 0, self.alpha.width - 1),
                )
                rgba = self.source[coordinates]
                weight = wy * wx
                premultiplied_weight = rgba[:, 3] * weight
                color += rgba[:, :3] * premultiplied_weight[:, None]
                alpha += premultiplied_weight
                detail += self.thin_detail[coordinates] * weight
        np.divide(color, alpha[:, None], out=color, where=alpha[:, None] > 0)
        np.divide(detail, alpha, out=detail, where=alpha > 0)
        return color, detail

    def refine(self, area: Rect, pixels: np.ndarray, coverage: np.ndarray, missing: np.ndarray):
        exterior = coverage <= 3
        if not np.any(coverage) or not np.any(exterior):
            return
        distance = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
        edge = (coverage > 0) & (distance < self.radius) & ~missing
        if not np.any(edge):
            return

        matte = np.array(TRANSPARENT_MATTE, dtype=np.int16)
        difference = np.abs(pixels.astype(np.int16) - matte).max(axis=2)
        trusted = (coverage > 24) & (difference >= 24) & ~missing
        if self.source is None:
            self._refine_without_source(pixels, edge, difference, trusted)
            return

        reference, detail = self._foreground(area, edge)
        original = reference - matte
        generated = pixels[edge].astype(np.float32) - matte
        opacity = coverage[edge].astype(np.float32) / 255
        edge_weight = 1 - _smoothstep(self.radius * 0.75, self.radius, distance[edge])
        contrast = np.abs(original).max(axis=1)
        near_matte = 1 - _smoothstep(3, 17, difference[edge])
        strength = _smoothstep(7, 20, contrast) * edge_weight

        if np.any(trusted):
            distance_to_material = cv2.distanceTransform(
                (~trusted).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
            )[edge]
            support = 1 - _smoothstep(self.radius * 0.85, self.radius, distance_to_material)
            ridge_kernel = np.ones((2 * self.ridge_radius + 1,) * 2, dtype=np.uint8)
            ridge = cv2.dilate(coverage, ridge_kernel)[edge]
            peak = cv2.dilate(coverage, np.ones((2 * self.radius + 1,) * 2, dtype=np.uint8))[edge]
            relative_opacity = coverage[edge].astype(np.float32) / ridge
            # The smaller window protects the center of translucent strokes next to opaque parts.
            fringe = (1 - _smoothstep(0.74, 0.98, relative_opacity)) * _smoothstep(110, 220, peak)
            fringe *= 1 - _smoothstep(0.78, 0.94, opacity)
            background = cv2.GaussianBlur(difference.astype(np.float32), (5, 5), 1)[edge]
            flat_matte = (1 - _smoothstep(2, 6, background)) * _smoothstep(4, 10, contrast)
            expected_color = _smoothstep(3, 11, opacity * contrast)
            # A translucent source detail may be next to an opaque part.
            broad_shape = 1 - _smoothstep(0.18, 0.65, detail)
            residue = 0.97 * np.maximum(flat_matte, expected_color) * near_matte
            residue *= edge_weight * support * fringe * broad_shape
            coverage[edge] = np.rint(coverage[edge] * (1 - residue)).astype(np.uint8)

        # Project the observed color onto the matte-to-foreground segment to recover the
        # edge color without dividing by a small alpha or discarding generated texture.
        projection = (generated * original).sum(axis=1) / np.maximum((original ** 2).sum(axis=1), 1)
        mixture = np.clip(projection, 0, 1)
        residual = np.abs(generated - mixture[:, None] * original).max(axis=1)
        compatible = 1 - _smoothstep(6, 18, residual)
        soft = 1 - _smoothstep(0.70, 0.98, opacity)
        restore = strength * np.maximum(near_matte, soft * compatible) * (1 - mixture)
        pixels[edge] = np.rint(np.clip(pixels[edge] + restore[:, None] * original, 0, 255)).astype(np.uint8)

    def _refine_without_source(self, pixels, edge, difference, trusted):
        # An alpha mask alone can help recover nearby RGB values. It has no source color.
        candidates = edge & (difference <= 13)
        if not np.any(candidates) or not np.any(trusted):
            return
        kernel = (2 * self.radius + 1,) * 2
        weight = cv2.boxFilter(trusted.astype(np.float32), -1, kernel)[candidates]
        color = cv2.boxFilter(pixels.astype(np.float32) * trusted[:, :, None], -1, kernel)[candidates]
        current = pixels[candidates]
        recovered = np.divide(color, weight[:, None], out=current.astype(np.float32), where=weight[:, None] > 0)
        mix = (1 - _smoothstep(6, 14, difference[candidates]))[:, None]
        pixels[candidates] = np.rint(current + mix * (recovered - current)).astype(np.uint8)
