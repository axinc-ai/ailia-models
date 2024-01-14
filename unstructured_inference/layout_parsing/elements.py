from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Collection, Optional, Union

import numpy as np
from constants import Source


def safe_division(a, b) -> float:
    """a safer division to avoid division by zero when b == 0

    returns a/b or a/FLOAT_EPSILON (should be around 2.2E-16) when b == 0

    Parameters:
    - a (int/float): a in a/b
    - b (int/float): b in a/b

    Returns:
    float: a/b or a/FLOAT_EPSILON (should be around 2.2E-16) when b == 0
    """
    return a / max(b, FLOAT_EPSILON)


@dataclass
class Rectangle:
    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]

    def pad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the boundary outward (resp. inward)."""
        out_object = self.hpad(padding).vpad(padding)
        return out_object

    def hpad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the left and right sides of the boundary outward (resp. inward)."""
        out_object = deepcopy(self)
        out_object.x1 -= padding
        out_object.x2 += padding
        return out_object

    def vpad(self, padding: Union[int, float]):
        """Increases (or decreases, if padding is negative) the size of the rectangle by extending
        the top and bottom of the boundary outward (resp. inward)."""
        out_object = deepcopy(self)
        out_object.y1 -= padding
        out_object.y2 += padding
        return out_object

    @property
    def width(self) -> Union[int, float]:
        """Width of rectangle"""
        return self.x2 - self.x1

    @property
    def height(self) -> Union[int, float]:
        """Height of rectangle"""
        return self.y2 - self.y1

    @property
    def x_midpoint(self) -> Union[int, float]:
        """Finds the horizontal midpoint of the object."""
        return (self.x2 + self.x1) / 2

    @property
    def y_midpoint(self) -> Union[int, float]:
        """Finds the vertical midpoint of the object."""
        return (self.y2 + self.y1) / 2

    def is_disjoint(self, other: Rectangle) -> bool:
        """Checks whether this rectangle is disjoint from another rectangle."""
        return not self.intersects(other)

    def intersects(self, other: Rectangle) -> bool:
        """Checks whether this rectangle intersects another rectangle."""
        if self._has_none() or other._has_none():
            return False
        return intersections(self, other)[0, 1]

    def is_in(
        self, other: Rectangle, error_margin: Optional[Union[int, float]] = None
    ) -> bool:
        """Checks whether this rectangle is contained within another rectangle."""
        padded_other = other.pad(error_margin) if error_margin is not None else other
        return all(
            [
                (self.x1 >= padded_other.x1),
                (self.x2 <= padded_other.x2),
                (self.y1 >= padded_other.y1),
                (self.y2 <= padded_other.y2),
            ],
        )

    def _has_none(self) -> bool:
        """return false when one of the coord is nan"""
        return any((self.x1 is None, self.x2 is None, self.y1 is None, self.y2 is None))

    @property
    def coordinates(self):
        """Gets coordinates of the rectangle"""
        return (
            (self.x1, self.y1),
            (self.x1, self.y2),
            (self.x2, self.y2),
            (self.x2, self.y1),
        )

    def intersection(self, other: Rectangle) -> Optional[Rectangle]:
        """Gives the rectangle that is the intersection of two rectangles, or None if the
        rectangles are disjoint."""
        if self._has_none() or other._has_none():
            return None
        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        y1 = max(self.y1, other.y1)
        y2 = min(self.y2, other.y2)
        if x1 > x2 or y1 > y2:
            return None
        return Rectangle(x1, y1, x2, y2)

    @property
    def area(self) -> float:
        """Gives the area of the rectangle."""
        return self.width * self.height

    def intersection_over_union(self, other: Rectangle) -> float:
        """Gives the intersection-over-union of two rectangles. This tends to be a good metric of
        how similar the regions are. Returns 0 for disjoint rectangles, 1 for two identical
        rectangles -- area of intersection / area of union."""
        intersection = self.intersection(other)
        intersection_area = 0.0 if intersection is None else intersection.area
        union_area = self.area + other.area - intersection_area
        return safe_division(intersection_area, union_area)

    def intersection_over_minimum(self, other: Rectangle) -> float:
        """Gives the area-of-intersection over the minimum of the areas of the rectangles. Useful
        for identifying when one rectangle is almost-a-subset of the other. Returns 0 for disjoint
        rectangles, 1 when either is a subset of the other."""
        intersection = self.intersection(other)
        intersection_area = 0.0 if intersection is None else intersection.area
        min_area = min(self.area, other.area)
        return safe_division(intersection_area, min_area)

    def is_almost_subregion_of(
        self, other: Rectangle, subregion_threshold: float = 0.75
    ) -> bool:
        """Returns whether this region is almost a subregion of other. This is determined by
        comparing the intersection area over self area to some threshold, and checking whether self
        is the smaller rectangle."""
        intersection = self.intersection(other)
        intersection_area = 0.0 if intersection is None else intersection.area
        return (subregion_threshold < safe_division(intersection_area, self.area)) and (
            self.area <= other.area
        )


def minimal_containing_region(*regions: Rectangle) -> Rectangle:
    """Returns the smallest rectangular region that contains all regions passed"""
    x1 = min(region.x1 for region in regions)
    y1 = min(region.y1 for region in regions)
    x2 = max(region.x2 for region in regions)
    y2 = max(region.y2 for region in regions)

    return Rectangle(x1, y1, x2, y2)


def intersections(*rects: Rectangle):
    """Returns a square boolean matrix of intersections of an arbitrary number of rectangles, i.e.
    the ijth entry of the matrix is True if and only if the ith Rectangle and jth Rectangle
    intersect."""
    # NOTE(alan): Rewrite using line scan
    coords = np.stack([[[r.x1, r.y1], [r.x2, r.y2]] for r in rects], axis=-1)

    (x1s, y1s), (x2s, y2s) = coords

    # Use broadcasting to get comparison matrices.
    # For Rectangles r1 and r2, any of the following conditions makes the rectangles disjoint:
    # r1.x1 > r2.x2
    # r1.y1 > r2.y2
    # r2.x1 > r1.x2
    # r2.y1 > r1.y2
    # Then we take the complement (~) of the disjointness matrix to get the intersection matrix.
    intersections = ~(
        (x1s[None] > x2s[..., None])
        | (y1s[None] > y2s[..., None])
        | (x1s[None] > x2s[..., None]).T
        | (y1s[None] > y2s[..., None]).T
    )

    return intersections


@dataclass
class TextRegion:
    bbox: Rectangle
    text: Optional[str] = None
    source: Optional[Source] = None

    def __str__(self) -> str:
        return str(self.text)

    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
    ) -> str:
        """Extracts text contained in region."""
        if self.text is not None:
            # If block text is already populated, we'll assume it's correct
            text = self.text
        elif objects is not None:
            text = aggregate_by_block(self, objects)
        else:
            text = ""
        cleaned_text = remove_control_characters(text)
        return cleaned_text

    @classmethod
    def from_coords(
        cls,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        text: Optional[str] = None,
        source: Optional[Source] = None,
        **kwargs,
    ) -> TextRegion:
        """Constructs a region from coordinates."""
        bbox = Rectangle(x1, y1, x2, y2)

        return cls(text=text, source=source, bbox=bbox, **kwargs)


class EmbeddedTextRegion(TextRegion):
    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
    ) -> str:
        """Extracts text contained in region."""
        if self.text is None:
            return ""
        else:
            return self.text


class ImageTextRegion(TextRegion):
    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
    ) -> str:
        """Extracts text contained in region."""
        if self.text is None:
            return ""
        else:
            return super().extract_text(objects)


def aggregate_by_block(
    text_region: TextRegion,
    pdf_objects: Collection[TextRegion],
) -> str:
    """Extracts the text aggregated from the elements of the given layout that lie within the given
    block."""
    filtered_blocks = [
        obj for obj in pdf_objects if obj.bbox.is_in(text_region.bbox, error_margin=5)
    ]
    text = " ".join([x.text for x in filtered_blocks if x.text])
    return text


def cid_ratio(text: str) -> float:
    """Gets ratio of unknown 'cid' characters extracted from text to all characters."""
    if not is_cid_present(text):
        return 0.0
    cid_pattern = r"\(cid\:(\d+)\)"
    unmatched, n_cid = re.subn(cid_pattern, "", text)
    total = n_cid + len(unmatched)
    return n_cid / total


def is_cid_present(text: str) -> bool:
    """Checks if a cid code is present in a text selection."""
    if len(text) < len("(cid:x)"):
        return False
    return text.find("(cid:") != -1


def remove_control_characters(text: str) -> str:
    """Removes control characters from text."""

    # Replace newline character with a space
    text = text.replace("\n", " ")
    # Remove other control characters
    out_text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return out_text


def region_bounding_boxes_are_almost_the_same(
    region1: Rectangle,
    region2: Rectangle,
    same_region_threshold: float = 0.75,
) -> bool:
    """Returns whether bounding boxes are almost the same. This is determined by checking if the
    intersection over union is above some threshold."""
    return region1.intersection_over_union(region2) > same_region_threshold


def grow_region_to_match_region(region_to_grow: Rectangle, region_to_match: Rectangle):
    """Grows a region to the minimum size necessary to contain both regions."""
    (new_x1, new_y1), _, (new_x2, new_y2), _ = minimal_containing_region(
        region_to_grow,
        region_to_match,
    ).coordinates
    region_to_grow.x1, region_to_grow.y1, region_to_grow.x2, region_to_grow.y2 = (
        new_x1,
        new_y1,
        new_x2,
        new_y2,
    )
