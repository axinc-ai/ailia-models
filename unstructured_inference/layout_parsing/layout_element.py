from dataclasses import dataclass
from typing import Collection, List, Optional

import numpy as np
from config import inference_config
from constants import (CHIPPER_VERSIONS, FULL_PAGE_REGION_THRESHOLD,
                       ElementType, Source)
from layoutparser.elements.layout import TextBlock
from pandas import DataFrame
from scipy.sparse.csgraph import connected_components

from elements import (ImageTextRegion, Rectangle, TextRegion,
                      grow_region_to_match_region, intersections,
                      region_bounding_boxes_are_almost_the_same)


@dataclass
class LayoutElement(TextRegion):
    type: Optional[str] = None
    prob: Optional[float] = None
    image_path: Optional[str] = None
    parent: Optional[LayoutElement] = None

    def extract_text(
        self,
        objects: Optional[Collection[TextRegion]],
    ):
        """Extracts text contained in region"""
        text = super().extract_text(
            objects=objects,
        )
        return text

    def to_dict(self) -> dict:
        """Converts the class instance to dictionary form."""
        out_dict = {
            "coordinates": None if self.bbox is None else self.bbox.coordinates,
            "text": self.text,
            "type": self.type,
            "prob": self.prob,
            "source": self.source,
        }
        return out_dict

    @classmethod
    def from_region(cls, region: TextRegion):
        """Create LayoutElement from superclass."""
        text = region.text if hasattr(region, "text") else None
        type = region.type if hasattr(region, "type") else None
        prob = region.prob if hasattr(region, "prob") else None
        source = region.source if hasattr(region, "source") else None
        return cls(text=text, source=source, type=type, prob=prob, bbox=region.bbox)

    @classmethod
    def from_lp_textblock(cls, textblock: TextBlock):
        """Create LayoutElement from layoutparser TextBlock object."""
        x1, y1, x2, y2 = textblock.coordinates
        text = textblock.text
        type = textblock.type
        prob = textblock.score
        return cls.from_coords(
            x1,
            y1,
            x2,
            y2,
            text=text,
            source=Source.DETECTRON2_LP,
            type=type,
            prob=prob,
        )


def merge_inferred_layout_with_extracted_layout(
    inferred_layout: Collection[LayoutElement],
    extracted_layout: Collection[TextRegion],
    page_image_size: tuple,
    same_region_threshold: float = inference_config.LAYOUT_SAME_REGION_THRESHOLD,
    subregion_threshold: float = inference_config.LAYOUT_SUBREGION_THRESHOLD,
) -> List[LayoutElement]:
    """Merge two layouts to produce a single layout."""
    extracted_elements_to_add: List[TextRegion] = []
    inferred_regions_to_remove = []
    w, h = page_image_size
    full_page_region = Rectangle(0, 0, w, h)
    for extracted_region in extracted_layout:
        extracted_is_image = isinstance(extracted_region, ImageTextRegion)
        if extracted_is_image:
            # Skip extracted images for this purpose, we don't have the text from them and they
            # don't provide good text bounding boxes.

            is_full_page_image = region_bounding_boxes_are_almost_the_same(
                extracted_region.bbox,
                full_page_region,
                FULL_PAGE_REGION_THRESHOLD,
            )

            if is_full_page_image:
                continue
        region_matched = False
        for inferred_region in inferred_layout:
            if inferred_region.source in CHIPPER_VERSIONS:
                continue

            if inferred_region.bbox.intersects(extracted_region.bbox):
                same_bbox = region_bounding_boxes_are_almost_the_same(
                    inferred_region.bbox,
                    extracted_region.bbox,
                    same_region_threshold,
                )
                inferred_is_subregion_of_extracted = inferred_region.bbox.is_almost_subregion_of(
                    extracted_region.bbox,
                    subregion_threshold=subregion_threshold,
                )
                inferred_is_text = inferred_region.type not in (
                    ElementType.FIGURE,
                    ElementType.IMAGE,
                    ElementType.PAGE_BREAK,
                    ElementType.TABLE,
                )
                extracted_is_subregion_of_inferred = extracted_region.bbox.is_almost_subregion_of(
                    inferred_region.bbox,
                    subregion_threshold=subregion_threshold,
                )
                either_region_is_subregion_of_other = (
                    inferred_is_subregion_of_extracted or extracted_is_subregion_of_inferred
                )
                if same_bbox:
                    # Looks like these represent the same region
                    if extracted_is_image:
                        # keep extracted region, remove inferred region
                        inferred_regions_to_remove.append(inferred_region)
                    else:
                        # keep inferred region, remove extracted region
                        grow_region_to_match_region(inferred_region.bbox, extracted_region.bbox)
                        inferred_region.text = extracted_region.text
                        region_matched = True
                elif extracted_is_subregion_of_inferred and inferred_is_text:
                    if extracted_is_image:
                        # keep both extracted and inferred regions
                        region_matched = False
                    else:
                        # keep inferred region, remove extracted region
                        grow_region_to_match_region(inferred_region.bbox, extracted_region.bbox)
                        region_matched = True
                elif (
                    either_region_is_subregion_of_other
                    and inferred_region.type != ElementType.TABLE
                ):
                    # keep extracted region, remove inferred region
                    inferred_regions_to_remove.append(inferred_region)
        if not region_matched:
            extracted_elements_to_add.append(extracted_region)
    # Need to classify the extracted layout elements we're keeping.
    categorized_extracted_elements_to_add = [
        LayoutElement(
            text=el.text,
            type=ElementType.IMAGE
            if isinstance(el, ImageTextRegion)
            else ElementType.UNCATEGORIZED_TEXT,
            source=el.source,
            bbox=el.bbox,
        )
        for el in extracted_elements_to_add
    ]
    inferred_regions_to_add = [
        region for region in inferred_layout if region not in inferred_regions_to_remove
    ]

    final_layout = categorized_extracted_elements_to_add + inferred_regions_to_add

    return final_layout


def separate(region_a: Rectangle, region_b: Rectangle):
    """Reduce leftmost rectangle to don't overlap with the other"""

    def reduce(keep: Rectangle, reduce: Rectangle):
        # Asume intersection

        # Other is down
        if reduce.y2 > keep.y2 and reduce.x1 < keep.x2:
            # other is down-right
            if reduce.x2 > keep.x2 and reduce.y2 > keep.y2:
                reduce.x1 = keep.x2 * 1.01
                reduce.y1 = keep.y2 * 1.01
                return
            # other is down-left
            if reduce.x1 < keep.x1 and reduce.y1 < keep.y2:
                reduce.y1 = keep.y2
                return
            # other is centered
            reduce.y1 = keep.y2
        else:  # other is up
            # other is up-right
            if reduce.x2 > keep.x2 and reduce.y1 < keep.y1:
                reduce.y2 = keep.y1
                return
            # other is left
            if reduce.x1 < keep.x1 and reduce.y1 < keep.y1:
                reduce.y2 = keep.y1
                return
            # other is centered
            reduce.y2 = keep.y1

    if not region_a.intersects(region_b):
        return
    else:
        if region_a.area > region_b.area:
            reduce(keep=region_a, reduce=region_b)
        else:
            reduce(keep=region_b, reduce=region_a)


def table_cells_to_dataframe(cells: dict, nrows: int = 1, ncols: int = 1, header=None) -> DataFrame:
    """convert table-transformer's cells data into a pandas dataframe"""
    arr = np.empty((nrows, ncols), dtype=object)
    for cell in cells:
        rows = cell["row_nums"]
        cols = cell["column_nums"]
        if rows[0] >= nrows or cols[0] >= ncols:
            new_arr = np.empty((max(rows[0] + 1, nrows), max(cols[0] + 1, ncols)), dtype=object)
            new_arr[:nrows, :ncols] = arr
            arr = new_arr
            nrows, ncols = arr.shape
        arr[rows[0], cols[0]] = cell["cell text"]

    return DataFrame(arr, columns=header)


def partition_groups_from_regions(regions: Collection[TextRegion]) -> List[List[TextRegion]]:
    """Partitions regions into groups of regions based on proximity. Returns list of lists of
    regions, each list corresponding with a group"""
    if len(regions) == 0:
        return []
    padded_regions = [
        r.bbox.vpad(r.bbox.height * inference_config.ELEMENTS_V_PADDING_COEF).hpad(
            r.bbox.height * inference_config.ELEMENTS_H_PADDING_COEF,
        )
        for r in regions
    ]

    intersection_mtx = intersections(*padded_regions)

    _, group_nums = connected_components(intersection_mtx)
    groups: List[List[TextRegion]] = [[] for _ in range(max(group_nums) + 1)]
    for region, group_num in zip(regions, group_nums):
        groups[group_num].append(region)

    return groups
