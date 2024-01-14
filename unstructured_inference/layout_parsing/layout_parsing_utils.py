from typing import List, Optional

import pdf2image


def pdf_to_images(pdf_filename: str, dpi: int = 200, paths_only: bool = True, output_folder: Optional, Optional[str] = None) -> List[str]:

    if output_folder is not None:
        image_paths = pdf2image.convert_from_path(
            pdf_filename,
            dpi=dpi,
            output_folder=output_folder,
            paths_only=paths_only,
        )
    else:
        image_paths = pdf2image.convert_from_path(
            filename,
            dpi=dpi,
            paths_only=paths_only,
        )

    return image_paths
