import matplotlib.pyplot as plt
import numpy as np


class BaseRender:
    """
    BaseRender class
    """

    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize
        self.fig, self.axes = None, None

    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis("off")
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()

    def close_canvas(self):
        plt.close()

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        print(f"saving to {filename}")
        plt.savefig(filename)


class BEVRender(BaseRender):
    """
    Render class for BEV
    """

    def __init__(
        self,
        figsize=(20, 20),
        margin: float = 50,
        view: np.ndarray = np.eye(4),
        show_gt_boxes=False,
    ):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes

    def set_plot_cfg(self):
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect("equal")
        self.axes.grid(False)


class CameraRender(BaseRender):
    def __init__(self, figsize=(10, 10), show_gt_boxes=False):
        super().__init__(figsize=figsize)

        self.cams = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ]
        self.show_gt_boxes = show_gt_boxes

    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis("off")
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()

    def close_canvas(self):
        plt.close()

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        print(f"saving to {filename}")
        plt.savefig(filename)

    def render_image_data(self, sample_token, nusc):
        """Load and annotate image based on the provided path."""
        sample = nusc.get("sample", sample_token)
        for i, cam in enumerate(self.cams):
            sample_data_token = sample["data"][cam]
            data_path, _, _, _, _ = self.get_image_info(sample_data_token, nusc)
            image = self.load_image(data_path, cam)
            self.update_image(image, i, cam)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index // 3, index % 3]

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis("off")
        ax.axis("off")
        ax.grid(False)
