import matplotlib
import torch

matplotlib.use("Agg")

from jepa_lab.masking import MultiBlockMasker
from jepa_lab.video_jepa import generate_tube_masks
from jepa_lab.visualization import image_mask_map, plot_image_masks, plot_tube_mask


def test_mask_visualizations_build_figures() -> None:
    masks = MultiBlockMasker((14, 14))(1, generator=torch.Generator().manual_seed(1))
    assert image_mask_map(masks).shape == (14, 14)
    image_figure = plot_image_masks(masks)
    tube_figure = plot_tube_mask(generate_tube_masks((4, 7, 7), seed=1).target)
    assert image_figure.axes
    assert tube_figure.axes
