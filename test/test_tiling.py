import numpy as np
import pytest
from deep_image_matching.tiling import Tiler


@pytest.fixture
def sample_image():
    return np.random.rand(100, 100)  # Sample image for testing


def test_tiler_init():
    tiler = Tiler()
    assert tiler.grid == [1, 1]
    assert tiler.overlap == 0
    assert tiler.origin == [0, 0]
    assert tiler.limits is None


def test_compute_limits_by_grid(sample_image):
    grid = [2, 2]
    overlap = 5
    tiler = Tiler(grid=grid, overlap=overlap)
    limits, origin = tiler.compute_limits_by_grid(sample_image)
    assert len(limits) == 4  # Expecting 4 tiles for 2x2 grid
    assert origin == [0, 0]  # Expecting origin to be [0, 0]
    w, h = sample_image.shape
    tile0_limits = [
        0,
        0,
        int(w / grid[0] + overlap - 1),
        int(h / grid[1] + overlap - 1),
    ]
    assert all([x == y for x, y in zip(limits[0], tile0_limits)])  # Check tile 0 limits


def test_extract_patch(sample_image):
    tiler = Tiler()
    patch = tiler.extract_patch(sample_image, [10, 10, 20, 20])
    assert patch.shape == (10, 10)


def test_read_all_tiles(sample_image):
    tiler = Tiler(grid=[2, 2], overlap=5)
    tiler.compute_limits_by_grid(sample_image)
    tiler.read_all_tiles()
    assert len(tiler._tiles) == 4  # Expecting 4 tiles for 2x2 grid


def test_read_tile(sample_image):
    grid = [2, 2]
    overlap = 5
    w, h = sample_image.shape
    tiler = Tiler(grid=grid, overlap=overlap)
    tiler.compute_limits_by_grid(sample_image)
    tiler.read_all_tiles()
    tile_idx = list(tiler._tiles.keys())[0]
    tile = tiler.read_tile(tile_idx)
    tile_shape = (
        int(w / grid[0] + overlap - 1),
        int(h / grid[1] + overlap - 1),
    )
    assert tile.shape == tile_shape


if __name__ == "__main__":
    pytest.main([__file__])
