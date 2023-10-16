import cv2

from deep_image_matching.logger import setup_logger
from deep_image_matching import (
    LightGlueMatcher,
    Quality,
    TileSelection,
    GeometricVerification,
)

logger = setup_logger()

# img0 = cv2.imread("./data/img/IMG_2650.jpg", cv2.COLOR_RGB2BGR)
# img1 = cv2.imread("./data/img/IMG_1125.jpg", cv2.COLOR_RGB2BGR)

img0 = cv2.imread("./data/forni/c1/IMG_4529.JPG", cv2.COLOR_RGB2BGR)
img1 = cv2.imread("./data/forni/c2/IMG_4500.JPG", cv2.COLOR_RGB2BGR)


# Test LightGlue
matcher = LightGlueMatcher(
    geometric_verification=GeometricVerification.PYDEGENSAC,
    quality=Quality.HIGH,
    tile_selection=TileSelection.PRESELECTION,
    save_dir="res/LIGHTGLUE",
    max_keypoints=10240,
    grid=[2, 3],
    overlap=200,
    threshold=2,
    confidence=0.9999,
    do_viz=True,
    do_viz_tiles=True,
    fast_viz=True,
    hide_matching_track=True,
)
matcher.match(
    img0,
    img1,
)
mm = matcher.mkpts0
