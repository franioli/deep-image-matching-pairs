import cv2

from deep_image_matcher.logger import setup_logger
from deep_image_matcher import (
    LightGlueMatcher,
    Quality,
    TileSelection,
    GeometricVerification,
)

logger = setup_logger()

img0 = cv2.imread("./data/img/IMG_2650.jpg", cv2.COLOR_RGB2BGR)
img1 = cv2.imread("./data/img/IMG_1125.jpg", cv2.COLOR_RGB2BGR)

# Test LightGlue
matcher = LightGlueMatcher()
matcher.match(
    img0,
    img1,
    quality=Quality.LOW,
    tile_selection=TileSelection.NONE,
    save_dir="res/LIGHTGLUE",
    geometric_verification=GeometricVerification.PYDEGENSAC,
    threshold=2,
    confidence=0.9999,
)
mm = matcher.mkpts0
