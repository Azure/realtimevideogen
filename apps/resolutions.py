"""
Resolutions:
16:9 (YouTube standard):
    *  144p:  256 x  144
    *  240p:  426 x  240
    *         512 x  288 <- sketch
    *  360p:  640 x  360
    *         768 x  432
    *  480p:  854 x  480 <- video+audio (fantasy talk ~856x480)
    *        1024 x  576
    *  720p: 1280 x  720 <- final output
    *        1366 x  768
    * 1080p: 1920 x 1080
    * 1440p: 2560 x 1440
    * 2160p: 3840 x 2160
4:3:
    *         320 x  240
    *         480 x  360
    *   VGA:  640 x  480 <- sketch
    *         768 x  576 <- video+audio
    *  SVGA:  800 x  600
    *   XGA: 1024 x  768
    *        1152 x  864
    *  SXGA: 1280 x  960 <- final output
    *        1440 x 1080
    *  QXGA: 2048 x 1536
16:10:
    *  256 x  160
    *  384 x  240
    *  512 x  320
    *  640 x  400
    *  768 x  480
    *  896 x  560
    * 1024 x  640
    * 1280 x  800
    * 1440 x  900
5:4:
    *  320 x  256
    *  640 x  512
    *  800 x  640
    * 1280 x 1024
    * 1600 x 1280

Fantasy Talking:
    16:9:
        *  256 x 144 V
        *  320 x 180 X // 8 != 0
        *  426 x 240 X // 8 != 0
        *  512 x 288 V
        *  640 x 360 V (<=8 GPUs)
        *  768 x 432 V
        *  854 x 480 X // 8 != 0
        *  856 x 480 V
        * 1280 x 720 V
    4:3
        *  320 x 240 V (X 8 GPUs)
        *  480 x 360 V (X 8 GPUs)
        *  640 x 480 V (<=8 GPUs)
        *  768 x 576 V (<=8 GPUs)
        * 1024 x 768 V (<=8 GPUs)
        * 1280 x 960 V (<=8 GPUs)
    16:10:
        *  256 x 160 V (<=8 GPUs)
        *  384 x 240 V (<=8 GPUs)
        *  512 x 320 V (<=8 GPUs)
        *  640 x 400 V (<=8 GPUs)
        *  768 x 480 V (<=8 GPUs)
        *  896 x 560 V (<=8 GPUs)
        * 1024 x 640 V (<=8 GPUs)
    5:4:
        *  320 x  256 V (<=8 GPUs)
        *  480 x  384 V (<=8 GPUs)
        *  640 x  512 V (<=8 GPUs)
        *  720 x  576 V (X 8 GPUs)
        *  800 x  640 V (<=8 GPUs)
        *  960 x  768 V (<=8 GPUs)
        * 1280 x 1024 V (<=8 GPUs)
Hunyuan FramePack F1:
    16:9:
        *  256 x 144 V
        *  320 x 180 X // 8 != 0
        *  426 x 240 X // 8 != 0
        *  512 x 288 V
        *  640 x 360 X Dynamo failed
        *  768 x 432 V
        *  854 x 480 X // 8 != 0
        *  856 x 480 V
        *  896 x 540 X // 8 != 0
        * 1280 x 720 X OOM
    4:3:
        *  320 x 240 V
        *  480 x 360 X (7797) not supported for 8 GPUs.
        *  640 x 480 X (13820) not supported for 8 GPUs.
        *  768 x 576 V
        *  800 x 600 X (21620) not supported for 8 GPUs.
        * 1024 x 768 V Slow
        * 1280 x 960 ?
    16:10:
        *  256 x 160 V (4 GPUs)
        *  384 x 240 V (4 GPUs)
        *  512 x 320 V (4 GPUs)
        *  640 x 400 V (4 GPUs)
        *  768 x 480 V (4 GPUs)
        *  896 x 560 V Slow
        * 1024 x 640 V Slow
    5:4:
        *  240 x  192 V
        *  320 x  256 V
        *  480 x  384 V
        *  640 x  512 V
        *  720 x  576 V
        *  800 x  640 V
        *  960 x  768 V Slow
        * 1280 x 1024 V Slow
"""

from typing import Dict
from typing import Tuple

ASPECT_RATIO = "16:10"  # Close to 16:9 and allows 8 GPU parallelism

RESOLUTIONS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "4:3": {
        # "Low":    (640,  480), # Sketch  #  Not supported with 8 GPUs
        "low": (768, 576),  # Sketch
        "medium": (768, 576),  # Video+Audio
        "high": (1280, 960)  # Final output
    },
    "16:9": {
        "low": (854, 480),  # Sketch
        "medium": (1280, 720),  # Video+Audio
        "high": (1920, 1080)  # Final output
    },
    "5:4": {
        "low": (480, 384),  # Sketch
        "medium": (640, 512),  # Video+Audio
        "high": (1280, 1024),  # Final output
    },
    "16:10": {
        "low": (512, 320),  # Sketch
        "medium": (640, 400),  # Video+Audio
        "high": (1280, 800),  # Final output
    }
}
