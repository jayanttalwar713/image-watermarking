"""
dwt_descriptor.py
-----------------
Implements a 4-dimensional DWT energy descriptor for image regions,
then demos it on three region types to show how the descriptor differs.

Dependencies: pywt, numpy, Pillow, scikit-image
"""

import sys
import argparse
import numpy as np
import pywt
from PIL import Image
from skimage import color
from skimage import data as skdata


# ---------------------------------------------------------------------------
# Core descriptor function
# ---------------------------------------------------------------------------

def dwt_descriptor(region: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 4-element DWT energy descriptor for an image region.

    Parameters
    ----------
    region : np.ndarray
        H×W×3 uint8 (or float) RGB patch.

    Returns
    -------
    np.ndarray, shape (4,)
        Normalized energy vector [E_LL, E_LH, E_HL, E_HH] summing to 1.
        Subband order matches pywt convention: LL, LH (horizontal edges),
        HL (vertical edges), HH (diagonal / texture noise).
    """

    # Step 1 — Convert to grayscale.
    # DWT operates on a single-channel 2-D signal.  rgb2gray applies the
    # standard luminance weights (0.2126 R + 0.7152 G + 0.0722 B) so the
    # result reflects perceived brightness rather than a raw average.
    if region.ndim == 3:
        gray = color.rgb2gray(region.astype(np.float64))
    else:
        # Already single-channel; just normalise to [0, 1] if needed.
        gray = region.astype(np.float64)
        if gray.max() > 1.0:
            gray /= 255.0

    # Step 2 — Apply a single-level 2-D Discrete Wavelet Transform (Haar).
    # pywt.dwt2 returns:
    #   cA  — approximation coefficients (LL subband): low-pass in both dims,
    #          captures the coarse, average structure of the region.
    #   (cH, cV, cD) — detail coefficients:
    #     cH (LH): low-pass horizontal × high-pass vertical → horizontal edges
    #     cV (HL): high-pass horizontal × low-pass vertical → vertical edges
    #     cD (HH): high-pass in both dims → diagonal edges / fine texture
    # Haar is the simplest wavelet (box filter); it's fast and gives clean
    # energy separation between smooth and sharp content.
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')

    # Step 3 — Compute energy (sum of squared coefficients) for each subband.
    # Energy is a natural summary statistic for wavelet coefficients:
    # large coefficients (= strong response) contribute quadratically,
    # so a single high-contrast edge dominates over many weak ones.
    E_LL = float(np.sum(cA ** 2))   # coarse / low-frequency energy
    E_LH = float(np.sum(cH ** 2))   # horizontal-edge energy
    E_HL = float(np.sum(cV ** 2))   # vertical-edge energy
    E_HH = float(np.sum(cD ** 2))   # diagonal / texture energy

    energies = np.array([E_LL, E_LH, E_HL, E_HH], dtype=np.float64)

    # Step 4 — Normalise by total energy so the vector sums to 1.
    # This makes the descriptor invariant to overall brightness: a dark patch
    # and a bright patch of the same texture pattern will get the same vector.
    total = energies.sum()
    if total > 0:
        energies /= total

    return energies  # shape (4,), values in [0, 1], sum == 1


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _crop(image: np.ndarray, row: int, col: int, size: int = 64) -> np.ndarray:
    """Return a size×size RGB patch starting at (row, col)."""
    return image[row: row + size, col: col + size]


def main() -> None:
    # ---- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="DWT region descriptor demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3.14 dwt_descriptor.py photo.jpg\n"
            "  python3.14 dwt_descriptor.py photo.jpg --resize 512\n"
            "  python3.14 dwt_descriptor.py img1.jpg img2.jpg --resize 512\n"
        ),
    )
    parser.add_argument(
        "images", nargs="*",
        help="Path(s) to image file(s). Omit to use the built-in test image.",
    )
    parser.add_argument(
        "--resize", type=int, default=None, metavar="N",
        help=(
            "Resize every image to N×N before patching. "
            "Use this when comparing images of different resolutions so that "
            "a 64×64 patch always covers the same proportional area of each image. "
            "Example: --resize 512"
        ),
    )
    args = parser.parse_args()

    # ---- Load image(s) -------------------------------------------------------
    images: list[tuple[str, np.ndarray]] = []

    if args.images:
        for path in args.images:
            # Pillow handles JPEG, PNG, TIFF, BMP, WEBP, etc.
            # Convert to RGB for a consistent 3-channel uint8 array regardless
            # of the source format (RGBA, palette, greyscale, etc.).
            pil_img = Image.open(path).convert("RGB")

            if args.resize:
                # Resize to a square canvas so patches cover the same
                # proportional region across images of different resolutions.
                # LANCZOS gives the best quality when downsampling.
                pil_img = pil_img.resize((args.resize, args.resize), Image.LANCZOS)

            images.append((path, np.array(pil_img)))
    else:
        # No file supplied — use the bundled scikit-image test image.
        img = skdata.astronaut()
        if args.resize:
            pil_img = Image.fromarray(img).resize((args.resize, args.resize), Image.LANCZOS)
            img = np.array(pil_img)
        images.append(("built-in astronaut", img))
        print("No image path provided — using built-in astronaut test image.")
        print("Usage: python3.14 dwt_descriptor.py <image> [image2 ...] [--resize N]\n")

    # ---- Run descriptor on each image ----------------------------------------
    for img_path, image in images:
        print(f"{'─' * 60}")
        print(f"Image : {img_path}")
        print(f"Shape : {image.shape}  dtype: {image.dtype}")
        if args.resize:
            print(f"Resized to {args.resize}×{args.resize} before patching")

        h, w = image.shape[:2]
        # Patch size: 64×64, or smaller if the image is tiny
        patch_size = min(64, h // 4, w // 4)
        print(f"Patch : {patch_size}×{patch_size}\n")

        # ---- Choose three representative patches ----------------------------
        # Patches are placed at proportional positions so they work for any
        # image size (or any resize target).
        regions = {
            # Top-left quadrant — often sky, background, or smooth areas.
            "region 1 (top-left)":    _crop(image, row=h // 8,              col=w // 8,              size=patch_size),
            # Centre — typically the main subject, often textured.
            "region 2 (centre)":      _crop(image, row=h // 2 - patch_size, col=w // 2 - patch_size, size=patch_size),
            # Upper-right — frequently edges, borders, or contrasting areas.
            "region 3 (upper-right)": _crop(image, row=h // 8,              col=w * 3 // 4,          size=patch_size),
        }

        # ---- Run descriptor on each patch -----------------------------------
        results = {}
        for name, patch in regions.items():
            results[name] = dwt_descriptor(patch)

        # ---- Print side-by-side comparison table ----------------------------
        labels = ["LL (approx)", "LH (horiz edges)", "HL (vert edges)", "HH (diag/texture)"]
        col_w = 22

        header = f"{'Subband':<20}" + "".join(f"{name:<{col_w}}" for name in results)
        print(header)
        print("-" * len(header))

        for i, label in enumerate(labels):
            row = f"{label:<20}"
            for descriptor in results.values():
                row += f"{descriptor[i]:<{col_w}.4f}"
            print(row)

        print()

    # ---- Interpretation hints (printed once) --------------------------------
    print("Interpretation:")
    print("  LL  — smooth regions score HIGH here (most energy is coarse structure)")
    print("  LH  — horizontal edges score HIGH here")
    print("  HL  — vertical edges score HIGH here")
    print("  HH  — textured/noisy regions score HIGH here")
    if not args.resize and len(images) > 1:
        print("\nTip: images have different resolutions — add --resize 512 for a fair comparison.")


if __name__ == "__main__":
    main()
