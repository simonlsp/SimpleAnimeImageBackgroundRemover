import argparse
import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops_table, label
from skimage.morphology import skeletonize
from skimage.morphology import binary_erosion, disk
from ImageAlphaToolkit import calculate_neighborhood_max_alpha, reconstruct_fg

def update_alpha_by_cellular_automata(alpha_matrix, initial_transparent_mask = 0.125, radius = 2):
    """
    Iteratively update the Alpha matrix using 8-neighbor voting rules:
    - More than 80% neighbors are "semi-transparent" or "opaque" -> become opaque (1.0)
    - More than 80% neighbors are "transparent" -> become transparent (0.0)
    - Otherwise (semi-transparent points) keep the original Alpha value unchanged.

    Args:
    alpha_matrix: 2D NumPy array (H, W), values in [0.0, 1.0]
    max_iters: Max iterations, default 100

    Returns:
    updated_alpha: New Alpha matrix (H, W) after iterative filtering
    """
    assert len(alpha_matrix.shape) == 2, "Input must be a 2D Alpha matrix (H, W)"

    # Copy the original matrix for final output, avoid modifying original directly
    updated_alpha = alpha_matrix.astype(np.float64).copy()

    # Initialize state mask (0: transparent, 1: semi-transparent, 2: opaque)
    state = np.ones_like(alpha_matrix, dtype=np.uint8)
    state[alpha_matrix < initial_transparent_mask] = 0

    # Convolution kernel
    kernel = disk(radius).astype(np.uint8)
    kernel[radius, radius] = 0

    mostly_threshould = min(np.round(np.sum(kernel) * 0.8), np.sum(kernel) - 1)
    exist_threshould = max(np.round(np.sum(kernel) * 0.1), 1)
    # Cellular automaton iteration
    for i in range(256): # Max iterations = 256
        old_state = state.copy()

        opaque_mask = (state == 2)
        semi_mask = (state == 1)
        trans_mask = (state == 0)

        # Rule 1: Points connected to 80%+ semi-transparent / opaque points become opaque, except that they connected to at least 10% transparent point
        # (semi-transparency must occur at transparent/opaque boundaries, large blocks of semi-transparent cannot exist)
        rule_opaque = np.logical_and(
            convolve(np.logical_or(opaque_mask, semi_mask).astype(np.uint8), kernel, mode='constant', cval=0) >= mostly_threshould, 
            convolve(trans_mask.astype(np.uint8), kernel, mode='constant', cval=0) <= exist_threshould)
        # Rule 2: Points connected to 80%+ transparent points become transparent, except that they connected to at least 10% opaque point
        # Same reason as above
        rule_trans = np.logical_and(
            convolve(trans_mask.astype(np.uint8), kernel, mode='constant', cval=0) >= mostly_threshould,
            convolve(opaque_mask.astype(np.uint8), kernel, mode='constant', cval=0) <= exist_threshould)

        # Update internal state
        state[rule_opaque] = 2  # Become opaque
        state[rule_trans] = 0   # Become transparent

        # Check convergence
        if np.array_equal(state, old_state):
            break
    else:
        print(f"Warning: Reached max iterations 256.")

    updated_alpha[state == 0] = 0.0
    updated_alpha[state == 1] = alpha_matrix[state == 1] ** 0.5
    updated_alpha[state == 2] = 1.0

    return updated_alpha

def CheckSmoothness(sobel_image, mask, border_width):
    """
    Check the brightness smoothness at the boundary of the given mask.
    Args:
    sobel_image: Sobel gradient image, NumPy array, shape (H, W)
    mask: Binary mask, shape (H, W), True for inside region, False for outside
    Returns:
    0: Too small
    1: Smooth
    2: Rough
    """
    # Get border of the mask
    eroded_mask = binary_erosion(mask, disk(border_width))
    if np.sum(eroded_mask) <= 4:
        return 0  # Region too small
    border_mask = np.logical_and(mask, np.logical_not(eroded_mask))
    if np.sum(border_mask) <= 4:
        return 0  # Border too small
    border_sobel = sobel_image[border_mask]
    if np.mean(border_sobel) <= (16 / 256):
        return 1
    else:
        return 2

def update_alpha_by_regional_content(alpha_matrix, ca_alpha_matrix, border_width, noise_thres, transparent_thres):
    bg_candicates = label(ca_alpha_matrix < 32 / 256)

    micro_labels = set()
    smooth_labels = set()
    rough_labels = set()
    sobel_image = cv2.Sobel(alpha_matrix, cv2.CV_64F, 1, 1, ksize=3)
    sobel_props = regionprops_table(bg_candicates, intensity_image=sobel_image, properties=['label', 'area', 'image', 'image_intensity'])
    for area_label, area, image_boolean, image_intensity in zip(sobel_props['label'], sobel_props['area'], sobel_props['image'], sobel_props['image_intensity']):
        # Check region boundary smoothness
        smoothness = CheckSmoothness(image_intensity, image_boolean, border_width)
        if smoothness == 0:
            micro_labels.add(area_label)
        elif smoothness == 1:
            smooth_labels.add(area_label)
        else:
            rough_labels.add(area_label)
    

    foreground_labels = set()
    alpha_props = regionprops_table(bg_candicates, intensity_image=alpha_matrix, properties=['label', 'area', 'intensity_min', 'image', 'image_intensity'])
    for area_label, area, intensity_min, image_boolean, image_intensity in zip(alpha_props['label'], alpha_props['area'], alpha_props['intensity_min'], alpha_props['image'], alpha_props['image_intensity']):
        # Non-transparent region
        if intensity_min > noise_thres:
            foreground_labels.add(area_label)
        # Tiny region: may be affected by surrounding semi-transparency, check skeleton transparency
        if area_label in micro_labels:
            area_skeleton = skeletonize(image_boolean)
            skeleton_intensity = np.mean(image_intensity[area_skeleton])
            if skeleton_intensity > transparent_thres:
                foreground_labels.add(area_label)
        # Smooth region: check average intensity
        elif area_label in smooth_labels:
            if np.mean(image_intensity[image_boolean]) > transparent_thres:
                foreground_labels.add(area_label)
        # Rough region: check average intensity after erosion
        elif area_label in rough_labels:
            if np.mean(image_intensity[binary_erosion(image_boolean, disk(border_width))]) > transparent_thres:
                foreground_labels.add(area_label)

    bg_mask = bg_candicates != 0
    for fg_label in foreground_labels:
        bg_mask[bg_candicates == fg_label] = False
    fg_mask = binary_erosion(np.logical_not(bg_mask), disk(border_width))
    bg_eroded_mask = binary_erosion(bg_mask, disk(border_width))

    return_matrix = ca_alpha_matrix.copy()
    return_matrix[fg_mask] = 1.0
    return_matrix[bg_eroded_mask] = np.min(np.stack([return_matrix[bg_eroded_mask], alpha_matrix[bg_eroded_mask]], axis=0), axis=0)
    
    return return_matrix

def parse_bg_color(value: str) -> list[int]:
    """Parse a background color string like '255,255,255' or '[255,255,255]' into a list of ints."""
    value = value.strip().strip("[]()")
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"bg_color must be three comma-separated integers, e.g. '255,255,255', got: '{value}'"
        )
    return [int(p.strip()) for p in parts][::-1] # RGB to BGR


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Remove background from anime-style images using alpha matting."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input image.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output image (RGBA PNG).")
    parser.add_argument("--bg_color", required=True, type=parse_bg_color,help="Background color as comma-separated RGB, e.g. '255,255,255'.")
    parser.add_argument("--transparent_thres", type=float, default=(16 / 256),help="Minimum alpha value for semi-transparent regions (default: 16/256 ≈ %.4f)." % (16 / 256))
    parser.add_argument("--border_width", type=int, default=2, help="Width of the anime-style border line in pixels (default: 2).")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    bg_color = args.bg_color
    transparent_thres = args.transparent_thres
    border_width = args.border_width

    # Could be advanced hyper-parameters, but for default, we can set them as follows:
    background_noise_thres = transparent_thres / 2
    initial_transparent_thres = transparent_thres * 2

    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    raw_alpha = calculate_neighborhood_max_alpha(image, bg_color)
    ca_alpha = update_alpha_by_cellular_automata(raw_alpha, initial_transparent_mask = initial_transparent_thres, radius = border_width)
    final_alpha = update_alpha_by_regional_content(raw_alpha, ca_alpha, border_width = border_width, noise_thres = background_noise_thres, transparent_thres=transparent_thres)

    reconstructed_fg = reconstruct_fg(image, bg_color, final_alpha)
    alpha_uint8 = np.clip(final_alpha * 255, 0, 255).astype(np.uint8)
    output_image = np.dstack((reconstructed_fg, alpha_uint8))
    # Save the result
    cv2.imwrite(output_path, output_image)
