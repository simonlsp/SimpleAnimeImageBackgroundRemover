import numpy as np

def find_minimal_alpha(RGBimage, bg_color):
    alpha_stack = list()

    for channel in range(3):
        input_channel = RGBimage[:, :, channel].astype(np.float64)
        background_channel = float(bg_color[channel])

        current_channel_min_alpha = np.zeros(RGBimage.shape[:2], dtype=np.float64)
        BrighterMask = (input_channel - background_channel) > 1e-8
        DarkerMask = (background_channel - input_channel) > 1e-8

        # Constraint 1: Merged > BG, calculate min alpha with FG = 255, As Merged > BG, 255 - BG > 0.
        if np.any(BrighterMask):
            current_channel_min_alpha[BrighterMask] = (input_channel[BrighterMask] - background_channel) / (255.0 - background_channel)
        # Constraint 2: Merged < BG, calculate min alpha with FG = 0, As Merged < BG, BG > 0.
        if np.any(DarkerMask):
            current_channel_min_alpha[DarkerMask] = (background_channel - input_channel[DarkerMask]) / background_channel
        # Constraint 3: Merged == BG, min alpha = 0, no need to change anything.

        alpha_stack.append(current_channel_min_alpha)

    # Stack the min alpha for each channel and take the maximum to satisfy all channels
    return np.maximum.reduce(alpha_stack)

def reconstruct_fg(RGBimage, bg_color, alpha):
    """
    Use input merged image, background color, and alpha matrix to reconstruct the foreground RGB color.
    """
    image_float = RGBimage.astype(np.float64)
    bg_float = np.array(bg_color, dtype=np.float64) # (3,)

    fg_image = np.zeros_like(image_float)
    alpha_mask = alpha > 1e-8

    # get the foreground color using the formula: FG = (img - bg * (1 - alpha)) / alpha
    using_alpha = alpha[:,:,np.newaxis][alpha_mask]
    fg_image[alpha_mask] = (image_float[alpha_mask] - bg_float * (1.0 - using_alpha)) / using_alpha

    return np.clip(fg_image, 0, 255).astype(np.uint8)

def estimate_best_alpha(foreground, background, merged):
    """
    Based on least squares (Euclidean distance), derive the best alpha jointly from foreground, background, and merged images.
    Requires input matrices to be of shape (H, W, 3) and type uint8 or float.
    Returns an alpha matrix of shape (H, W) with values in [0.0, 1.0].
    """
    F = foreground.astype(np.float64)
    B = np.array(background).astype(np.float64) # support RGB list or (H, W, 3) image
    M = merged.astype(np.float64)
    
    delta_FB = F - B  # 前景 - 背景
    delta_MB = M - B  # 混合 - 背景

    # Get the numerator and denominator for alpha calculation
    numerator = np.sum(delta_FB * delta_MB, axis=-1)
    denominator = np.sum(delta_FB ** 2, axis=-1)

    denominatorMask = denominator > 1e-8
    alpha = np.zeros_like(numerator, dtype=np.float64)
    alpha[denominatorMask] = numerator[denominatorMask] / denominator[denominatorMask]
    
    return np.clip(alpha, 0.0, 1.0).astype(np.float64)


def calculate_neighborhood_max_alpha(rgb_array, bg_color):
    """
    Core function:
    1. Compute foreground color within 4-neighborhood
    2. Compute the closest transparency
    3. Return the maximum of the 5 transparency values

    Input rgb_img must have shape (H, W, 3)
    """
    assert len(rgb_array.shape) == 3 and rgb_array.shape[-1] == 3, "Input must be an (H, W, 3) image matrix"

    # Get extreme foreground color for the entire image
    min_alpha = find_minimal_alpha(rgb_array, bg_color)
    assumed_fg = reconstruct_fg(rgb_array, bg_color, min_alpha)

    # Collect 4-neighbor + self, total 5 foreground color matrices
    fg_candidates = [assumed_fg,
        np.roll(assumed_fg, shift=-1, axis=0),
        np.roll(assumed_fg, shift=1, axis=0),
        np.roll(assumed_fg, shift=-1, axis=1),
        np.roll(assumed_fg, shift=1, axis=1)]

    # Compute optimal transparency for each of the 5 foreground colors under current color C
    alpha_candidates = []
    for FG_cand in fg_candidates:
        alpha_cand = estimate_best_alpha(FG_cand, bg_color, rgb_array)
        alpha_candidates.append(alpha_cand)

    return np.maximum.reduce(alpha_candidates)