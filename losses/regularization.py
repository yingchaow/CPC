def quantization_loss(image_hash, text_hash):
    target_magnitude = image_hash.shape[1] ** -0.5
    return (
        (image_hash.abs() - target_magnitude).pow(2).mean()
        + (text_hash.abs() - target_magnitude).pow(2).mean()
    ) / 2.0
