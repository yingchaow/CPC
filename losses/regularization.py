import torch
import torch.nn.functional as F


def quantization_loss(image_hash, text_hash):
    target_magnitude = image_hash.shape[1] ** -0.5
    return (
        (image_hash.abs() - target_magnitude).pow(2).mean()
        + (text_hash.abs() - target_magnitude).pow(2).mean()
    ) / 2.0


def bit_balance_loss(image_hash, text_hash):
    return (
        image_hash.mean(dim=0).pow(2).mean()
        + text_hash.mean(dim=0).pow(2).mean()
    ) / 2.0


def ema_consistency_loss(
    student_image,
    student_text,
    teacher_image,
    teacher_text,
    unselected,
):
    unselected = unselected.bool()
    if not unselected.any():
        return student_image.new_zeros(())
    if teacher_image is None or teacher_text is None:
        raise ValueError("EMA consistency requires Teacher outputs")
    return (
        F.mse_loss(student_image[unselected], teacher_image[unselected])
        + F.mse_loss(student_text[unselected], teacher_text[unselected])
    ) / 2.0
