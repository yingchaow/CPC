from copy import deepcopy

import torch


def create_teacher(student):
    teacher = deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(momentum).add_(
            student_parameter.detach(), alpha=1.0 - momentum
        )
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(), student.buffers()
    ):
        teacher_buffer.copy_(student_buffer)

