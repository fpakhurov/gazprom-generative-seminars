"""Display fixed-noise GAN samples without modifying training state."""

import math
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output, display


def gan_training_preview(generator, discriminator, latent, real_reference, step, total,
                         loss_d, loss_g, elapsed, handle=None, labels=None,
                         reference_labels=None, title="GAN"):
    generator_mode, discriminator_mode = generator.training, discriminator.training
    try:
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            images = generator(latent) if labels is None else generator(latent, labels)
            fake_logits = discriminator(images) if labels is None else discriminator(images, labels)
            real_logits = (discriminator(real_reference) if reference_labels is None
                           else discriminator(real_reference, reference_labels))
            scores = fake_logits.sigmoid().flatten().cpu()
            real_score = real_logits.sigmoid().mean().item()
            images = images.detach().cpu()
    finally:
        generator.train(generator_mode)
        discriminator.train(discriminator_mode)

    columns = 6
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(9, 1.7 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
            prefix = "" if labels is None else f"y={int(labels[index])}; "
            axis.set_title(f"{prefix}D={scores[index]:.2f}", fontsize=9)
    remaining = max(total - step, 0) * elapsed / max(step, 1)
    figure.suptitle(
        f"{title}: step {step}/{total} ({100 * step / total:.0f}%) | "
        f"{elapsed:.1f} s elapsed | about {remaining:.1f} s remaining\n"
        f"loss D={loss_d:.3f}; G={loss_g:.3f} | "
        f"mean D(real)={real_score:.2f}; D(fake)={scores.mean().item():.2f}", fontsize=10)
    figure.text(0.5, 0.01, "D score = sigmoid(current discriminator logit); not an image-quality metric.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.87))
    try:
        clear_output(wait=True)
        display(figure)
        handle = None
    finally:
        plt.close(figure)
    summary = {"step": step, "total": total, "real_score": real_score,
               "fake_scores": scores.tolist(), "elapsed_seconds": elapsed}
    return handle, images, summary
