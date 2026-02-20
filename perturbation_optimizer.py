import torch
import torch.nn.functional as F

def identity_loss(adv_emb, orig_emb, target_emb):

    sim_orig = F.cosine_similarity(adv_emb, orig_emb)
    sim_target = F.cosine_similarity(adv_emb, target_emb)

    # Strong directional push
    return (sim_orig - 3.5 * sim_target).mean()


def optimize_perturbation(model, image_tensor, orig_emb, target_emb,
                          steps=12, eps=12/255, alpha=3/255, device="cpu"):

    adv = image_tensor.clone().detach().to(device).float()

    for step in range(steps):

        adv.requires_grad_(True)

        adv_norm = (adv - 0.5) / 0.5
        adv_emb = model(adv_norm)

        loss = identity_loss(adv_emb, orig_emb.detach(), target_emb.detach())

        loss.backward()

        grad = adv.grad

        # ---- LOW FREQUENCY FILTER (important) ----
        # remove sharp pixel noise (patches)
        grad = torch.nn.functional.avg_pool2d(grad, 7, 1, 3)

        # normalize gradient strength
        grad = grad / (grad.abs().mean() + 1e-8)

        # smooth perturbation instead of pixel noise
        adv = adv - alpha * grad

        perturb = torch.clamp(adv - image_tensor, -eps, eps)
        adv = torch.clamp(image_tensor + perturb, 0, 1).detach()

        if step % 3 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    return adv
