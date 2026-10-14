import torch
from transformers_turbo import TurboTrainer
from megatron.core import mpu


class DftTurboTrainer(TurboTrainer):
    """TurboTrainer with DFT (Dynamic Fine-Tuning) loss.

    Overrides _post_compute_loss to apply exp(-loss) weighting before reduction,
    so tokens the model has already learned well contribute less gradient.
    """

    def __init__(self, *args, dft_weight_min=0.0, dft_start_epoch=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dft_weight_min = dft_weight_min
        self.dft_start_epoch = dft_start_epoch

    def _post_compute_loss(self, loss_mask, losses):
        loss_mask = loss_mask.view(-1).float()
        per_token_losses = losses.view(-1)

        use_dft = self.state.epoch >= self.dft_start_epoch

        # DFT weighting with optional lower-bound clipping
        with torch.no_grad():
            if use_dft:
                dft_weights = torch.exp(-per_token_losses)
                if self.dft_weight_min > 0:
                    clipped_count = ((dft_weights < self.dft_weight_min) * loss_mask).sum()
                    dft_weights = torch.clamp(dft_weights, min=self.dft_weight_min)
                else:
                    clipped_count = loss_mask.new_zeros(1).squeeze()
            else:
                dft_weights = torch.ones_like(per_token_losses)
                clipped_count = loss_mask.new_zeros(1).squeeze()

        losses = torch.sum(per_token_losses * dft_weights * loss_mask)
        loss_mask = loss_mask.sum()

        # Context parallel reduction
        cp_size = self.model.config.context_parallel_size
        if cp_size > 1:
            # gather losses, token count, and clipped count together
            loss_info = torch.cat([losses.view(1), loss_mask.view(1), clipped_count.view(1)])
            torch.distributed.all_reduce(
                loss_info, op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )
            losses, loss_mask, clipped_count = loss_info[0], loss_info[1], loss_info[2]

        loss = losses.clone()
        local_num_tokens = loss_mask.clone().detach()
        if local_num_tokens == 0:
            local_num_tokens += 1

        clipped_ratio = (clipped_count / local_num_tokens).clone().detach()

        if self.args.calculate_per_token_loss:
            metrics = {"loss": (loss.clone().detach(), local_num_tokens)}
        else:
            metrics = {"loss": (loss / local_num_tokens).clone().detach()}
        metrics["dft_clipped_ratio"] = clipped_ratio
        return loss, local_num_tokens.int(), metrics
