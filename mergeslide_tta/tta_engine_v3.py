"""
mergeslide_tta/tta_engine_v3.py
================================
MergeSlide-TTA Engine v3 — Task-Aware Routing Adaptation.

Changes vs. v2 (tta_engine.py):
    1. Phase 1d — Routing dùng z_teacher thay vì z_anchor.
       Lý do: teacher là EMA của student, phản ánh current embedding space
       tốt hơn anchor frozen. Consistent với PETAL: teacher là nguồn đáng
       tin cậy nhất tại mọi thời điểm.

    2. Phase 3 — Bổ sung L_task: task-level margin loss.
       L_task = ReLU(s_top2 - s_top1 + m), tính từ z_student.
       Push student embedding tăng gap giữa task dẫn đầu và task thứ hai
       trong không gian task prompt, bất kể task nào đang dẫn đầu.
       Không giả định routing đúng — chỉ push separation.

    3. Phase 5.5 — Task prompt EMA update (SwapPrompt-inspired).
       Sau teacher EMA, nếu routing confidence gap đủ lớn (delta_margin),
       kéo task_prompts[t_adapt] về phía z_teacher của slide hiện tại.
       Confidence gate (TPT-inspired): chỉ update khi top1 - top2 > delta_margin
       để tránh contamination khi routing không tin cậy.

    4. reset_to_source() — Reset cả task_prompts về source để tránh drift
       tích lũy giữa các tasks.

Modules:
    A  : WSI Bag-level Augmentation       (Frozen-to-Paraffin, MICCAI 2021)
    B  : DaPC Pseudo-label Correction     (WeiContrast-DaPC, IEEE TETC 2025)
    C  : PETAL Loss + L_class + L_task    (PETAL CVPR 2023; L_task: new)
    D  : FIM-based Parameter Restoration  (PETAL, CVPR 2023)
    E  : Adaptive LR via OOD score        (NOTE-inspired, NeurIPS 2022)
    P5b: Task Prompt EMA Update           (SwapPrompt-inspired + TPT gate)
    P7 : TCP Confidence Gate              (MergeSlide, arXiv 2025)

Usage::
    adapter = MergeSlide_TTA_Adapter_v3(...)
    pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(features, coords)
    adapter.reset_to_source()
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTAConfig_v3:
    # Final prediction backbone for naive CLASS-IL and TASK-IL.
    inference_model: str = "teacher"  # teacher | student

    # Module A — WSI Bag Augmentation (Frozen-to-Paraffin)
    K: int          = 8      # số augmented views cho teacher
    r_patch: float  = 0.75   # fraction của K_PATCHES cho mỗi aug view

    # Module B — DaPC pseudo-label correction
    tau_ap_ind: float = 0.92
    tau_ap_ood: float = 0.70
    beta_ind: float   = 1.2
    beta_ood: float   = 1.5

    # Module C — PETAL loss
    spw: float         = 1e-6
    lam_ce: float      = 1.0
    gamma_class: float = 1.0
    tau_c: float       = 0.07

    # Module C — L_task (NEW v3)
    gamma_task: float  = 0.5    # weight cho L_task trong L_total
    margin_task: float = 0.1    # margin m: L_task > 0 khi gap < margin_task

    # Module D — FIM Restoration
    delta: float = 0.03

    # Module E — Adaptive LR
    eta_base: float = 1e-4
    tau_ood: float  = 0.5

    # EMA teacher
    ema_alpha: float = 0.999

    # Phase 5b — Task Prompt EMA Update (SwapPrompt-inspired, NEW v3)
    alpha_task_prompt: float = 0.999   # EMA momentum cho task prompt update
    delta_margin: float      = 0.10    # confidence gate: chỉ update khi
                                        # top1_score - top2_score > delta_margin

    # Phase 7 — TCP Gate
    tau_task: float = 0.70

    # Misc
    eps: float         = 1e-6
    k_patches_std: int = 400


# ──────────────────────────────────────────────────────────────────────────────
# Adapter v3
# ──────────────────────────────────────────────────────────────────────────────

class MergeSlide_TTA_Adapter_v3:
    """
    Online, exemplar-free TTA adapter — v3 với Task-Aware Routing Adaptation.

    Parameters
    ----------
    Giống MergeSlide_TTA_Adapter (v2), không thay đổi interface.
    task_prompts : Tensor  shape [T, 768]
        Sẽ được clone thành self.task_prompts (mutable) và
        self.task_prompts_source (frozen, dùng cho reset).
    """

    def __init__(
        self,
        base_vision_encoder: nn.Module,
        backbone_sd: dict,
        mean_sd: dict,
        var_sd: dict,
        per_task_mlp_weights: dict,
        global_mlp_weight: Tensor,
        global_mlp_bias: Tensor,
        task_prompts: Tensor,
        task_class_ranges: dict,
        cfg: TTAConfig_v3,
        device: torch.device,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        # ── Frozen MLP weights (prompt prototypes) ──────────────────────────
        self.per_task_w: dict[int, Tensor] = {
            t: w.to(device) for t, (w, _) in per_task_mlp_weights.items()
        }
        self.per_task_b: dict[int, Tensor] = {
            t: b.to(device) for t, (_, b) in per_task_mlp_weights.items()
        }
        self.global_w = global_mlp_weight.to(device)
        self.global_b = global_mlp_bias.to(device)

        # ── Task prompts — MUTABLE (v3: EMA update tại TTA time) ────────────
        # task_prompts_source: frozen copy dùng cho reset
        self.task_prompts_source = task_prompts.clone().to(device)
        # task_prompts: working copy, được update tại Phase 5b
        self.task_prompts = task_prompts.clone().to(device)
        self.task_class_ranges = task_class_ranges

        # ── SWAG posterior ───────────────────────────────────────────────────
        self.mean_sd = mean_sd
        self.var_sd  = var_sd

        # ── TITAN ps argument ───────────────────────────────────────────────
        from mergeslide_tta.constants import TITAN_PS_ARG
        self.ps = torch.tensor(TITAN_PS_ARG).int().to(device)

        # ── Source anchor state dict ─────────────────────────────────────────
        self.anchor_sd = {k: v.clone().cpu() for k, v in backbone_sd.items()}

        # ── Build 3 backbone instances ───────────────────────────────────────
        self.anchor  = self._make_backbone(base_vision_encoder, backbone_sd, train=False)
        self.student = self._make_backbone(base_vision_encoder, backbone_sd, train=True)
        self.teacher = self._make_backbone(base_vision_encoder, backbone_sd, train=False)

        # ── Trainable LN param names ─────────────────────────────────────────
        self.ln_param_names = [
            n for n, p in self.student.named_parameters() if p.requires_grad
        ]

        # ── Optimizer (student LN params only) ──────────────────────────────
        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=cfg.eta_base, weight_decay=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def adapt_and_predict(
        self,
        features: Tensor,
        coords: Tensor,
        task_id: Optional[int] = None,
        use_tcp_gate: bool = True,
        inference_model: Optional[str] = None,
    ) -> tuple[int, int, dict]:
        """
        7 phases + Phase 5b (task prompt EMA update).

        Returns
        -------
        pred_local : int
        pred_task  : int
        prob_np    : np.ndarray
        debug      : dict  — bổ sung loss_task, task_margin, prompt_updated
        """
        inference_model = str(
            inference_model or self.cfg.inference_model
        ).strip().lower()
        if inference_model not in {"teacher", "student"}:
            raise ValueError(
                "inference_model must be 'teacher' or 'student', "
                f"got {inference_model!r}"
            )

        N = features.shape[0]

        # ── Phase 1a: Standard patch subsample ──────────────────────────────
        k = min(self.cfg.k_patches_std, N)
        idx = torch.randperm(N, device=self.device)[:k]
        feat_std  = features[idx]
        coord_std = coords[idx]

        # ── Phase 1b: Anchor + Teacher std forward ───────────────────────────
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_anchor      = self.anchor(feat_std, coord_std, self.ps).float()
            z_teacher_std = self.teacher(feat_std, coord_std, self.ps).float()

        # ── Phase 1c: OOD score ──────────────────────────────────────────────
        y_anchor_g  = F.softmax(self._global_logits(z_anchor),      dim=-1)
        y_teacher_g = F.softmax(self._global_logits(z_teacher_std), dim=-1)
        max_anchor_g  = y_anchor_g.max().item()
        max_teacher_g = y_teacher_g.max().item()

        ood_score = abs(1.0 - max_anchor_g / (max_teacher_g + self.cfg.eps))
        eta_eff = self.cfg.eta_base * torch.sigmoid(
            torch.tensor(ood_score - self.cfg.tau_ood, dtype=torch.float32)
        ).item()

        # ── Phase 1d: Task routing — V3: dùng z_teacher thay vì z_anchor ────
        # Lý do: z_teacher phản ánh current embedding space sau adaptation,
        # tốt hơn z_anchor frozen khi backbone đã drift do domain shift.
        # Consistent với PETAL: teacher là nguồn đáng tin nhất.
        if task_id is not None:
            t_adapt = task_id
        elif use_tcp_gate:
            t_adapt = self._tcp_route(z_teacher_std)   # V3: teacher routing
        else:
            t_adapt = None

        # ── Phase 1e: DaPC pseudo-label ──────────────────────────────────────
        if ood_score < self.cfg.tau_ood:
            tau_ap = self.cfg.tau_ap_ind
            beta   = self.cfg.beta_ind
        else:
            tau_ap = self.cfg.tau_ap_ood
            beta   = self.cfg.beta_ood

        if t_adapt is None:
            y_anchor_adapt   = y_anchor_g
            max_anchor_adapt = max_anchor_g
        else:
            y_anchor_adapt   = F.softmax(self._task_logits(z_anchor, t_adapt), dim=-1)
            max_anchor_adapt = y_anchor_adapt.max().item()

        use_aug = (max_anchor_g < tau_ap)

        if use_aug:
            y_tilde = self._aug_teacher_pred(features, coords, t_adapt, N)
        else:
            if t_adapt is None:
                y_tilde = F.softmax(self._global_logits(z_teacher_std), dim=-1)
            else:
                y_tilde = F.softmax(self._task_logits(z_teacher_std, t_adapt), dim=-1)

        if beta * y_tilde.max().item() > max_anchor_adapt:
            y_corrected = y_tilde.detach()
        else:
            y_corrected = 0.5 * (y_tilde + y_anchor_adapt).detach()

        # ── Phase 2: Student forward ─────────────────────────────────────────
        self.student.train()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_s = self.student(feat_std, coord_std, self.ps)
        z_s_f32 = z_s.float()

        logits_s = (
            self._global_logits(z_s_f32)
            if t_adapt is None
            else self._task_logits(z_s_f32, t_adapt)
        )

        # ── Phase 3: Loss computation ────────────────────────────────────────
        log_q = self._bayesian_log_q()

        loss_ce = -(
            y_corrected * F.log_softmax(logits_s, dim=-1)
        ).sum(dim=-1).mean()

        loss_petal = -self.cfg.spw * log_q + self.cfg.lam_ce * loss_ce

        loss_class = -(
            y_corrected * F.log_softmax(logits_s / self.cfg.tau_c, dim=-1)
        ).sum(dim=-1).mean()

        # L_task only applies to TCP routing. Naive CLASS-IL and TASK-IL do not
        # route with task prompts, so including it would alter gradients and FIM
        # using an objective that is absent from their inference path.
        if use_tcp_gate:
            task_scores_s = z_s_f32 @ self.task_prompts.T.detach()  # [1, T]
            sorted_scores = task_scores_s.sort(dim=-1, descending=True).values
            s_top1 = sorted_scores[:, 0]
            s_top2 = sorted_scores[:, 1]
            loss_task = F.relu(
                s_top2 - s_top1 + self.cfg.margin_task
            ).mean()
            gamma_task_eff = self.cfg.gamma_task
        else:
            loss_task = z_s_f32.new_zeros(())
            gamma_task_eff = 0.0

        # L_total
        loss_total = (
            loss_petal
            + self.cfg.gamma_class * loss_class
            + gamma_task_eff * loss_task
        )

        # ── Phase 4a: Backward ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss_total.backward()

        # ── Phase 4b: FIM computation ────────────────────────────────────────
        fisher_dict, fisher_flat = self._compute_fim()
        fim_threshold = self._find_quantile(fisher_flat, self.cfg.delta)

        # ── Phase 4c: Scale LR và step ───────────────────────────────────────
        for pg in self.optimizer.param_groups:
            pg['lr'] = eta_eff
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ── Phase 5: Teacher EMA ─────────────────────────────────────────────
        self._ema_update()

        # ── Phase 5b: Task Prompt EMA Update — V3 NEW (SwapPrompt-inspired) ──
        # Sau khi teacher EMA, z_teacher_std đã được tính từ teacher trước
        # update. Tính lại routing score trên teacher mới để đánh giá confidence.
        # Chỉ update khi gap đủ lớn (TPT confidence gate principle).
        prompt_updated = False
        task_margin_val = 0.0
        if task_id is None and use_tcp_gate and t_adapt is not None:
            with torch.no_grad():
                # Tính routing scores từ z_teacher_std (stable, no-grad)
                tcp_scores_ema = F.softmax(
                    z_teacher_std @ self.task_prompts.T, dim=-1
                )  # [1, T]
                sorted_tcp = tcp_scores_ema.sort(dim=-1, descending=True).values
                top1_score = sorted_tcp[0, 0].item()
                top2_score = sorted_tcp[0, 1].item()
                task_margin_val = top1_score - top2_score

                if task_margin_val > self.cfg.delta_margin:
                    # Update task_prompts[t_adapt] bằng EMA với z_teacher_std
                    self._update_task_prompt(t_adapt, z_teacher_std)
                    prompt_updated = True

        # ── Phase 6: FIM-based Parameter Restoration ─────────────────────────
        self._fim_restore(fisher_dict, fim_threshold)

        # ── Phase 7: Inference với TCP Confidence Gate ───────────────────────
        # Student prediction uses the post-step, post-FIM-restoration weights.
        # Teacher prediction uses the EMA-smoothed weights.
        inference_backbone = (
            self.student if inference_model == "student" else self.teacher
        )
        inference_backbone.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_inf = inference_backbone(feat_std, coord_std, self.ps).float()

        tcp_conf = 0.0
        if task_id is not None:
            logits_inf = self._task_logits(z_inf, task_id)
            pred_local = int(logits_inf.argmax(-1).item())
            pred_task  = task_id
            prob_inf   = F.softmax(logits_inf.float(), dim=-1)

        elif use_tcp_gate:
            # V3: routing tại inference cũng dùng task_prompts đã được update
            tcp_scores = F.softmax(z_inf.float() @ self.task_prompts.T, dim=-1)
            tcp_conf   = float(tcp_scores.max().item())

            if tcp_conf >= self.cfg.tau_task:
                t_hat      = int(tcp_scores.argmax(-1).item())
                logits_inf = self._task_logits(z_inf, t_hat)
                pred_local = int(logits_inf.argmax(-1).item())
                pred_task  = t_hat
                prob_inf   = F.softmax(logits_inf.float(), dim=-1)
            else:
                g_logits = self._global_logits(z_inf)
                g_pred   = int(g_logits.argmax(-1).item())
                pred_task, pred_local = self._global_to_task_local(g_pred)
                g_prob   = F.softmax(g_logits.float(), dim=-1)
                start, end = self.task_class_ranges[pred_task]
                prob_inf = g_prob[:, start:end + 1]

        else:
            logits_inf = self._global_logits(z_inf)
            g_pred     = int(logits_inf.argmax(-1).item())
            pred_task, pred_local = self._global_to_task_local(g_pred)
            prob_inf   = F.softmax(logits_inf.float(), dim=-1)

        prob_np = prob_inf.squeeze(0).detach().cpu().numpy()

        debug = {
            "ood_score":      ood_score,
            "eta_eff":        eta_eff,
            "tcp_conf":       tcp_conf,
            "t_adapt":        t_adapt,
            "use_aug":        use_aug,
            "loss_petal":     float(loss_petal.detach().item()),
            "loss_class":     float(loss_class.detach().item()),
            "loss_task":      float(loss_task.detach().item()),    # V3 NEW
            "loss_total":     float(loss_total.detach().item()),
            "gamma_task_eff": float(gamma_task_eff),
            "loss_task_active": bool(use_tcp_gate),
            "task_margin":    task_margin_val,                      # V3 NEW
            "prompt_updated": prompt_updated,                       # V3 NEW
            "inference_model": inference_model,
        }
        return pred_local, pred_task, prob_np, debug

    def reset_to_source(self) -> None:
        """
        Reset student, teacher về anchor θ_0 và task_prompts về source.
        V3: bổ sung reset task_prompts để tránh drift tích lũy giữa tasks.
        """
        merged_sd = {k: v.to(self.device) for k, v in self.anchor_sd.items()}

        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if name in merged_sd:
                    param.data.copy_(merged_sd[name])

        self.teacher.load_state_dict(merged_sd, strict=True)

        # V3: Reset task_prompts về source
        self.task_prompts.data.copy_(self.task_prompts_source)

        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=self.cfg.eta_base, weight_decay=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers (giữ nguyên từ v2, chỉ bổ sung _update_task_prompt)
    # ──────────────────────────────────────────────────────────────────────────

    def _make_backbone(self, base: nn.Module, sd: dict, train: bool) -> nn.Module:
        bb = copy.deepcopy(base).to(self.device)
        bb.load_state_dict(sd, strict=True)
        if train:
            bb.train()
            for p in bb.parameters():
                p.requires_grad_(False)
            for m in bb.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad_(True)
        else:
            bb.eval()
            for p in bb.parameters():
                p.requires_grad_(False)
        return bb

    def _task_logits(self, Z: Tensor, t: int) -> Tensor:
        return F.linear(Z, self.per_task_w[t], self.per_task_b[t])

    def _global_logits(self, Z: Tensor) -> Tensor:
        return F.linear(Z, self.global_w, self.global_b)

    def _tcp_route(self, Z: Tensor) -> int:
        """TCP routing dùng current task_prompts (có thể đã được EMA update)."""
        with torch.no_grad():
            scores = Z.float() @ self.task_prompts.T
            return int(scores.argmax(-1).item())

    def _global_to_task_local(self, g_pred: int) -> tuple[int, int]:
        for t, (start, end) in self.task_class_ranges.items():
            if start <= g_pred <= end:
                return t, g_pred - start
        return 0, 0

    def _aug_teacher_pred(
        self,
        features: Tensor,
        coords: Tensor,
        t_adapt: Optional[int],
        N: int,
    ) -> Tensor:
        n_aug = max(1, int(self.cfg.r_patch * self.cfg.k_patches_std))
        n_aug = min(n_aug, N)
        preds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(self.cfg.K):
                aug_idx = torch.randperm(N, device=self.device)[:n_aug]
                z_aug   = self.teacher(features[aug_idx], coords[aug_idx], self.ps).float()
                if t_adapt is None:
                    preds.append(F.softmax(self._global_logits(z_aug), dim=-1))
                else:
                    preds.append(F.softmax(self._task_logits(z_aug, t_adapt), dim=-1))
        return torch.stack(preds).mean(0)

    def _bayesian_log_q(self) -> Tensor:
        log_q = torch.tensor(0.0, device=self.device)
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.mean_sd:
                continue
            mu     = self.mean_sd[name].to(self.device)
            sigma2 = self.var_sd[name].to(self.device)
            log_q  = log_q - 0.5 * (
                (param - mu).pow(2) / (sigma2 + self.cfg.eps)
            ).sum()
        return log_q

    def _compute_fim(self) -> tuple[dict, Tensor]:
        fisher_dict: dict[str, Tensor] = {}
        fisher_vals: list[Tensor] = []
        for name, param in self.student.named_parameters():
            if param.requires_grad and param.grad is not None:
                fim_p = param.grad.data.pow(2).clone()
                fisher_dict[name] = fim_p
                fisher_vals.append(fim_p.reshape(-1))
        fisher_flat = torch.cat(fisher_vals) if fisher_vals else torch.zeros(1, device=self.device)
        return fisher_dict, fisher_flat

    @staticmethod
    def _find_quantile(arr: Tensor, perc: float) -> Tensor:
        arr_sorted = arr.sort().values
        n = arr_sorted.numel()
        if n == 0:
            return torch.tensor(0.0, device=arr.device)
        frac_idx  = perc * (n - 1)
        low_idx   = int(frac_idx)
        high_idx  = min(low_idx + 1, n - 1)
        frac_part = frac_idx - low_idx
        return arr_sorted[low_idx] + (arr_sorted[high_idx] - arr_sorted[low_idx]) * frac_part

    def _fim_restore(self, fisher_dict: dict, threshold: Tensor) -> None:
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if not param.requires_grad or name not in fisher_dict:
                    continue
                anchor_val = self.anchor_sd[name].to(self.device)
                mask = (fisher_dict[name] < threshold).float()
                param.data.copy_(anchor_val * mask + param.data * (1.0 - mask))

    def _ema_update(self) -> None:
        alpha = self.cfg.ema_alpha
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    def _update_task_prompt(self, t: int, z: Tensor) -> None:
        """
        Phase 5b — Task Prompt EMA Update (SwapPrompt-inspired).

        Kéo task_prompts[t] về phía z_teacher của slide hiện tại.
        Chỉ gọi khi routing confidence gap > delta_margin (TPT gate).

        Parameters
        ----------
        t : int
            Task index được route đến (t_adapt từ Phase 1d).
        z : Tensor  shape [1, 768]
            z_teacher_std — embedding của slide hiện tại từ teacher.
        """
        alpha = self.cfg.alpha_task_prompt
        with torch.no_grad():
            self.task_prompts[t] = (
                alpha * self.task_prompts[t]
                + (1.0 - alpha) * z.squeeze(0)
            )
