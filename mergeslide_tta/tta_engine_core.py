"""
mergeslide_tta/tta_engine_core.py
==================================
MergeSlide-TTA-Core — kiến trúc lõi, KHÔNG dùng TCP routing, KHÔNG dùng
L_task, KHÔNG dùng Task Prompt EMA. Chỉ tối ưu naive/global objective
(flat softmax trên toàn bộ TOTAL_CLASSES).

Tài liệu thiết kế tương ứng: MergeSlide-TTA-Core_naive-focus_v1.md
(§4-§10). File này KHÔNG sửa tta_engine_v3.py — theo đúng nguyên tắc
"không sửa file gốc" của project, đây là module bổ sung độc lập.

So với tta_engine_v3.py, các thay đổi:
    - BỎ: task_prompts, TCP routing (_tcp_route), L_task (Module C thành
      phần 3), Phase 5b (Task Prompt EMA Update), Phase 7 TCP Confidence
      Gate. Constructor không còn cần `task_prompts` hay `per_task_mlp_weights`.
    - GIỮ NGUYÊN logic: Module A (bag augmentation), Module B (DaPC, giờ
      thuần global — không còn nhánh t_adapt), Module C' (L_PETAL + L_class,
      có gating bởi Module F), Module D (FIM restoration động), Module E
      (Adaptive LR theo OOD score).
    - MỚI — Module F: Reliable & Non-redundant Sample Selection dựa trên
      độ đồng thuận (consensus) giữa K augmented views của CHÍNH slide đó
      (không dùng EMA lịch sử slide-stream). Phân rã:
          H(mean)  =  H_bar (bất định nội tại)  +  JSD_K (bất định do bất đồng)
      Sample bị gate (S(x)=0) => bỏ qua backward + FIM hoàn toàn, chỉ suy luận.
    - MỚI (tuỳ chọn, mặc định TẮT) — Module D': Fisher-weighted anti-forgetting
      regularizer kiểu EATA, tính một lần trước khi TTA bắt đầu (offline),
      cộng thêm vào loss dưới dạng soft L2 penalty. Không thay thế Module D
      mặc định; hai module có thể bật cùng lúc để ablate (AC-6).
    - MỚI (tuỳ chọn, mặc định n_steps=1 = hành vi cũ hệt trước) — Module G:
      Multi-step adaptation per slide. Chỉ áp dụng cho slide đã qua cổng
      Module F (sample_active=True) — slide bị gate vẫn bỏ qua hoàn toàn
      như trước, KHÔNG có gì thay đổi cho nhánh đó.
      Thiết kế (xem thảo luận rủi ro collapse dựa trên zMEMO.pdf, mục
      "preliminary experiments" của chính tác giả MEMO):
        * y_corrected (DaPC) tính MỘT LẦN, dùng chung cho cả N bước —
          không tính lại mỗi bước (anchor/teacher không đổi trong 1 slide).
        * Bước đầu tiên (step 0) tái sử dụng ĐÚNG feat_std/coord_std đã
          dùng cho Phase 1b — đảm bảo n_steps=1 cho kết quả HỆT như trước
          khi chưa có Module G (backward-compatible tuyệt đối).
        * Các bước sau (step 1..N-1), nếu resample_per_step=True (mặc
          định), lấy lại sub-bag MỚI mỗi bước — tránh overfit vào 1
          subsample cố định với pseudo-label có thể sai (AC-16).
        * FIM tích luỹ (cộng dồn bình phương gradient) qua cả N bước,
          Module D chỉ restore MỘT LẦN sau khi kết thúc N bước — không
          restore giữa chừng (tránh xoá tiến độ của các bước trước đó).
        * Teacher EMA cũng chỉ update MỘT LẦN sau N bước.
        * step_lr_policy quyết định learning rate mỗi bước: "same" (giữ
          nguyên eta_eff mỗi bước — tổng "quãng đường" ~N lần so với 1
          bước, rủi ro cao nhất), "div_n" (eta_eff/N — tổng quãng đường
          gần như giữ nguyên, an toàn nhất), "div_sqrt_n" (trung dung).

Modules:
    A   : WSI Bag-level Augmentation        (Frozen-to-Paraffin, MICCAI 2021)
    B   : DaPC Pseudo-label Correction       (WeiContrast-DaPC, IEEE TETC 2025) — global only
    C'  : L_PETAL + L_class, có Module-F gating
    D   : FIM-based Parameter Restoration    (PETAL, CVPR 2023) — online, mặc định BẬT
    D'  : Fisher-weighted regularizer        (EATA-inspired) — offline, mặc định TẮT
    E   : Adaptive LR via OOD score          (NOTE-inspired, NeurIPS 2022)
    F   : Consensus-based Sample Selection   (EATA-inspired, K-view JSD)
    G   : Multi-step adaptation per slide    (MỚI, mặc định n_steps=1 = tắt)
    H   : Regularizer bám-nguồn có thể chọn  (MỚI, "swag" mặc định | "l2_anchor")

Usage::
    adapter = MergeSlide_TTA_Adapter_Core(...)
    adapter.reset_task_boundary()          # gọi 1 lần khi bắt đầu mỗi task
    pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(features, coords)
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTAConfig_Core:
    # Module A — WSI Bag Augmentation (Frozen-to-Paraffin)
    K: int          = 8      # số augmented views (dùng chung cho Module A và Module F)
    r_patch: float  = 0.75   # fraction của k_patches_std cho mỗi aug view

    # Module B — DaPC pseudo-label correction (global only, không t_adapt)
    tau_ap_ind: float = 0.92
    tau_ap_ood: float = 0.70
    beta_ind: float   = 1.2
    beta_ood: float   = 1.5

    # Module C' — PETAL loss (global)
    spw: float         = 1e-6
    lam_ce: float      = 1.0
    gamma_class: float = 1.0
    tau_c: float       = 0.07

    # Module D — FIM Restoration (online, mặc định bật)
    use_fim_restore: bool = True
    delta: float           = 0.03

    # Module H — Chọn loại regularizer bám-nguồn (source-anchoring), MỚI.
    # "swag" (mặc định, hành vi cũ hệt trước): -spw * log q(theta_s), q là
    #   Gaussian posterior ước lượng bằng SWAG (mean_sd, var_sd), có
    #   chuẩn hoá theo phương sai từng tham số.
    # "l2_anchor": beta * sum((theta_s - theta_0)^2), theta_0 là checkpoint
    #   đã merge (anchor_sd) — L2 đơn giản, KHÔNG chuẩn hoá theo phương sai,
    #   theo đúng công thức l2_anchor_loss() trong ADAPT-Slide (EATA
    #   simplified). Không cần mean_sd/var_sd để tính (vẫn phải truyền vào
    #   constructor vì lý do tương thích chữ ký, nhưng không được dùng).
    # LƯU Ý QUAN TRỌNG: hai regularizer KHÔNG cùng thang đo — spw (mặc định
    # 1e-6) được hiệu chỉnh cho log-probability có chuẩn hoá phương sai,
    # còn l2_anchor_beta (mặc định 1.0, theo đúng default của ADAPT-Slide)
    # là trọng số cho tổng bình phương sai khác THÔ, không chuẩn hoá. TUYỆT
    # ĐỐI không tái dùng giá trị spw đã tune cho l2_anchor_beta — cần sweep
    # riêng (xem AC-18 trong hướng dẫn kèm theo).
    regularizer_type: str = "swag"
    l2_anchor_beta: float = 1.0

    # Module D' — Fisher-weighted regularizer (offline, mặc định tắt — AC-6)
    # LƯU Ý: omega(theta) phụ thuộc checkpoint đã merge của TỪNG FOLD, nên
    # KHÔNG lưu path cố định ở đây. TTAConfig_Core chỉ giữ cờ bật/tắt +
    # trọng số; dict omega thực tế được caller (test_tta_core.py) load
    # riêng cho mỗi fold và truyền trực tiếp vào constructor của adapter
    # qua tham số `fisher_omega` (xem MergeSlide_TTA_Adapter_Core.__init__).
    use_fisher_reg: bool = False
    beta_fisher: float   = 0.0

    # Module E — Adaptive LR
    eta_base: float = 1e-4
    tau_ood: float  = 0.5

    # EMA teacher
    ema_alpha: float = 0.999

    # Module F — Consensus-based Sample Selection (K-view JSD, MỚI)
    use_module_f: bool       = True
    conf_window: int         = 300     # kích thước cửa sổ trượt cho S_conf (H_bar)
    agree_window: int        = 300     # kích thước cửa sổ trượt cho S_agree (JSD_K)
    conf_percentile: float   = 0.5     # giữ lại % thấp nhất theo H_bar (0.5 = giữ nửa "tự tin" hơn)
    agree_percentile: float  = 0.5     # giữ lại % thấp nhất theo JSD_K (0.5 = giữ nửa "đồng thuận" hơn)
    min_window_fill: int     = 30      # số sample tối thiểu trong cửa sổ trước khi gating có hiệu lực

    # Module G — Multi-step adaptation per slide (MỚI, mặc định TẮT = hành vi cũ)
    # n_steps=1 tuyệt đối tương đương hành vi trước khi có Module G (đã kiểm
    # chứng bằng thiết kế: step đầu tiên luôn tái dùng feat_std/coord_std gốc).
    n_steps: int              = 1
    # "same": eta_step = eta_eff mỗi bước (tổng quãng đường ~N lần, rủi ro cao nhất)
    # "div_n": eta_step = eta_eff / N (tổng quãng đường ~giữ nguyên, an toàn nhất)
    # "div_sqrt_n": eta_step = eta_eff / sqrt(N) (trung dung)
    step_lr_policy: str       = "same"
    # True (mặc định): mỗi bước (trừ bước 0) lấy lại sub-bag MỚI, tránh
    # overfit vào 1 subsample cố định. False: dùng chung 1 subsample cho cả
    # N bước (baseline để ablate AC-16, giống "N epoch trên 1 batch cố định").
    resample_per_step: bool   = True

    # Misc
    eps: float         = 1e-6
    k_patches_std: int = 400


# ──────────────────────────────────────────────────────────────────────────────
# Module F helper: sliding-window percentile gate
# ──────────────────────────────────────────────────────────────────────────────

class _PercentileGate:
    """
    Cửa sổ trượt (deque) lưu giá trị gần đây, quyết định giữ/loại sample
    hiện tại dựa trên percentile của chính cửa sổ đó (KHÔNG bao gồm giá
    trị hiện tại — percentile tính trước, rồi mới append).

    Trước khi cửa sổ đạt `min_fill` phần tử: luôn giữ (keep=True), tránh
    hành vi suy biến khi chưa đủ dữ liệu để ước lượng percentile.
    """

    def __init__(self, window: int, percentile: float, min_fill: int) -> None:
        self.buf: "collections.deque[float]" = collections.deque(maxlen=window)
        self.percentile = percentile
        self.min_fill   = min_fill

    def reset(self) -> None:
        self.buf.clear()

    def decide(self, value: float) -> tuple[bool, float]:
        if len(self.buf) < self.min_fill:
            keep, thresh = True, float("nan")
        else:
            thresh = float(np.percentile(np.asarray(self.buf, dtype=np.float64),
                                          self.percentile * 100.0))
            keep = bool(value <= thresh)
        self.buf.append(value)
        return keep, thresh


# ──────────────────────────────────────────────────────────────────────────────
# Adapter Core
# ──────────────────────────────────────────────────────────────────────────────

class MergeSlide_TTA_Adapter_Core:
    """
    Online, exemplar-free TTA adapter — kiến trúc lõi (no TCP, no L_task).

    So với MergeSlide_TTA_Adapter_v3, constructor KHÔNG cần `task_prompts`
    hay `per_task_mlp_weights` — chỉ cần global MLP weights (prompt
    prototypes toàn bộ TOTAL_CLASSES) và `task_class_ranges` (chỉ dùng để
    quy đổi global prediction → (task, local class) cho mục đích logging/
    đánh giá theo task, không dùng cho adaptation hay routing).
    """

    def __init__(
        self,
        base_vision_encoder: nn.Module,
        backbone_sd: dict,
        mean_sd: dict,
        var_sd: dict,
        global_mlp_weight: Tensor,
        global_mlp_bias: Tensor,
        task_class_ranges: dict,
        cfg: TTAConfig_Core,
        device: torch.device,
        fisher_omega: Optional[dict[str, Tensor]] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        self.global_w = global_mlp_weight.to(device)
        self.global_b = global_mlp_bias.to(device)
        self.task_class_ranges = task_class_ranges

        self.mean_sd = mean_sd
        self.var_sd  = var_sd

        from mergeslide_tta.constants import TITAN_PS_ARG
        self.ps = torch.tensor(TITAN_PS_ARG).int().to(device)

        self.anchor_sd = {k: v.clone().cpu() for k, v in backbone_sd.items()}

        self.anchor  = self._make_backbone(base_vision_encoder, backbone_sd, train=False)
        self.student = self._make_backbone(base_vision_encoder, backbone_sd, train=True)
        self.teacher = self._make_backbone(base_vision_encoder, backbone_sd, train=False)

        self.ln_param_names = [
            n for n, p in self.student.named_parameters() if p.requires_grad
        ]

        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=cfg.eta_base, weight_decay=0.0)

        # ── Module D' — Fisher importance (offline, optional) ────────────────
        # fisher_omega được caller load riêng theo fold (checkpoint mỗi fold
        # khác nhau) và truyền trực tiếp vào đây — xem test_tta_core.py.
        self.fisher_omega: Optional[dict[str, Tensor]] = fisher_omega
        if cfg.use_fisher_reg and self.fisher_omega is None:
            raise ValueError(
                "use_fisher_reg=True nhưng fisher_omega không được cung cấp cho constructor. "
                "Chạy tools/compute_fisher_importance.py trước để tạo file .pt theo từng fold, "
                "rồi load và truyền qua tham số fisher_omega= khi khởi tạo adapter."
            )

        # ── Module F — sliding-window gates (reset theo task, xem reset_task_boundary) ──
        self._conf_gate  = _PercentileGate(cfg.conf_window,  cfg.conf_percentile,  cfg.min_window_fill)
        self._agree_gate = _PercentileGate(cfg.agree_window, cfg.agree_percentile, cfg.min_window_fill)

        # ── Module G — validate step_lr_policy sớm (fail-fast trước khi chạy TTA) ──
        if cfg.step_lr_policy not in ("same", "div_n", "div_sqrt_n"):
            raise ValueError(
                f"step_lr_policy phải là 'same', 'div_n', hoặc 'div_sqrt_n', "
                f"nhận được: {cfg.step_lr_policy!r}"
            )
        if cfg.n_steps < 1:
            raise ValueError(f"n_steps phải >= 1, nhận được: {cfg.n_steps}")

        # ── Module H — validate regularizer_type sớm ─────────────────────────
        if cfg.regularizer_type not in ("swag", "l2_anchor"):
            raise ValueError(
                f"regularizer_type phải là 'swag' hoặc 'l2_anchor', "
                f"nhận được: {cfg.regularizer_type!r}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def adapt_and_predict(
        self,
        features: Tensor,
        coords: Tensor,
    ) -> tuple[int, int, "np.ndarray", dict]:
        """
        Naive-only forward: KHÔNG routing, KHÔNG L_task.

        Returns
        -------
        pred_local : int   — local class index trong task được suy ra
                              (từ global prediction, xem _global_to_task_local)
        pred_task  : int   — task index suy ra từ global prediction
        prob_np    : np.ndarray  — xác suất trong không gian LOCAL của pred_task
                                    (để tương thích với eval loop kiểu v3;
                                     AUC/metrics chi tiết nên dùng "prob_global"
                                     trong debug nếu cần toàn bộ 13 lớp)
        debug      : dict  — bổ sung h_bar, jsd_k, sample_active, conf_thresh,
                              agree_thresh so với v3
        """
        N = features.shape[0]
        eps = self.cfg.eps

        # ── Phase 0 (Module F): K-view teacher predictions — LUÔN tính ──────
        # Đây là thay đổi thiết kế so với v3: Module A giờ luôn chạy (không
        # điều kiện theo use_aug của DaPC), vì Module F cần nó cho MỌI slide.
        # Kết quả này được TÁI SỬ DỤNG cho DaPC (Phase 1e) khi use_aug=True,
        # tránh forward pass thêm — đúng như đề xuất "tận dụng compute có sẵn".
        p_k = self._k_view_teacher_predictions(features, coords, N)   # [K, C]
        h_bar, jsd_k, p_bar = self._module_f_score(p_k, eps)

        if self.cfg.use_module_f:
            keep_conf,  conf_thresh  = self._conf_gate.decide(h_bar)
            keep_agree, agree_thresh = self._agree_gate.decide(jsd_k)
            sample_active = bool(keep_conf and keep_agree)
        else:
            keep_conf = keep_agree = sample_active = True
            conf_thresh = agree_thresh = float("nan")

        # ── Phase 1a: Standard patch subsample ──────────────────────────────
        k = min(self.cfg.k_patches_std, N)
        idx = torch.randperm(N, device=self.device)[:k]
        feat_std  = features[idx]
        coord_std = coords[idx]

        # ── Phase 1b: Anchor + Teacher std forward (cần cho OOD score + inference) ──
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_anchor      = self.anchor(feat_std, coord_std, self.ps).float()
            z_teacher_std = self.teacher(feat_std, coord_std, self.ps).float()

        y_anchor_g  = F.softmax(self._global_logits(z_anchor),      dim=-1)
        y_teacher_g = F.softmax(self._global_logits(z_teacher_std), dim=-1)
        max_anchor_g  = y_anchor_g.max().item()
        max_teacher_g = y_teacher_g.max().item()

        ood_score = abs(1.0 - max_anchor_g / (max_teacher_g + eps))
        eta_eff = self.cfg.eta_base * torch.sigmoid(
            torch.tensor(ood_score - self.cfg.tau_ood, dtype=torch.float32)
        ).item()

        # ── MODULE F GATE: nếu sample không đáng tin → bỏ qua adaptation ────
        if not sample_active:
            self.teacher.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                z_inf = self.teacher(feat_std, coord_std, self.ps).float()
            logits_inf = self._global_logits(z_inf)
            g_pred     = int(logits_inf.argmax(-1).item())
            pred_task, pred_local = self._global_to_task_local(g_pred)
            g_prob   = F.softmax(logits_inf.float(), dim=-1)
            start, end = self.task_class_ranges[pred_task]
            prob_local = g_prob[:, start:end + 1]
            prob_np = prob_local.squeeze(0).detach().cpu().numpy()

            debug = {
                "sample_active": False,
                "keep_conf": keep_conf, "keep_agree": keep_agree,
                "h_bar": h_bar, "jsd_k": jsd_k,
                "conf_thresh": conf_thresh, "agree_thresh": agree_thresh,
                "ood_score": ood_score, "eta_eff": eta_eff,
                "use_aug": False,
                "n_steps_used": 0,
                "eta_step": float("nan"),
                "loss_petal": float("nan"), "loss_class": float("nan"),
                "loss_total": float("nan"),
                "loss_reg": float("nan"),
                "regularizer_type": self.cfg.regularizer_type,
                "prob_global": g_prob.squeeze(0).detach().cpu().numpy(),
            }
            return pred_local, pred_task, prob_np, debug

        # ── Phase 1e: DaPC pseudo-label (global only) ────────────────────────
        if ood_score < self.cfg.tau_ood:
            tau_ap, beta = self.cfg.tau_ap_ind, self.cfg.beta_ind
        else:
            tau_ap, beta = self.cfg.tau_ap_ood, self.cfg.beta_ood

        use_aug = (max_anchor_g < tau_ap)
        if use_aug:
            y_tilde = p_bar.unsqueeze(0)          # tái sử dụng Module F K-view avg
        else:
            y_tilde = y_teacher_g

        if beta * y_tilde.max().item() > max_anchor_g:
            y_corrected = y_tilde.detach()
        else:
            y_corrected = 0.5 * (y_tilde + y_anchor_g).detach()

        # ── Phase 2-4 (Module G): N-step student adaptation ──────────────────
        # n_steps=1 (mặc định) tương đương HỆT hành vi trước khi có Module G:
        # bước duy nhất luôn tái dùng feat_std/coord_std đã tính OOD score ở
        # Phase 1b, không có gì thay đổi. Chỉ khi n_steps>1 mới có khác biệt.
        n_steps = self.cfg.n_steps
        eta_step = self._compute_step_lr(eta_eff)

        fisher_accum: dict[str, Tensor] = {}
        loss_petal_sum = 0.0
        loss_class_sum = 0.0
        loss_total_sum = 0.0
        loss_reg_sum   = 0.0
        feat_step, coord_step = feat_std, coord_std  # step 0 luôn dùng subsample gốc

        for step_i in range(n_steps):
            if step_i > 0 and self.cfg.resample_per_step:
                idx_step = torch.randperm(N, device=self.device)[:k]
                feat_step  = features[idx_step]
                coord_step = coords[idx_step]
            # step_i > 0 và resample_per_step=False: giữ nguyên feat_step/coord_step
            # của vòng lặp trước (dùng chung 1 subsample cho cả N bước — AC-16).

            # ── Phase 2: Student forward (global logits, không t_adapt) ─────
            self.student.train()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                z_s = self.student(feat_step, coord_step, self.ps)
            z_s_f32 = z_s.float()
            logits_s = self._global_logits(z_s_f32)

            # ── Phase 3: Loss computation (Module C' + Module H regularizer) ──
            if self.cfg.regularizer_type == "l2_anchor":
                reg_term = self.cfg.l2_anchor_beta * self._l2_anchor_loss()
            else:  # "swag" — mặc định, hành vi cũ hệt trước
                log_q = self._bayesian_log_q()
                reg_term = -self.cfg.spw * log_q

            loss_ce = -(
                y_corrected * F.log_softmax(logits_s, dim=-1)
            ).sum(dim=-1).mean()

            loss_petal = reg_term + self.cfg.lam_ce * loss_ce

            loss_class = -(
                y_corrected * F.log_softmax(logits_s / self.cfg.tau_c, dim=-1)
            ).sum(dim=-1).mean()

            loss_total = loss_petal + self.cfg.gamma_class * loss_class

            # Module D' — Fisher-weighted regularizer (optional, AC-6)
            if self.cfg.use_fisher_reg and self.fisher_omega is not None:
                reg = z_s_f32.new_zeros(())
                for name, param in self.student.named_parameters():
                    if not param.requires_grad or name not in self.fisher_omega:
                        continue
                    omega      = self.fisher_omega[name].to(self.device)
                    anchor_val = self.anchor_sd[name].to(self.device)
                    reg = reg + (omega * (param - anchor_val).pow(2)).sum()
                loss_total = loss_total + self.cfg.beta_fisher * reg

            # ── Phase 4a: Backward (mỗi bước) ────────────────────────────────
            self.optimizer.zero_grad()
            loss_total.backward()

            # ── Phase 4b: FIM bước này — CỘNG DỒN vào fisher_accum, KHÔNG
            # restore ngay (restore một lần duy nhất sau khi hết N bước, xem
            # Phase 6) để tránh xoá tiến độ của các bước trước đó. ─────────
            step_fisher_dict, _ = self._compute_fim()
            for name, val in step_fisher_dict.items():
                if name in fisher_accum:
                    fisher_accum[name] = fisher_accum[name] + val
                else:
                    fisher_accum[name] = val.clone()

            # ── Phase 4c: Scale LR (theo step_lr_policy) và step ─────────────
            for pg in self.optimizer.param_groups:
                pg['lr'] = eta_step
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_petal_sum += float(loss_petal.detach().item())
            loss_class_sum += float(loss_class.detach().item())
            loss_total_sum += float(loss_total.detach().item())
            loss_reg_sum   += float(reg_term.detach().item())

        fisher_flat = (
            torch.cat([v.reshape(-1) for v in fisher_accum.values()])
            if fisher_accum else torch.zeros(1, device=self.device)
        )
        fim_threshold = self._find_quantile(fisher_flat, self.cfg.delta)

        # ── Phase 5: Teacher EMA (một lần duy nhất, sau khi hết N bước) ─────
        self._ema_update()

        # ── Phase 6: FIM-based Parameter Restoration (Module D, một lần) ────
        if self.cfg.use_fim_restore:
            self._fim_restore(fisher_accum, fim_threshold)

        # ── Phase 7: Inference (naive, KHÔNG TCP) — dùng feat_std/coord_std
        # GỐC (không phải feat_step cuối cùng), để suy luận không phụ thuộc
        # vào sub-bag ngẫu nhiên nào được lấy ở bước cuối. ───────────────────
        self.teacher.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_inf = self.teacher(feat_std, coord_std, self.ps).float()
        logits_inf = self._global_logits(z_inf)
        g_pred     = int(logits_inf.argmax(-1).item())
        pred_task, pred_local = self._global_to_task_local(g_pred)
        g_prob     = F.softmax(logits_inf.float(), dim=-1)
        start, end = self.task_class_ranges[pred_task]
        prob_local = g_prob[:, start:end + 1]
        prob_np    = prob_local.squeeze(0).detach().cpu().numpy()

        debug = {
            "sample_active": True,
            "keep_conf": keep_conf, "keep_agree": keep_agree,
            "h_bar": h_bar, "jsd_k": jsd_k,
            "conf_thresh": conf_thresh, "agree_thresh": agree_thresh,
            "ood_score": ood_score, "eta_eff": eta_eff, "use_aug": use_aug,
            "n_steps_used": n_steps,
            "eta_step": eta_step,
            "loss_petal": loss_petal_sum / n_steps,
            "loss_class": loss_class_sum / n_steps,
            "loss_total": loss_total_sum / n_steps,
            "loss_reg": loss_reg_sum / n_steps,
            "regularizer_type": self.cfg.regularizer_type,
            "prob_global": g_prob.squeeze(0).detach().cpu().numpy(),
        }
        return pred_local, pred_task, prob_np, debug

    def reset_adaptation_state(self) -> None:
        """Reset online model + optimizer (dùng cho ablation --reset_per_slide)."""
        merged_sd = {k: v.to(self.device) for k, v in self.anchor_sd.items()}
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if name in merged_sd:
                    param.data.copy_(merged_sd[name])
        self.teacher.load_state_dict(merged_sd, strict=True)

        ln_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(ln_params, lr=self.cfg.eta_base, weight_decay=0.0)

    def reset_task_boundary(self) -> None:
        """
        Reset trạng thái Module F (cửa sổ trượt S_conf/S_agree) khi bắt đầu
        một task mới. KHÔNG reset model — model tiếp tục continual adaptation
        xuyên suốt task stream (đúng thiết kế CLASS-IL mặc định của v3).

        Gọi 1 lần ở đầu vòng lặp mỗi task trong entrypoint script.
        """
        self._conf_gate.reset()
        self._agree_gate.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
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

    def _global_logits(self, Z: Tensor) -> Tensor:
        return F.linear(Z, self.global_w, self.global_b)

    def _global_to_task_local(self, g_pred: int) -> tuple[int, int]:
        for t, (start, end) in self.task_class_ranges.items():
            if start <= g_pred <= end:
                return t, g_pred - start
        return 0, 0

    def _compute_step_lr(self, eta_eff: float) -> float:
        """
        Module G: quy đổi eta_eff (Module E — adaptive theo OOD score) thành
        learning rate cho MỖI bước trong N bước, theo step_lr_policy.
        n_steps=1 luôn cho kết quả = eta_eff bất kể policy nào (đảm bảo
        tương thích ngược tuyệt đối với hành vi trước khi có Module G).
        """
        n = max(1, self.cfg.n_steps)
        if n == 1:
            return eta_eff
        if self.cfg.step_lr_policy == "div_n":
            return eta_eff / n
        if self.cfg.step_lr_policy == "div_sqrt_n":
            return eta_eff / (n ** 0.5)
        return eta_eff  # "same"

    def _k_view_teacher_predictions(self, features: Tensor, coords: Tensor, N: int) -> Tensor:
        """
        Module A + Module F input: K forward pass qua teacher trên K sub-bag
        ngẫu nhiên, trả về [K, C] softmax (global, KHÔNG t_adapt).
        """
        n_aug = max(1, int(self.cfg.r_patch * self.cfg.k_patches_std))
        n_aug = min(n_aug, N)
        preds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(self.cfg.K):
                aug_idx = torch.randperm(N, device=self.device)[:n_aug]
                z_aug = self.teacher(features[aug_idx], coords[aug_idx], self.ps).float()
                preds.append(F.softmax(self._global_logits(z_aug), dim=-1))
        return torch.stack(preds, dim=0)   # [K, C]

    @staticmethod
    def _module_f_score(p_k: Tensor, eps: float) -> tuple[float, float, Tensor]:
        """
        Phân rã entropy theo Generalized Jensen-Shannon Divergence (§10.2,
        tài liệu nghiên cứu):
            H(p_bar) = H_bar + JSD_K
        p_k: [K, C] — softmax riêng lẻ từng view.
        Trả về (H_bar, JSD_K, p_bar) — p_bar để tái sử dụng cho DaPC.
        """
        p_bar = p_k.mean(dim=0)                                             # [C]
        h_individual = -(p_k * (p_k + eps).log()).sum(dim=-1)               # [K]
        h_bar = float(h_individual.mean().item())
        h_mean = float((-(p_bar * (p_bar + eps).log()).sum(dim=-1)).item())
        jsd_k = max(0.0, h_mean - h_bar)   # clamp: sai số float có thể cho giá trị âm rất nhỏ
        return h_bar, jsd_k, p_bar

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

    def _l2_anchor_loss(self) -> Tensor:
        """
        Module H (regularizer_type="l2_anchor"): L2 đơn giản về checkpoint
        đã merge (anchor_sd = theta_0), KHÔNG chuẩn hoá theo phương sai —
        đúng công thức l2_anchor_loss() trong ADAPT-Slide (adapt_slide/
        tta_losses.py):
            sum_i ((theta_s_i - theta_0_i)^2).sum()
        Trọng số áp dụng ở nơi gọi (self.cfg.l2_anchor_beta), không nhân
        sẵn trong hàm này — giữ hàm thuần "khoảng cách", giống thiết kế gốc.
        """
        reg = torch.tensor(0.0, device=self.device)
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            anchor_val = self.anchor_sd[name].to(self.device)
            reg = reg + ((param - anchor_val) ** 2).sum()
        return reg

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
