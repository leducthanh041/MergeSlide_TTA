"""
mergeslide_tta/tta_engine_core.py
==================================
MergeSlide-TTA-Unified — HAI đường adapt riêng biệt trong cùng 1 engine:

    mode="naive" / "task_il"  ->  kiến trúc MỚI (Module A-H, xem
        07_MergeSlide-TTA-Unified.md): KHÔNG L_task, KHÔNG Task Prompt EMA,
        CÓ Module F (consensus sample selection), Module G (multi-step),
        Module H (regularizer bám-nguồn chọn được).

    mode="tcp"  ->  kiến trúc CŨ y hệt tta_engine_v3.py gốc (xem
        06. PETAL_WSI.md), PHỤC HỒI ĐẦY ĐỦ, không rút gọn:
            - DaPC pseudo-label theo t_adapt (task-conditioned, dùng
              _task_logits thay vì global) — t_adapt route bằng z_teacher.
            - L_task (margin loss trên task-routing logits).
            - Task Prompt EMA Update (SwapPrompt-inspired) có confidence
              gate (delta_margin) — task_prompts là MUTABLE state, được
              cập nhật liên tục qua các slide (continual), có
              reset_task_prompts() cho ablation.
            - TCP Confidence Gate tại Phase 7 (tau_task) với fallback về
              naive khi routing không đủ tin cậy.
            - FIM restoration có thể bật/tắt riêng bằng
              tcp_use_fim_restore; cấu hình runtime mặc định hiện tại tắt.
            - KHÔNG có Module F/G/H — một bước, một lần, mọi slide đều adapt.
        SỬA LẦN TRƯỚC ĐÃ SAI: từng biến TCP thành "chỉ đọc kết quả", không
        ảnh hưởng adapt — ĐÂY LÀ LỖI ĐÃ ĐƯỢC NGƯỜI DÙNG CHỈ RA VÀ SỬA LẠI.
        Toàn bộ pipeline adapt của mode="tcp" giờ PHỤC HỒI Y HỆT bản gốc,
        dùng bộ hyperparameter riêng (tiền tố tcp_*) để không lẫn với
        cấu hình đã tune cho naive/task_il.

Modules (naive/task_il):
    A   : WSI Bag-level Augmentation        (Frozen-to-Paraffin, MICCAI 2021)
    B   : DaPC Pseudo-label Correction       (WeiContrast-DaPC, IEEE TETC 2025) — global/task-local, KHÔNG t_adapt
    C'  : L_PETAL + L_class, có Module-F gating
    D   : FIM-based Parameter Restoration    (PETAL, CVPR 2023) — online, mặc định TẮT (đã chọn "no FIM")
    E   : Adaptive LR via OOD score          (NOTE-inspired, NeurIPS 2022)
    F   : Consensus-based Sample Selection   (EATA-inspired, K-view JSD)
    G   : Multi-step adaptation per slide    (mặc định n_steps=3, đã chọn qua thực nghiệm)
    H   : Regularizer bám-nguồn              (mặc định "swag", đã chọn qua thực nghiệm so với "l2_anchor")

Modules (tcp — PHỤC HỒI NGUYÊN BẢN v3):
    A  : WSI Bag-level Augmentation       (Frozen-to-Paraffin, MICCAI 2021)
    B  : DaPC Pseudo-label Correction     (WeiContrast-DaPC, IEEE TETC 2025) — t_adapt-conditioned
    C  : PETAL Loss + L_class + L_task    (PETAL CVPR 2023; L_task margin loss)
    D  : FIM-based Parameter Restoration  (PETAL, CVPR 2023) — mặc định TẮT
    E  : Adaptive LR via OOD score        (NOTE-inspired, NeurIPS 2022)
    P5b: Task Prompt EMA Update           (SwapPrompt-inspired + TPT confidence gate)
    P7 : TCP Confidence Gate              (MergeSlide, arXiv 2025)

Module G — thiết kế chi tiết (CHỈ áp dụng cho naive/task_il, rủi ro collapse
dựa trên zMEMO.pdf, mục "preliminary experiments" của chính tác giả MEMO —
continual adaptation có thể suy biến thành "predicting a constant label
with maximal confidence regardless of the input"):
    * y_corrected (DaPC) tính MỘT LẦN, dùng chung cho cả N bước.
    * Bước đầu tiên (step 0) tái sử dụng ĐÚNG feat_std/coord_std đã dùng
      cho Phase 1b — n_steps=1 tương đương hệt hành vi trước khi có Module G.
    * Các bước sau (resample_per_step=True, mặc định), lấy lại sub-bag MỚI
      mỗi bước — tránh overfit vào 1 subsample cố định với pseudo-label
      có thể sai.
    * FIM tích luỹ qua N bước; Module D (nếu bật) chỉ restore MỘT LẦN sau
      khi kết thúc N bước, không restore giữa chừng.
    * Teacher EMA chỉ update MỘT LẦN sau N bước.
    * step_lr_policy: "same" | "div_n" (eta_eff/N) | "div_sqrt_n". Cấu hình
      đã chọn qua thực nghiệm: n_steps=3, step_lr_policy="div_n".

Usage::
    adapter = MergeSlide_TTA_Adapter_Core(...)
    adapter.reset_task_boundary()          # gọi 1 lần khi bắt đầu mỗi task (naive/task_il)
    pred_local, pred_task, prob_np, debug = adapter.adapt_and_predict(
        features, coords, mode="naive")    # hoặc mode="tcp" / mode="task_il" (+ task_id)
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

    # Module D — FIM Restoration (online). ĐÃ CHỌN qua thực nghiệm: TẮT
    # ("no FIM" cho kết quả tốt nhất khi kết hợp SWAG + n_steps=3).
    use_fim_restore: bool = False
    delta: float           = 0.03

    # Module H — Chọn loại regularizer bám-nguồn (source-anchoring).
    # "swag" (ĐÃ CHỌN qua thực nghiệm, mặc định): -spw * log q(theta_s), q là
    #   Gaussian posterior ước lượng bằng SWAG (mean_sd, var_sd), có
    #   chuẩn hoá theo phương sai từng tham số.
    # "l2_anchor": beta * sum((theta_s - theta_0)^2), theta_0 là checkpoint
    #   đã merge (anchor_sd) — L2 đơn giản, KHÔNG chuẩn hoá theo phương sai,
    #   theo đúng công thức l2_anchor_loss() trong ADAPT-Slide (EATA
    #   simplified). Giữ lại làm phương án ablation — thực nghiệm cho thấy
    #   SWAG tốt hơn, không phải kết luận lý thuyết tuyệt đối.
    # LƯU Ý QUAN TRỌNG: hai regularizer KHÔNG cùng thang đo — spw (mặc định
    # 1e-6) được hiệu chỉnh cho log-probability có chuẩn hoá phương sai,
    # còn l2_anchor_beta (mặc định 1.0, theo đúng default của ADAPT-Slide)
    # là trọng số cho tổng bình phương sai khác THÔ, không chuẩn hoá. TUYỆT
    # ĐỐI không tái dùng giá trị spw đã tune cho l2_anchor_beta — cần sweep riêng.
    # CẢNH BÁO KHÁC: spw=1e-6 được tune cho n_steps=1 (single-step). Với
    # n_steps=3 (mặc định mới), tổng "áp lực adapt" mỗi slide tăng lên —
    # CHƯA có bằng chứng spw=1e-6 vẫn là điểm cân bằng tối ưu, cần re-sweep
    # trước khi đưa vào paper (xem checklist trong 07_MergeSlide-TTA-Unified.md).
    regularizer_type: str = "swag"
    l2_anchor_beta: float = 1.0

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

    # Module G — Multi-step adaptation per slide. ĐÃ CHỌN qua thực nghiệm:
    # n_steps=3, step_lr_policy="div_n". n_steps=1 (nếu muốn quay lại hành
    # vi single-step cũ) vẫn tương thích ngược tuyệt đối (step đầu tiên
    # luôn tái dùng feat_std/coord_std gốc — xem docstring đầu file).
    n_steps: int              = 3
    # "same": eta_step = eta_eff mỗi bước (tổng quãng đường ~N lần, rủi ro cao nhất)
    # "div_n": eta_step = eta_eff / N (tổng quãng đường ~giữ nguyên, an toàn nhất) — ĐÃ CHỌN
    # "div_sqrt_n": eta_step = eta_eff / sqrt(N) (trung dung)
    step_lr_policy: str       = "div_n"
    # True (mặc định): mỗi bước (trừ bước 0) lấy lại sub-bag MỚI, tránh
    # overfit vào 1 subsample cố định. False: dùng chung 1 subsample cho cả
    # N bước (baseline để ablate AC-16, giống "N epoch trên 1 batch cố định").
    resample_per_step: bool   = True

    # ── TCP-legacy (mode="tcp") — PHỤC HỒI Y HỆT TTAConfig_v3 gốc ──────────
    # Bộ field RIÊNG, tiền tố tcp_*, KHÔNG chia sẻ với field naive/task_il ở
    # trên — tránh việc tune 1 bên vô tình ảnh hưởng bên kia. Giá trị mặc
    # định copy chính xác từ TTAConfig_v3 (tta_engine_v3.py gốc, đã xoá).
    tcp_K: int             = 8
    tcp_r_patch: float     = 0.75
    tcp_tau_ap_ind: float  = 0.92
    tcp_tau_ap_ood: float  = 0.70
    tcp_beta_ind: float    = 1.2
    tcp_beta_ood: float    = 1.5
    tcp_spw: float         = 1e-6
    tcp_lam_ce: float      = 1.0
    tcp_gamma_class: float = 1.0
    tcp_tau_c: float       = 0.07
    tcp_gamma_task: float  = 0.5     # trọng số L_task
    tcp_margin_task: float = 0.1     # margin m cho L_task
    tcp_use_fim_restore: bool = False
    tcp_delta: float       = 0.03    # ngưỡng quantile khi FIM restore được bật
    tcp_eta_base: float    = 1e-4
    tcp_tau_ood: float     = 0.5
    tcp_ema_alpha: float   = 0.999
    tcp_alpha_task_prompt: float = 0.999   # EMA momentum cho task prompt update
    tcp_delta_margin: float      = 0.10    # confidence gate: chỉ update prompt khi top1-top2 > delta_margin
    tcp_tau_task: float          = 0.70    # TCP Confidence Gate tại Phase 7

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
    Online, exemplar-free TTA adapter — HAI đường adapt trong 1 class:
    mode="naive"/"task_il" (kiến trúc mới, Module A-H) và mode="tcp"
    (kiến trúc cũ PHỤC HỒI NGUYÊN BẢN — xem docstring đầu file).

    `task_prompts` bắt buộc nếu định dùng mode="tcp" (không cần cho
    naive/task_il). Với mode="tcp", task_prompts là MUTABLE — được cập
    nhật qua EMA tại Phase 5b (giống hệt tta_engine_v3.py gốc). Bản gốc
    (frozen) được giữ lại ở self.task_prompts_source để reset_task_prompts().
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
        task_prompts: Optional[Tensor] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        self.global_w = global_mlp_weight.to(device)
        self.global_b = global_mlp_bias.to(device)
        self.task_class_ranges = task_class_ranges

        self.mean_sd = mean_sd
        self.var_sd  = var_sd

        if task_prompts is not None:
            # task_prompts_source: bản gốc CỐ ĐỊNH, dùng cho reset_task_prompts().
            # task_prompts: working copy MUTABLE, cập nhật qua EMA khi mode="tcp"
            # (Phase 5b, xem _adapt_and_predict_tcp). Với mode="naive"/"task_il",
            # task_prompts KHÔNG được đọc tới ở bất kỳ đâu trong pipeline mới.
            self.task_prompts_source = task_prompts.detach().clone().to(device)
            self.task_prompts        = task_prompts.detach().clone().to(device)
        else:
            self.task_prompts_source = None
            self.task_prompts        = None

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
        task_id: Optional[int] = None,
        mode: str = "naive",
    ) -> tuple[int, int, "np.ndarray", dict]:
        """
        Dispatcher công khai — ĐÂY LÀ CHỖ SỬA LẠI SAU KHI PHÁT HIỆN LỖI: lần
        trước "tcp" chỉ là readout, không ảnh hưởng adapt. Giờ "tcp" gọi
        thẳng pipeline adapt CŨ (_adapt_and_predict_tcp), tách biệt hoàn
        toàn khỏi pipeline mới (_adapt_and_predict_new).

        mode="naive"   : kiến trúc mới, flat/global, task_id bị bỏ qua.
        mode="task_il" : kiến trúc mới, giới hạn theo task_id (bắt buộc).
        mode="tcp"     : kiến trúc CŨ nguyên bản (yêu cầu self.task_prompts
                          đã được cung cấp lúc khởi tạo) — task_id bị bỏ
                          qua (TCP tự route, hoặc nhận task_id như Task-IL
                          nếu truyền vào, giống hệt hành vi gốc).
        """
        if mode not in ("naive", "task_il", "tcp"):
            raise ValueError(f"mode phải là 'naive', 'task_il', hoặc 'tcp', nhận được: {mode!r}")

        if mode == "task_il" and task_id is None:
            raise ValueError("mode='task_il' yêu cầu task_id != None.")

        if mode == "tcp":
            if self.task_prompts is None:
                raise ValueError(
                    "mode='tcp' yêu cầu task_prompts đã được cung cấp lúc khởi tạo adapter "
                    "(tham số task_prompts= trong __init__)."
                )
            return self._adapt_and_predict_tcp(features, coords, task_id=task_id)

        return self._adapt_and_predict_new(
            features, coords, task_id=(task_id if mode == "task_il" else None)
        )

    def _adapt_and_predict_new(
        self,
        features: Tensor,
        coords: Tensor,
        task_id: Optional[int] = None,
    ) -> tuple[int, int, "np.ndarray", dict]:
        """
        Kiến trúc MỚI (Module A-H) — mode="naive" (task_id=None) hoặc
        mode="task_il" (task_id=<int>). KHÔNG L_task, KHÔNG Task Prompt EMA.

        task_id=None: Class-IL naive — flat 13 lớp.
        task_id=<int>: Task-IL — task đã biết trước tại INFERENCE, KHÔNG
            cần routing. TOÀN BỘ pipeline (Module F K-view, Module B DaPC,
            Module C' loss, Phase 7 readout) đều bị giới hạn xuống đúng
            không gian lớp của task_id (qua _restricted_logits).

        Returns
        -------
        pred_local : int   — local class index trong task được suy ra
                              (từ global prediction, xem _global_to_task_local)
        pred_task  : int   — task index (= task_id nếu Task-IL, suy ra nếu không)
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
        p_k = self._k_view_teacher_predictions(features, coords, N, task_id=task_id)   # [K, C]
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

        y_anchor_g  = F.softmax(self._restricted_logits(z_anchor, task_id),      dim=-1)
        y_teacher_g = F.softmax(self._restricted_logits(z_teacher_std, task_id), dim=-1)
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
            g_pred, pred_task, pred_local, g_prob = self._readout(z_inf, task_id=task_id)
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
                "task_id_fixed": task_id if task_id is not None else -1,
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
            logits_s = self._restricted_logits(z_s_f32, task_id)

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

            # ── Phase 4a: Backward (mỗi bước) ────────────────────────────────
            self.optimizer.zero_grad()
            loss_total.backward()

            # Chỉ tính Fisher khi restoration được bật. Với cấu hình no-FIM,
            # bỏ toàn bộ computation này thay vì tính rồi không sử dụng.
            if self.cfg.use_fim_restore:
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

        # ── Phase 5: Teacher EMA (một lần duy nhất, sau khi hết N bước) ─────
        self._ema_update()

        # ── Phase 6: FIM-based Parameter Restoration (Module D, một lần) ────
        if self.cfg.use_fim_restore:
            fisher_flat = torch.cat([v.reshape(-1) for v in fisher_accum.values()])
            fim_threshold = self._find_quantile(fisher_flat, self.cfg.delta)
            self._fim_restore(fisher_accum, fim_threshold)

        # ── Phase 7: Readout (naive hoặc task_il, xem _readout) — dùng
        # feat_std/coord_std GỐC (không phải feat_step cuối cùng), để suy
        # luận không phụ thuộc vào sub-bag ngẫu nhiên nào được lấy ở bước
        # cuối. Routing (nếu TCP) KHÔNG ảnh hưởng ngược lại gradient đã
        # tính ở Phase 2-4 phía trên. ─────────────────────────────────────
        self.teacher.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_inf = self.teacher(feat_std, coord_std, self.ps).float()
        g_pred, pred_task, pred_local, g_prob = self._readout(z_inf, task_id=task_id)
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
            "task_id_fixed": task_id if task_id is not None else -1,
            "prob_global": g_prob.squeeze(0).detach().cpu().numpy(),
        }
        return pred_local, pred_task, prob_np, debug

    # ──────────────────────────────────────────────────────────────────────────
    # TCP-legacy adapt path — PHỤC HỒI NGUYÊN BẢN tta_engine_v3.py (đã xoá).
    # KHÔNG dùng Module F/G/H. Port gần như nguyên văn 1:1, chỉ đổi
    # _task_logits(Z,t) (per_task_w/b riêng) thành _restricted_logits(Z,t)
    # (slice từ global classifier) — hai cách ĐÃ XÁC NHẬN tương đương toán
    # học tuyệt đối (cùng nguồn classifier[:, start:end+1], cùng bias=0),
    # nên hành vi số học giống hệt bản gốc, chỉ gọn hơn về code.
    # ──────────────────────────────────────────────────────────────────────────

    def _adapt_and_predict_tcp(
        self,
        features: Tensor,
        coords: Tensor,
        task_id: Optional[int] = None,
    ) -> tuple[int, int, "np.ndarray", dict]:
        """
        mode="tcp" — Task-Aware Routing Adaptation, y hệt tta_engine_v3.py
        gốc: DaPC theo t_adapt (route bằng z_teacher), L_task (margin loss),
        Task Prompt EMA Update (SwapPrompt-inspired, có confidence gate),
        TCP Confidence Gate tại readout (có fallback về naive). FIM restore
        được điều khiển riêng bởi cfg.tcp_use_fim_restore.

        task_id != None: hành vi Task-IL gốc (t_adapt=task_id cố định,
        KHÔNG có L_task, KHÔNG có prompt update — giữ đúng logic gốc, xem
        `if task_id is None and ... :` ở Phase 5b bên dưới).
        """
        c = self.cfg
        N = features.shape[0]

        # ── Phase 1a: Standard patch subsample ──────────────────────────────
        k = min(c.k_patches_std, N)
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

        ood_score = abs(1.0 - max_anchor_g / (max_teacher_g + c.eps))
        eta_eff = c.tcp_eta_base * torch.sigmoid(
            torch.tensor(ood_score - c.tcp_tau_ood, dtype=torch.float32)
        ).item()

        # ── Phase 1d: Task routing — dùng z_teacher (không phải z_anchor) ───
        if task_id is not None:
            t_adapt = task_id
        else:
            t_adapt = self._tcp_route(z_teacher_std)

        # ── Phase 1e: DaPC pseudo-label (t_adapt-conditioned) ────────────────
        if ood_score < c.tcp_tau_ood:
            tau_ap, beta = c.tcp_tau_ap_ind, c.tcp_beta_ind
        else:
            tau_ap, beta = c.tcp_tau_ap_ood, c.tcp_beta_ood

        y_anchor_adapt   = F.softmax(self._restricted_logits(z_anchor, t_adapt), dim=-1)
        max_anchor_adapt = y_anchor_adapt.max().item()

        use_aug = (max_anchor_g < tau_ap)
        if use_aug:
            y_tilde = self._tcp_aug_teacher_pred(features, coords, t_adapt, N)
        else:
            y_tilde = F.softmax(self._restricted_logits(z_teacher_std, t_adapt), dim=-1)

        if beta * y_tilde.max().item() > max_anchor_adapt:
            y_corrected = y_tilde.detach()
        else:
            y_corrected = 0.5 * (y_tilde + y_anchor_adapt).detach()

        # ── Phase 2: Student forward ─────────────────────────────────────────
        self.student.train()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_s = self.student(feat_std, coord_std, self.ps)
        z_s_f32 = z_s.float()
        logits_s = self._restricted_logits(z_s_f32, t_adapt)

        # ── Phase 3: Loss computation ────────────────────────────────────────
        log_q = self._bayesian_log_q()

        loss_ce = -(
            y_corrected * F.log_softmax(logits_s, dim=-1)
        ).sum(dim=-1).mean()

        loss_petal = -c.tcp_spw * log_q + c.tcp_lam_ce * loss_ce

        loss_class = -(
            y_corrected * F.log_softmax(logits_s / c.tcp_tau_c, dim=-1)
        ).sum(dim=-1).mean()

        # L_task chỉ active khi routing thật sự diễn ra (task_id=None) —
        # giữ đúng logic gốc: Task-IL (task_id cho trước) không có L_task.
        if task_id is None:
            task_scores_s = z_s_f32 @ self.task_prompts.T.detach()  # [1, T]
            sorted_scores = task_scores_s.sort(dim=-1, descending=True).values
            s_top1 = sorted_scores[:, 0]
            s_top2 = sorted_scores[:, 1]
            loss_task = F.relu(s_top2 - s_top1 + c.tcp_margin_task).mean()
            gamma_task_eff = c.tcp_gamma_task
        else:
            loss_task = z_s_f32.new_zeros(())
            gamma_task_eff = 0.0

        loss_total = (
            loss_petal
            + c.tcp_gamma_class * loss_class
            + gamma_task_eff * loss_task
        )

        # ── Phase 4a: Backward ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss_total.backward()

        # Chỉ tính Fisher khi TCP FIM restoration được bật.
        fisher_dict: dict[str, Tensor] = {}
        if c.tcp_use_fim_restore:
            fisher_dict, fisher_flat = self._compute_fim()
            fim_threshold = self._find_quantile(fisher_flat, c.tcp_delta)

        # ── Phase 4c: Scale LR và step ───────────────────────────────────────
        for pg in self.optimizer.param_groups:
            pg['lr'] = eta_eff
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ── Phase 5: Teacher EMA ─────────────────────────────────────────────
        self._ema_update(alpha=c.tcp_ema_alpha)

        # ── Phase 5b: Task Prompt EMA Update (SwapPrompt-inspired) ──────────
        prompt_updated = False
        task_margin_val = 0.0
        if task_id is None:
            with torch.no_grad():
                tcp_scores_ema = F.softmax(
                    z_teacher_std @ self.task_prompts.T, dim=-1
                )
                sorted_tcp = tcp_scores_ema.sort(dim=-1, descending=True).values
                top1_score = sorted_tcp[0, 0].item()
                top2_score = sorted_tcp[0, 1].item()
                task_margin_val = top1_score - top2_score

                if task_margin_val > c.tcp_delta_margin:
                    self._update_task_prompt(t_adapt, z_teacher_std)
                    prompt_updated = True

        # ── Phase 6: Optional FIM-based Parameter Restoration ────────────────
        if c.tcp_use_fim_restore:
            self._fim_restore(fisher_dict, fim_threshold)

        # ── Phase 7: Inference với TCP Confidence Gate ───────────────────────
        self.teacher.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_inf = self.teacher(feat_std, coord_std, self.ps).float()

        tcp_conf = 0.0
        if task_id is not None:
            logits_inf = self._restricted_logits(z_inf, task_id)
            pred_local = int(logits_inf.argmax(-1).item())
            pred_task  = task_id
            prob_inf   = F.softmax(logits_inf.float(), dim=-1)
        else:
            tcp_scores = F.softmax(z_inf.float() @ self.task_prompts.T, dim=-1)
            tcp_conf   = float(tcp_scores.max().item())

            if tcp_conf >= c.tcp_tau_task:
                t_hat      = int(tcp_scores.argmax(-1).item())
                logits_inf = self._restricted_logits(z_inf, t_hat)
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

        prob_np = prob_inf.squeeze(0).detach().cpu().numpy()

        # prob_global: LUÔN tính đủ 13 lớp (rẻ, tái dùng z_inf đã có) — để
        # tương thích schema debug với _adapt_and_predict_new (test_tta_core.py
        # đọc debug["prob_global"] vô điều kiện cho AUC). KHÔNG ảnh hưởng
        # pred_task/pred_local (đã quyết định xong ở trên theo đúng TCP gốc).
        prob_global = F.softmax(self._global_logits(z_inf).float(), dim=-1)

        debug = {
            "ood_score":         ood_score,
            "eta_eff":           eta_eff,
            "tcp_conf":          tcp_conf,
            "t_adapt":           t_adapt,
            "use_aug":           use_aug,
            "loss_petal":        float(loss_petal.detach().item()),
            "loss_class":        float(loss_class.detach().item()),
            "loss_task":         float(loss_task.detach().item()),
            "loss_total":        float(loss_total.detach().item()),
            "gamma_task_eff":    float(gamma_task_eff),
            "loss_task_active":  bool(task_id is None),
            "task_margin":       task_margin_val,
            "prompt_updated":    prompt_updated,
            "use_fim_restore":   bool(c.tcp_use_fim_restore),
            "prob_global":       prob_global.squeeze(0).detach().cpu().numpy(),
        }
        return pred_local, pred_task, prob_np, debug

    def reset_adaptation_state(self) -> None:
        """
        Reset online model + optimizer (dùng cho ablation --reset_per_slide).
        Nếu task_prompts đã được cung cấp (mode="tcp" khả dụng), cũng reset
        task_prompts về bản gốc — đúng hành vi tta_engine_v3.py gốc (nơi
        reset_adaptation_state() luôn gọi kèm reset_task_prompts()).
        """
        merged_sd = {k: v.to(self.device) for k, v in self.anchor_sd.items()}
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if name in merged_sd:
                    param.data.copy_(merged_sd[name])
        self.teacher.load_state_dict(merged_sd, strict=True)
        self.reset_task_prompts()

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

    def _restricted_logits(self, Z: Tensor, task_id: Optional[int]) -> Tensor:
        """
        Task-IL: nếu task_id được cho trước (đã biết task, không cần
        routing), giới hạn logits xuống đúng khoảng lớp của task đó —
        dùng CHUNG classifier toàn cục (global_w/global_b), chỉ slice cột,
        không cần MLP riêng theo task. task_id=None -> trả về global logits
        nguyên vẹn (hành vi cũ, dùng cho naive/tcp mode).
        """
        logits = self._global_logits(Z)
        if task_id is None:
            return logits
        start, end = self.task_class_ranges[task_id]
        return logits[:, start:end + 1]

    def _global_to_task_local(self, g_pred: int) -> tuple[int, int]:
        for t, (start, end) in self.task_class_ranges.items():
            if start <= g_pred <= end:
                return t, g_pred - start
        return 0, 0

    def _tcp_route(self, Z: Tensor) -> int:
        """
        TCP routing dùng self.task_prompts hiện tại (mutable, có thể đã
        được EMA update qua _update_task_prompt). Dùng trong CẢ HAI chỗ:
        (1) Phase 1d của _adapt_and_predict_tcp (routing ảnh hưởng trực
        tiếp DaPC/loss — đây là hành vi TCP thật, phục hồi nguyên bản v3),
        (2) Phase 7 readout của _adapt_and_predict_tcp (đọc kết quả cuối).
        KHÔNG được gọi từ _adapt_and_predict_new (kiến trúc mới không routing).
        """
        with torch.no_grad():
            scores = Z.float() @ self.task_prompts.T
            return int(scores.argmax(-1).item())

    def _tcp_aug_teacher_pred(
        self, features: Tensor, coords: Tensor, t_adapt: int, N: int,
    ) -> Tensor:
        """
        Port nguyên bản _aug_teacher_pred của tta_engine_v3.py — K forward
        pass qua teacher, lấy trung bình softmax TRONG không gian lớp của
        t_adapt (task đã route). Chỉ dùng bởi _adapt_and_predict_tcp.
        """
        c = self.cfg
        n_aug = max(1, int(c.tcp_r_patch * c.k_patches_std))
        n_aug = min(n_aug, N)
        preds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(c.tcp_K):
                aug_idx = torch.randperm(N, device=self.device)[:n_aug]
                z_aug = self.teacher(features[aug_idx], coords[aug_idx], self.ps).float()
                preds.append(F.softmax(self._restricted_logits(z_aug, t_adapt), dim=-1))
        return torch.stack(preds).mean(0)

    def _update_task_prompt(self, t: int, z: Tensor) -> None:
        """
        Phase 5b — Task Prompt EMA Update (SwapPrompt-inspired), port
        nguyên văn từ tta_engine_v3.py. Kéo task_prompts[t] về phía
        z_teacher_std của slide hiện tại. Chỉ gọi khi routing confidence
        gap > tcp_delta_margin (TPT confidence gate, xem _adapt_and_predict_tcp).
        """
        alpha = self.cfg.tcp_alpha_task_prompt
        with torch.no_grad():
            self.task_prompts[t] = (
                alpha * self.task_prompts[t] + (1.0 - alpha) * z.squeeze(0)
            )

    def reset_task_prompts(self) -> None:
        """
        Reset task_prompts (mutable) về task_prompts_source (frozen) — dùng
        cho ablation --reset_prompt_per_task, port nguyên văn từ v3 gốc.
        Chỉ có ý nghĩa khi mode="tcp" (naive/task_il không đọc task_prompts).
        """
        if self.task_prompts is None or self.task_prompts_source is None:
            return
        with torch.no_grad():
            self.task_prompts.copy_(self.task_prompts_source)

    def _readout(self, z_inf: Tensor, task_id: Optional[int] = None) -> tuple[int, int, int, Tensor]:
        """
        Phase 7 readout dùng chung cho cả nhánh active và nhánh bị Module F
        gate của kiến trúc MỚI (_adapt_and_predict_new). Chỉ 2 trường hợp:
        task_id (Task-IL, đã biết trước, không cần routing) hoặc naive (flat
        argmax). TCP có readout RIÊNG trong _adapt_and_predict_tcp (Phase 7
        gốc, với TCP Confidence Gate + fallback) — không dùng hàm này.

        Trả về (g_pred, pred_task, pred_local, g_prob) — g_prob luôn có độ
        dài TOTAL_CLASSES (13); ở nhánh Task-IL, phần ngoài task_id được
        zero-pad (không phải "xác suất thật" toàn cục, chỉ để tương thích
        schema debug/AUC — các slide Task-IL chỉ đến từ đúng task_id nên
        AUC của các lớp khác vẫn không bị ảnh hưởng, chỉ là NaN do không
        có positive example).
        """
        if task_id is not None:
            logits_local = self._restricted_logits(z_inf, task_id)
            prob_local = F.softmax(logits_local.float(), dim=-1)
            pred_local = int(logits_local.argmax(-1).item())
            start, end = self.task_class_ranges[task_id]
            g_pred = start + pred_local
            g_prob = torch.zeros(1, self.global_w.shape[0], device=self.device, dtype=prob_local.dtype)
            g_prob[:, start:end + 1] = prob_local
            return g_pred, task_id, pred_local, g_prob

        logits_inf = self._global_logits(z_inf)
        g_prob = F.softmax(logits_inf.float(), dim=-1)
        g_pred = int(logits_inf.argmax(-1).item())
        pred_task, pred_local = self._global_to_task_local(g_pred)
        return g_pred, pred_task, pred_local, g_prob

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

    def _k_view_teacher_predictions(
        self, features: Tensor, coords: Tensor, N: int, task_id: Optional[int] = None,
    ) -> Tensor:
        """
        Module A + Module F input: K forward pass qua teacher trên K sub-bag
        ngẫu nhiên, trả về [K, C] softmax. C = TOTAL_CLASSES (naive/tcp) hoặc
        C = số lớp của task_id (Task-IL, đã biết trước) — Module F hoạt
        động y hệt trong cả hai trường hợp, chỉ khác chiều rộng không gian.
        """
        n_aug = max(1, int(self.cfg.r_patch * self.cfg.k_patches_std))
        n_aug = min(n_aug, N)
        preds = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for _ in range(self.cfg.K):
                aug_idx = torch.randperm(N, device=self.device)[:n_aug]
                z_aug = self.teacher(features[aug_idx], coords[aug_idx], self.ps).float()
                preds.append(F.softmax(self._restricted_logits(z_aug, task_id), dim=-1))
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

    def _ema_update(self, alpha: Optional[float] = None) -> None:
        alpha = self.cfg.ema_alpha if alpha is None else alpha
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)
