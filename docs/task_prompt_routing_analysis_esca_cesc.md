# Phan tich routing cua task_prompts.pt: ESCA vs CESC va cac task khac

Ngay cap nhat: 2026-06-04

Tai lieu nay tong hop van de da quan sat trong MergeSlide CLASS-IL TCP routing, dac biet cap ESCA-CESC, va dua ra cac nguyen tac thiet ke `task_prompts.pt`/task prompt de giam routing nham trong WSI continual learning.

## 1. Vai tro cua task_prompts.pt trong MergeSlide

Trong CLASS-IL TCP, model khong biet truoc slide thuoc task nao. Pipeline baseline trong `test_classIL_task_prompt.py` gom 2 buoc:

1. Route task:

```python
slide_embed = model.backbone(features, coords, ps)
route_logits = slide_embed.float() @ task_prompts.float().T
pred_task_id = argmax(route_logits)
```

2. Predict class bang MLP head cua task duoc route:

```python
mlp = Linear(768, num_classes[pred_task_id])
logits = mlp(slide_embed)
pred_local = argmax(logits)
```

`task_prompts.pt` vi vay khong phai model trainable day du. No la tensor task embedding, shape hien tai la:

```text
[6, 768]
```

Moi row dai dien cho mot task trong thu tu forward:

```text
0 BRCA
1 RCC
2 NSCLC
3 ESCA
4 TGCT
5 CESC
```

Neu `task_prompts.pt` khong tach duoc task identity, buoc routing se sai, va CLASS-IL TCP se dung sai MLP head.

## 2. Hien tuong da quan sat: CESC bi route sang ESCA

Tu debug route da chay tren IND, CESC la task co routing kem nhat. Tom tat tren 10 folds:

```text
BRCA  route acc ~96.81%
RCC   route acc ~99.65%
NSCLC route acc ~93.63%
ESCA  route acc ~88.51%
TGCT  route acc ~85.04%
CESC  route acc ~30.47%
```

Phan lon slide CESC bi route sang ESCA:

```text
CESC -> ESCA: ~64.30%
CESC -> CESC: ~30.47%
```

Vi du fold 0:

```text
CESC route_dist:
  ESCA=89
  CESC=32
  TGCT=3
  BRCA=2
  RCC=1
  NSCLC=1

avg_route_scores:
  ESCA=5.2650
  CESC=4.2895
```

Nghia la voi nhieu slide CESC, dot product `slide_embed @ task_prompts.T` cho ESCA cao hon CESC. Day la dau hieu task prompt dang encode manh histology subtype hon organ/site identity.

## 3. Nguyen nhan chinh: prompt ESCA va CESC trung histology terms

Trong `mergeslide_tta/prompts_zeroshot.py`, ESCA va CESC deu dung hai nhom class histology gan nhu giong nhau:

ESCA:

```text
adenocarcinoma
esophageal adenocarcinoma
adenocarcinoma of the esophagus

squamous cell carcinoma
esophageal squamous cell carcinoma
squamous cell carcinoma of the esophagus
```

CESC:

```text
adenocarcinoma
cervical adenocarcinoma
adenocarcinoma of the cervix uteri

squamous cell carcinoma
cervical squamous cell carcinoma
squamous cell carcinoma of the cervix uteri
```

Hai task chia se cac token histology rat manh:

```text
adenocarcinoma
squamous cell carcinoma
```

Neu task embedding duoc tong hop tu class-aware prompts, embedding task co the bi chi phoi boi subtype morphology chung thay vi site-specific context. Trong WSI H&E, adenocarcinoma/squamous morphology giua cac co quan co the gan nhau hon so voi ky vong tu text prompt. Dieu nay lam CESC gan ESCA trong task-prompt space.

Mot kiem tra prompt da cho thay CESC prompt co cosine cao voi ESCA:

```text
cos(CESC, ESCA) ~0.8099
cos(CESC, NSCLC) ~0.7528
cos(CESC, CESC) =1.0000
```

Day khong tu dong sai, nhung margin nhu vay co the khong du khi slide embedding cua CESC bi anh huong boi morphology squamous/adenocarcinoma.

## 4. Vi sao routing loi co the bi che boi metric legacy

Trong baseline `test_classIL_task_prompt.py`, reported CLASS-IL TCP metric dang giu behavior legacy:

```python
g_pred = seq_dataset.task_to_global_class[task_id].get(pred, -1)
```

Trong do `task_id` la task ground-truth cua loader hien tai, khong phai `pred_task_id`.

Strict routed mapping se la:

```python
routed_g_pred = seq_dataset.task_to_global_class[pred_task_id].get(pred, -1)
```

He qua:

- Metric legacy co the van cao ngay ca khi routing sai.
- Strict metric moi phan anh dung tac dong cua routing trong CLASS-IL that su.
- Khi debug routing/TTA, can log ca hai:
  - `legacy_global_acc`
  - `strict_routed_global_acc`

Voi CESC fold 0, co hien tuong:

```text
legacy_global_acc cao
strict_routed_global_acc thap
route_acc rat thap
```

Dieu nay cho thay class head co the van predict local subtype dung, nhung routing task sai lam global class sai trong CLASS-IL strict.

## 5. Cac task khac cung co nguy co tuong tu

### NSCLC vs ESCA vs CESC

NSCLC cung gom:

```text
lung adenocarcinoma
lung squamous cell carcinoma
```

Day la cung bo histology voi ESCA/CESC, chi khac organ/site. Vi vay NSCLC cung co nguy co nham routing voi ESCA/CESC neu task prompt khong encode ro lung/esophagus/cervix.

### TGCT

TGCT co prompt:

```text
seminoma
mixed germ cell tumor
testicular ...
```

No khac histology hon so voi carcinoma tasks, nen de tach hon ve text. Tuy nhien TGCT route acc da quan sat van thap hon BRCA/RCC/NSCLC, co the do:

- Task prompt it class/site variants hon.
- Slide embedding cua TGCT co distribution gan mot so carcinoma task trong TITAN space.
- Class 1 "mixed germ cell tumor" co noi ham rong.

### BRCA va RCC

BRCA/RCC routing tot hon vi prompt co site va subtype dac thu:

```text
breast invasive ductal/lobular carcinoma
clear cell/papillary/chromophobe renal cell carcinoma
```

RCC dac biet tot vi subtype renal co lexical va morphology identity kha rieng.

## 6. Nguyen tac thiet ke task_prompts.pt tot hon

### 6.1. Task prompt phai encode task identity, khong chi class histology

Neu class prompts dung cho class prediction, task prompts nen uu tien phan biet task/cohort:

```text
TCGA-CESC cervical cancer whole slide histopathology
cervix uteri tumor H&E slide
cervical squamous or glandular epithelial malignancy
```

Khong nen de task prompt chi la trung binh cua:

```text
adenocarcinoma
squamous cell carcinoma
```

Vi cac term nay xuat hien o ESCA, CESC, NSCLC.

### 6.2. Giam generic disease terms trong task-level prompt

Can han che cac term qua chung trong task routing:

```text
adenocarcinoma
squamous cell carcinoma
carcinoma
tumor
cancer
histopathology
H&E
```

Nhung term nay co ich cho class prompt, nhung co the lam task prompt bi collapse giua cac carcinoma tasks.

### 6.3. Them organ/site anchor ro rang

Voi cac task de nham, nen them site-specific anchors:

ESCA:

```text
esophagus
esophageal mucosa
gastroesophageal junction
Barrett-associated esophageal adenocarcinoma
esophageal squamous epithelium
```

CESC:

```text
cervix uteri
cervical transformation zone
endocervical glandular epithelium
ectocervical squamous epithelium
HPV-associated cervical carcinoma
```

NSCLC:

```text
lung parenchyma
bronchial epithelium
alveolar architecture
pulmonary adenocarcinoma
pulmonary squamous carcinoma
```

### 6.4. Task prompt nen co negative/contrastive awareness

Voi cap ESCA-CESC, prompt nen lam ro "not merely squamous/adenocarcinoma" ma la site-specific:

```text
cervical carcinoma from cervix uteri, not esophageal carcinoma
esophageal carcinoma from esophagus, not cervical carcinoma
```

Khong chac TITAN text encoder xu ly negation tot, nen nen dung thang site-positive phrasing hon la dua qua nhieu "not".

### 6.5. Kiem tra cosine matrix cua task_prompts.pt truoc khi eval

Can luon in:

```text
shape
norm moi task
cosine matrix 6x6
top nearest task cho moi task
```

Neu `cos(CESC, ESCA)` qua cao, routing CESC->ESCA la rui ro co the du doan truoc.

### 6.6. Kiem tra route confusion tren validation/debug split

Can log route confusion:

```text
true_task x pred_task
route_acc per task
avg_route_score per task
route_margin distribution
```

Dac biet can xem:

```text
CESC -> ESCA
ESCA -> CESC
NSCLC -> ESCA/CESC
TGCT -> other
```

## 7. Tac dong den TTA

Neu routing goc sai, TTA co 2 nguy co:

1. Adaptation theo pseudo-label sai task.
2. Teacher/student update lam embedding drift ve task sai, khien routing cang sai hon.

Voi CESC, neu anchor route sang ESCA, TTA co the update LayerNorm dua slide embedding gan ESCA/class ESCA hon. Khi do CESC performance giam manh.

Do do voi TTA cho CLASS-IL TCP:

- Khong nen update task prompt goc `task_prompts.pt` truc tiep.
- Nen giu `task_prompts.pt` frozen va luu runtime calibration artifact rieng.
- Nen debug `anchor_route`, `teacher_route`, `route_margin`, `tcp_conf`, `strict_routed_global_acc`.
- Nen dung `episodic` hoac reset theo task de tranh drift tich luy.
- Nen gate/disable adaptation khi route margin thap hoac cap route thuoc ambiguous pair ESCA-CESC.

## 8. Huong fix/thuc nghiem nen uu tien

### A. Runtime routing calibration cho ESCA-CESC

Khong ghi de `task_prompts.pt`. Them artifact rieng:

```text
route_calibration_artifact.json
debug_route_calibration.csv
```

Y tuong:

- Neu top route la ESCA/CESC hoac CESC/ESCA margin thap, dung them evidence tu head confidence.
- Chi ap dung cho ambiguous pair, khong thay doi cac task routing dang tot nhu BRCA/RCC.
- Log base route va calibrated route de so sanh.

### B. Thiet ke lai task prompts va tao task_prompts_v2.pt

Tao ban moi, khong overwrite `task_prompts.pt`:

```text
task_prompts_v2_site_anchor.pt
```

Sau do so sanh:

```text
cosine matrix
route_acc per task
strict_routed_global_acc
legacy_global_acc
```

### C. Task-prompt ensemble

Thay vi 1 vector/task, dung nhieu prompt variants/task va aggregate robust:

```text
task score = max/mean over task prompt variants
```

Voi ESCA/CESC, co the tach variants:

```text
site prompt
histology prompt
TCGA cohort prompt
organ microenvironment prompt
```

### D. Conservative TTA khi routing bat on

Neu `route_margin` thap:

- giam LR
- bo qua adaptation
- chi predict bang fallback global head
- reset episodic sau slide

## 9. Checklist khi danh gia prompt/routing

Truoc khi ket luan TTA tot/xau, can bao cao:

```text
1. baseline legacy CLASS-IL metric
2. strict routed CLASS-IL metric
3. route_acc per task
4. route confusion matrix
5. CESC->ESCA rate
6. ESCA->CESC rate
7. cosine matrix cua task_prompts.pt
8. per-task bACC/ACC
9. TTA route before/after adaptation
10. debug rows cho ambiguous pair
```

Neu chi nhin legacy metric, co the bo sot loi routing that su.

## 10. Ket luan

Van de ESCA-CESC khong phai chi la loi implementation. No la van de semantic alignment giua:

```text
task_prompts.pt
slide embedding TITAN
histology overlap cua adenocarcinoma/squamous carcinoma
CLASS-IL routing objective
```

Voi MergeSlide, `task_prompts.pt` la thanh phan bat buoc cua TCP routing. Tuy nhien task prompt can duoc thiet ke nhu task/site identifier, khong nen chi la trung binh cua class histology prompts. Doi voi cac task co histology chung nhu NSCLC, ESCA, CESC, can them site-specific anchors, route diagnostics, va co the can runtime routing calibration thay vi update truc tiep file `task_prompts.pt` goc.
