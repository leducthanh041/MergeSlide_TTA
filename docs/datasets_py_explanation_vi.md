# Giải thích file `mergeslide_tta/datasets.py`

File `mergeslide_tta/datasets.py` là tầng chuẩn bị dữ liệu chính của project MergeSlide_TTA. Nó định nghĩa cách đọc annotation, cách ánh xạ nhãn, cách đọc feature WSI đã được trích xuất sẵn, cách chia train/val/test theo fold, và cách tạo `DataLoader` cho từng task trong bài toán continual learning trên WSI.

Trong pipeline của project, file này không huấn luyện TITAN, không merge checkpoint, và không tính metric. Nhiệm vụ của nó là cung cấp dữ liệu đúng định dạng cho các script ở root như `train_random_sampling.py`, `test_classIL_task_prompt.py`, `test_classIL_task_prompt_other_metrics.py`, và `test_taskIL.py`.

## Vai trò trong toàn project

MergeSlide_TTA xử lý một chuỗi 6 task TCGA:

1. TCGA-BRCA
2. TCGA-RCC
3. TCGA-NSCLC
4. TCGA-ESCA
5. TCGA-TGCT
6. TCGA-CESC

Luồng sử dụng dữ liệu điển hình là:

1. Script train hoặc test import `Sequential_Generic_MIL_Dataset`.
2. Script tạo object `seq_dataset = Sequential_Generic_MIL_Dataset()`.
3. Với mỗi `fold_id` và `task_id`, script gọi:

```python
train_loader, val_loader, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
```

4. `get_data_loaders()` chọn đúng dataset của task đó, đọc file split `splits_<fold>.csv`, tạo 3 split train/val/test.
5. Mỗi sample trong split là một WSI bag gồm:

```python
(features, coords, label)
```

6. `DataLoader` dùng `collate_MIL()` để gom batch và trả về:

```python
[img, coord, label]
```

Trong đó:

- `img`: tensor feature patch-level của một hoặc nhiều WSI.
- `coord`: tensor tọa độ patch tương ứng.
- `label`: nhãn slide/task.

Vì các script hiện tại dùng `batch_size=1`, mỗi batch thường tương ứng với một slide bag, phù hợp với Multiple Instance Learning (MIL) trên WSI.

## Định dạng dữ liệu mà file này kỳ vọng

File này giả định dữ liệu đã được xử lý trước. Nó không đọc ảnh WSI gốc `.svs` trực tiếp. Thay vào đó, nó đọc feature đã được trích xuất bằng vision encoder của TITAN.

### Feature directory

Mỗi task có một thư mục feature dạng:

```text
TCGA-*_processed/features/
├── h5_files/
│   └── <slide_id>.h5
└── pt_files/
    └── <slide_id>.pt
```

Với file `.h5`, code kỳ vọng có ít nhất 2 dataset:

```text
features
coords
```

Trong đó:

- `features`: embedding của các patch trong WSI.
- `coords`: tọa độ các patch.

### Annotation và split

Ba task đầu, gồm BRCA, RCC, NSCLC, dùng `Generic_MIL_Dataset`. Nhóm này đọc annotation CSV/CSV zip, thường có các cột như:

- `slide_id`
- `case_id`
- `oncotree_code`

File split của nhóm này có các cột:

- `train`
- `val`
- `test`

Các giá trị trong split được ghép thêm hậu tố `.svs` khi đối chiếu với `slide_data['slide_id']`.

Ba task sau, gồm ESCA, TGCT, CESC, dùng `Generic_MIL_Dataset2`. Nhóm này đọc trực tiếp file split có cả slide id và label:

- `train`
- `train_label`
- `val`
- `val_label`
- `test`
- `test_label`

## Các import ở đầu file

File import các thư viện chính:

- `torch`, `torch.utils.data.DataLoader`, `Dataset`: tạo dataset và dataloader.
- `h5py`: đọc feature WSI từ file `.h5`.
- `pandas`: đọc annotation CSV và split CSV.
- `numpy`: xử lý mask, index, class id.
- `scipy.stats`: hỗ trợ vote nhãn theo patient.
- `bisect`: dùng trong `ConcatDataset` để map index global sang dataset con.
- `torchvision.datasets`, `torchvision.transforms`: chủ yếu còn lại từ khung continual learning tổng quát.

Một số import đang bị lặp hoặc chưa được dùng trực tiếp, ví dụ `numpy`, `Tuple`, `DataLoader`, `torch.nn.functional as F`. Điều này không làm thay đổi logic chạy hiện tại nhưng cho thấy file có phần code kế thừa từ framework khác.

## `ContinualDataset`

`ContinualDataset` là class nền cho thiết lập continual learning.

Các thuộc tính class:

- `NAME`: tên dataset.
- `SETTING`: kiểu bài toán, ví dụ `class-il`.
- `N_CLASSES_PER_TASK`: số class mỗi task theo thiết kế tổng quát.
- `N_TASKS`: số task.
- `TRANSFORM`: transform dữ liệu, hiện không dùng cho feature WSI.

Trong `__init__`, class tạo các biến trạng thái:

- `self.train_loader`: dataloader train hiện tại.
- `self.test_loaders`: danh sách test loader đã được tạo.
- `self.i`: con trỏ class/task trong một số helper continual learning.

Các method như `get_data_loaders()`, `get_backbone()`, `get_transform()`, `get_loss()`, `get_scheduler()` được đánh dấu abstract nhưng phần lớn không được implement đầy đủ ở class con. Với project này, method quan trọng thật sự là `get_data_loaders(FOLD, task_id)` trong `Sequential_Generic_MIL_Dataset`.

## `store_masked_loaders()`

Hàm này là helper tổng quát để chia dataset thành các task theo nhãn class.

Nó nhận:

- `train_dataset`
- `test_dataset`
- `setting`

Logic:

1. Tạo mask để giữ lại các sample có label nằm trong khoảng:

```python
[setting.i, setting.i + setting.N_CLASSES_PER_TASK)
```

2. Lọc `train_dataset.data`, `test_dataset.data`, `train_dataset.targets`, `test_dataset.targets`.
3. Tạo `DataLoader` cho train và test.
4. Thêm test loader vào `setting.test_loaders`.
5. Tăng `setting.i`.

Trong pipeline MergeSlide hiện tại, hàm này không phải đường chính cho dữ liệu WSI. Dữ liệu WSI được lấy qua split CSV sẵn có trong `Sequential_Generic_MIL_Dataset.get_data_loaders()`.

## `get_previous_train_loader()`

Hàm này tạo train loader cho task ngay trước task hiện tại, dựa trên `setting.i` và `N_CLASSES_PER_TASK`.

Nó cũng là helper continual learning tổng quát, không phải logic chính đang được các script root dùng cho WSI.

## `collate_MIL()`

Đây là hàm collate quan trọng cho WSI MIL.

Code:

```python
def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coord = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, coord, label]
```

Mỗi item trong batch có dạng:

```python
(features, coords, label)
```

Hàm này:

- nối các tensor `features` theo chiều đầu tiên;
- nối các tensor `coords` theo chiều đầu tiên;
- gom label thành `LongTensor`;
- trả về list `[img, coord, label]`.

Với `batch_size=1`, phép `cat` gần như giữ nguyên bag của một WSI. Nếu tăng batch size, hàm này sẽ nối patch của nhiều slide vào cùng một tensor, nên cần chắc chắn model phía sau hiểu được ranh giới giữa các slide. Các script hiện tại tránh vấn đề đó bằng cách dùng `batch_size=1`.

## `generate_split()`

Hàm này sinh các split train/val/test theo class.

Đầu vào chính:

- `cls_ids`: danh sách index sample theo từng class.
- `val_num`: số sample validation mỗi class.
- `test_num`: số sample test mỗi class.
- `samples`: tổng số sample.
- `n_splits`: số split/fold cần sinh.
- `seed`: random seed.
- `label_frac`: tỷ lệ label train được giữ lại.
- `custom_test_ids`: test ids cố định nếu có.

Mỗi lần yield, hàm trả về:

```python
(sampled_train_ids, all_val_ids, all_test_ids)
```

Trong project hiện tại, split 10-fold đã có sẵn trong thư mục annotation nên hàm này chủ yếu là logic hỗ trợ/legacy để tự sinh split nếu cần.

## `nth()`

Helper nhỏ để lấy phần tử thứ `n` từ một iterator.

Nó được dùng trong `set_splits()` để nhảy tới split mong muốn trong generator do `generate_split()` tạo ra.

## `save_splits()`

Hàm này lưu các split dataset ra CSV.

Nếu `boolean_style=False`, file output có các cột như:

```text
train,val,test
```

Nếu `boolean_style=True`, file output dùng dạng one-hot boolean cho train/val/test.

Trong pipeline hiện tại, các script chủ yếu đọc split có sẵn thay vì gọi hàm này để tạo split mới.

## `Generic_WSI_Classification_Dataset`

Đây là class nền để đọc metadata slide-level từ annotation CSV.

Nó kế thừa `torch.utils.data.Dataset` và xử lý:

- đọc file annotation;
- lọc sample;
- bỏ qua class không dùng;
- map label dạng chuỗi sang số nguyên;
- tạo thống kê theo slide và patient;
- tạo hoặc đọc split train/val/test.

### Tham số khởi tạo

Các tham số chính:

- `csv_path`: đường dẫn annotation CSV/CSV zip.
- `shuffle`: có shuffle metadata hay không.
- `seed`: seed dùng khi shuffle hoặc sinh split.
- `print_info`: cờ in thông tin, hiện không tự gọi `summarize()`.
- `label_dict`: map label gốc sang label số.
- `filter_dict`: lọc DataFrame theo giá trị cột.
- `ignore`: danh sách label cần loại bỏ.
- `patient_strat`: nếu `True`, split theo patient thay vì slide.
- `label_col`: cột chứa nhãn; mặc định là `oncotree_code`.
- `patient_voting`: cách gán nhãn patient khi một patient có nhiều slide.

### `df_prep()`

Hàm này chuẩn hóa label.

Logic:

1. Nếu `label_col` không phải `label`, tạo cột `label` từ cột đó.
2. Loại các dòng có label nằm trong `ignore`.
3. Reset index.
4. Đổi label gốc thành số nguyên theo `label_dict`.

Ví dụ với BRCA:

```python
label_dict={'IDC': 0, 'ILC': 1}
```

Các label không thuộc bài toán như `MDLC`, `PD`, `ACBC`, ... bị loại qua `ignore`.

### `filter_df()`

Lọc DataFrame theo `filter_dict`. Nếu `filter_dict` rỗng, giữ nguyên dữ liệu.

### `patient_data_prep()`

Tạo dữ liệu mức patient từ dữ liệu mức slide.

Nó gom các slide theo `case_id`, sau đó chọn label patient bằng:

- `max`: lấy label lớn nhất;
- `maj`: lấy majority vote bằng `stats.mode`.

Thông tin này cần cho trường hợp `patient_strat=True`.

### `cls_ids_prep()`

Tạo danh sách index theo từng class:

- `self.patient_cls_ids`: index patient theo class.
- `self.slide_cls_ids`: index slide theo class.

Các danh sách này dùng khi sinh split tự động.

### `create_splits()` và `set_splits()`

`create_splits()` cấu hình generator split bằng `generate_split()`.

`set_splits()` lấy một split từ generator đó và gán:

```python
self.train_ids
self.val_ids
self.test_ids
```

Nếu `patient_strat=True`, class sẽ map patient id về toàn bộ slide thuộc patient đó.

Trong project hiện tại, đường chính là đọc split CSV sẵn có bằng `return_splits(from_id=False, csv_path=...)`.

### `get_split_from_df()`

Hàm này lấy một cột split từ CSV, ví dụ `train`, `val`, hoặc `test`, rồi tạo dataset con `Generic_Split`.

Điểm quan trọng:

```python
mask = self.slide_data['slide_id'].isin([i + '.svs' for i in split.tolist()])
```

Nghĩa là split CSV của nhóm BRCA/RCC/NSCLC được kỳ vọng chứa slide id chưa có hậu tố `.svs`, còn annotation `slide_data['slide_id']` có hậu tố `.svs`.

### `get_merged_split_from_df()`

Hàm này ghép nhiều cột split lại thành một split duy nhất.

Hiện nó không phải đường chính trong các script train/eval. Cần chú ý là hàm này không thêm `.svs` như `get_split_from_df()`, nên nếu dùng lại cần kiểm tra định dạng slide id.

### `return_splits()`

Đây là method quan trọng để trả về:

```python
(train_split, val_split, test_split)
```

Nếu `from_id=True`, nó dùng `self.train_ids`, `self.val_ids`, `self.test_ids` đã được sinh trước đó.

Nếu `from_id=False`, nó đọc split từ `csv_path`, sau đó gọi:

```python
self.get_split_from_df(all_splits, 'train')
self.get_split_from_df(all_splits, 'val')
self.get_split_from_df(all_splits, 'test')
```

Trong MergeSlide_TTA hiện tại, `Sequential_Generic_MIL_Dataset.get_data_loaders()` gọi `return_splits(from_id=False, csv_path=...)`.

### `test_split_gen()` và `save_split()`

Hai method này phục vụ kiểm tra/lưu split:

- `test_split_gen()` kiểm tra train/val/test không giao nhau và có thể trả về bảng thống kê class.
- `save_split()` lưu `train_ids`, `val_ids`, `test_ids` ra CSV.

## `Generic_MIL_Dataset`

`Generic_MIL_Dataset` kế thừa `Generic_WSI_Classification_Dataset`.

Nó được dùng cho 3 task đầu:

1. BRCA
2. RCC
3. NSCLC

Khác với class cha chỉ xử lý metadata, class này implement `__getitem__()` để đọc feature thật từ disk.

### `__getitem__()`

Với một index, class lấy:

```python
slide_id = self.slide_data['slide_id'][idx]
label = self.slide_data['label'][idx]
```

Sau đó tạo đường dẫn:

```python
<data_dir>/h5_files/<slide_id_without_svs>.h5
```

Nó mở file `.h5` và đọc:

```python
features = hdf5_file['features'][:]
coords = hdf5_file['coords'][:]
```

Nếu đọc `features` trong `.h5` lỗi, code fallback sang:

```python
<data_dir>/pt_files/<slide_id_without_svs>.pt
```

Sau đó convert:

```python
features = torch.from_numpy(features)
coords = torch.from_numpy(coords)
```

Và trả về:

```python
(features, coords, label)
```

Lưu ý: fallback sang `.pt` chỉ xảy ra sau khi file `.h5` đã mở được nhưng đọc `features` bị lỗi. Nếu chính file `.h5` không tồn tại, code sẽ lỗi ngay tại `h5py.File(...)`.

## `Generic_MIL_Dataset2`

`Generic_MIL_Dataset2` được dùng cho 3 task sau:

1. ESCA
2. TGCT
3. CESC

Nhóm này có định dạng split khác nhóm đầu. Thay vì đọc annotation riêng rồi map split theo `slide_id`, class này đọc trực tiếp file split chứa cả id và label.

### `return_splits()`

Nó đọc CSV:

```python
slide_data = pd.read_csv(csv_path, index_col=0)
```

Sau đó lấy:

- `train` và `train_label`
- `val` và `val_label`
- `test` và `test_label`

Label được map qua:

```python
self.label_dict[int(l)]
```

Cuối cùng trả về 3 object:

```python
Generic_MIL_Dataset2_Split(...)
```

## `Generic_MIL_Dataset2_Split`

Đây là split dataset cho nhóm ESCA/TGCT/CESC.

Nó giữ:

- `self.data_dir`: thư mục feature.
- `self.data`: danh sách slide id.
- `self.label`: danh sách label tương ứng.

### `__getitem__()`

Với một index, class tạo đường dẫn:

```python
<data_dir>/h5_files/<slide_id>.h5
```

Sau đó đọc:

```python
features
coords
```

và trả về:

```python
(features, coords, label)
```

Khác với `Generic_MIL_Dataset`, class này không bỏ `.svs` bằng `split('.svs')[0]`. Vì vậy slide id trong split CSV của nhóm này phải khớp trực tiếp với tên file `.h5`.

## `Generic_Split`

`Generic_Split` là dataset con cho các split của `Generic_MIL_Dataset`.

Nó kế thừa `Generic_MIL_Dataset`, nhưng không gọi lại `__init__()` của class cha. Thay vào đó, nó nhận trực tiếp:

- `slide_data`: DataFrame đã được lọc cho train/val/test.
- `data_dir`: thư mục feature.
- `num_classes`: số class.

Vì kế thừa `Generic_MIL_Dataset`, nó dùng lại `__getitem__()` để đọc feature `.h5` hoặc `.pt`.

Nói ngắn gọn: `Generic_WSI_Classification_Dataset.return_splits()` tạo ra `Generic_Split`, và `Generic_Split` là object thật sự được đưa vào `DataLoader` cho BRCA/RCC/NSCLC.

## `ConcatDataset`

`ConcatDataset` nối nhiều dataset lại thành một dataset lớn.

Nó tương tự ý tưởng của `torch.utils.data.ConcatDataset`:

- tính cumulative size của từng dataset con;
- với một index global, dùng `bisect` để tìm dataset con tương ứng;
- trả về sample từ dataset con đó.

Trong file này, `ConcatDataset` được dùng bởi:

```python
Sequential_Generic_MIL_Dataset.get_joint_data_loaders()
```

Mục đích là tạo train/val loader chứa dữ liệu của tất cả task, phù hợp nếu muốn chạy joint training hoặc baseline không theo từng task.

## `Sequential_Generic_MIL_Dataset`

Đây là class quan trọng nhất của file đối với project MergeSlide_TTA.

Nó kế thừa `ContinualDataset` và định nghĩa chuỗi task WSI.

Các thuộc tính:

```python
NAME = 'seq-wsi'
SETTING = 'class-il'
N_CLASSES_PER_TASK = 2
N_TASKS = 6
TRANSFORM = None
```

`SETTING = 'class-il'` cho biết thiết lập chính là class-incremental learning.

### Danh sách dataset

Class này khai báo sẵn 6 dataset:

```python
datasets = [
    Generic_MIL_Dataset(... TCGA-BRCA ...),
    Generic_MIL_Dataset(... TCGA-RCC ...),
    Generic_MIL_Dataset(... TCGA-NSCLC ...),
    Generic_MIL_Dataset2(... TCGA-ESCA ...),
    Generic_MIL_Dataset2(... TCGA-TGCT ...),
    Generic_MIL_Dataset2(... TCGA-CESC ...),
]
```

Ý nghĩa từng task:

| `task_id` | Dataset | Class trong code |
|---:|---|---|
| 0 | TCGA-BRCA | `Generic_MIL_Dataset` |
| 1 | TCGA-RCC | `Generic_MIL_Dataset` |
| 2 | TCGA-NSCLC | `Generic_MIL_Dataset` |
| 3 | TCGA-ESCA | `Generic_MIL_Dataset2` |
| 4 | TCGA-TGCT | `Generic_MIL_Dataset2` |
| 5 | TCGA-CESC | `Generic_MIL_Dataset2` |

Ba task đầu dùng annotation CSV riêng và split chỉ chứa slide id. Ba task sau dùng split CSV có sẵn label.

### `split_dirs`

`split_dirs` là danh sách thư mục chứa file split cho từng task:

```python
split_dirs = [
    '/home/bui/datasets/wsi_dataset_annotation/tcga_brca',
    '/home/bui/datasets/wsi_dataset_annotation/tcga_rcc',
    '/home/bui/datasets/wsi_dataset_annotation/tcga_nsclc',
    '/home/bui/datasets/wsi_dataset_annotation/tcga_esca',
    '/home/bui/datasets/wsi_dataset_annotation/tcga_tgct',
    '/home/bui/datasets/wsi_dataset_annotation/tcga_cesc',
]
```

Với `FOLD = 0`, file được đọc sẽ là:

```python
<split_dirs[task_id]>/splits_0.csv
```

Project README cũng nhấn mạnh cần sửa các path này cho đúng môi trường chạy.

### `get_data_loaders(FOLD, task_id)`

Đây là API chính mà các script train/test dùng.

Logic:

1. Chọn dataset theo `task_id`:

```python
dataset = self.datasets[task_id]
```

2. Đọc split:

```python
train_dataset, val_dataset, test_dataset = dataset.return_splits(
    from_id=False,
    csv_path='{}/splits_{}.csv'.format(self.split_dirs[task_id], FOLD)
)
```

3. Tạo dataloader:

```python
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_MIL)
```

4. Lưu trạng thái vào object:

```python
self.test_loaders.append(test_loader)
self.train_loader = train_loader
self.val_loader = val_loader
```

5. Trả về:

```python
(train_loader, val_loader, test_loader)
```

Train và validation được shuffle, test không shuffle.

### `get_joint_data_loaders(FOLD)`

Hàm này đọc tất cả 6 task trong cùng một fold, nối train split của toàn bộ task thành một dataset chung, và nối val split thành một dataset chung.

Luồng:

1. Lặp qua `n` từ 0 đến `N_TASKS - 1`.
2. Đọc split của từng task.
3. Đưa train split vào `train_datasets`.
4. Đưa val split vào `val_datasets`.
5. Tạo test loader từng task và thêm vào `self.test_loaders`.
6. Nối train/val bằng `ConcatDataset`.
7. Tạo joint train loader và joint val loader.

Hàm trả về:

```python
(train_loader, val_loader, test_loader)
```

Lưu ý: biến `test_loader` được trả về ở cuối là test loader của task cuối cùng trong vòng lặp. Nếu cần toàn bộ test loader, nên dùng `self.test_loaders`.

## Block `if __name__ == '__main__'`

Phần cuối file cho phép chạy thử trực tiếp:

```python
if __name__ == '__main__':
    seq_dataset = Sequential_Generic_MIL_Dataset()
    fold = 0
    task_id = 0
    trains, vals, tests = seq_dataset.get_data_loaders(fold, task_id)
```

Đây là kiểm tra tối thiểu để tạo dataloader cho fold 0, task 0. Tuy nhiên, muốn chạy được block này thì các path hardcode trong file phải tồn tại.

## File này được gọi ở đâu?

Các script chính import và dùng `Sequential_Generic_MIL_Dataset`.

### `train_random_sampling.py`

Script finetune theo task gọi:

```python
seq_dataset = Sequential_Generic_MIL_Dataset()
train_loader, val_loader, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
```

Mỗi task lấy dataloader riêng, sau đó model TITAN được finetune trên task đó.

Lưu ý quan trọng: trong trạng thái code hiện tại, script có `num_tasks = 6` nhưng vòng lặp train đang là:

```python
for task_id in range(3):
```

Nghĩa là đoạn train hiện tại chỉ chạy 3 task đầu nếu không sửa lại.

### `test_classIL_task_prompt.py`

Script CLASS-IL final evaluation tạo `seq_dataset`, rồi với từng task:

```python
_, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
```

Nó chỉ dùng test loader để đánh giá merged model trên tất cả task.

### `test_classIL_task_prompt_other_metrics.py`

Script tính forgetting/BWT/FWT cũng gọi:

```python
_, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
```

Khác biệt là nó đánh giá theo từng trạng thái sequence `seq_task`, tương ứng checkpoint merge sau khi học đến task đó.

### `test_taskIL.py`

Script TASK-IL cũng dùng:

```python
_, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
```

Sau đó load head/task-specific MLP tương ứng để đánh giá trong setting TASK-IL.

## Những điểm cần chú ý khi dùng hoặc sửa file này

### 1. Path dữ liệu đang hardcode

Các path hiện tại trỏ tới:

```text
/home/bui/datasets/...
```

Nếu chạy trên máy khác, cần sửa:

- `datasets` trong `Sequential_Generic_MIL_Dataset`;
- `split_dirs` trong `Sequential_Generic_MIL_Dataset`.

Nếu path không tồn tại, việc import hoặc khởi tạo dataset có thể lỗi trước khi vào training.

### 2. Dataset object được tạo ngay ở cấp class

`Sequential_Generic_MIL_Dataset.datasets` là class variable, và nó tạo các object dataset ngay khi class được định nghĩa.

Với `Generic_MIL_Dataset`, constructor đọc annotation CSV ngay:

```python
slide_data = pd.read_csv(csv_path)
```

Do đó, chỉ cần import module trong môi trường thiếu annotation path là có thể lỗi. Đây là lý do cần kiểm tra path trước khi chạy các script root.

### 3. `N_CLASSES_PER_TASK = 2` không phản ánh đúng mọi task

RCC có 3 class:

```python
label_dict={'CCRCC': 0, 'PRCC': 1, 'CHRCC': 2}
```

Trong các script train/test, số class thật được khai báo riêng:

```python
num_classes = [2, 3, 2, 2, 2, 2]
```

Vì vậy `N_CLASSES_PER_TASK = 2` trong `Sequential_Generic_MIL_Dataset` không nên được hiểu là số class thật của mọi task. Nó chủ yếu còn lại từ khung continual learning tổng quát.

### 4. `load_from_h5()` chưa thật sự điều khiển logic đọc file

Một số class có method:

```python
load_from_h5(self, toggle)
```

Nhưng `__getitem__()` vẫn ưu tiên mở `.h5`. Biến `self.use_h5` không được kiểm tra rõ ràng để chuyển hẳn sang `.pt`.

### 5. `collate_MIL()` giả định có thể nối patch theo batch

Với `batch_size=1`, giả định này ổn. Nếu tăng batch size, feature patch của nhiều slide sẽ bị nối chung, có thể làm mất ranh giới slide nếu model phía sau không tự xử lý.

### 6. Hai nhóm dataset có quy ước slide id khác nhau

Với `Generic_MIL_Dataset`, code bỏ `.svs` khi tìm file `.h5`:

```python
slide_id.split('.svs')[0]
```

Với `Generic_MIL_Dataset2_Split`, code dùng trực tiếp `slide_id`:

```python
'{}.h5'.format(slide_id)
```

Do đó split CSV và tên file feature phải khớp đúng theo từng nhóm task.

### 7. `get_joint_data_loaders()` trả về test loader cuối cùng

Hàm này append test loader của từng task vào `self.test_loaders`, nhưng giá trị `test_loader` trả về trực tiếp là của task cuối cùng. Nếu cần test loader của tất cả task, dùng `seq_dataset.test_loaders`.

### 8. Một số helper không nằm trên đường chạy chính

Các hàm như `store_masked_loaders()`, `get_previous_train_loader()`, `generate_split()`, `save_splits()` hữu ích cho setup continual learning tổng quát hoặc tạo split, nhưng pipeline hiện tại chủ yếu dùng split CSV đã có.

## Tóm tắt ngắn

`mergeslide_tta/datasets.py` là file chịu trách nhiệm biến dữ liệu TCGA WSI đã xử lý trước thành `DataLoader` PyTorch cho từng task continual learning. Nó đọc annotation/split, map label, mở feature `.h5` hoặc `.pt`, trả về `(features, coords, label)`, rồi cung cấp train/val/test loader cho các script finetune và evaluation.

Class quan trọng nhất là `Sequential_Generic_MIL_Dataset`. Nếu cần thay đổi dữ liệu, fold, task, hoặc đường dẫn feature/split, đây là nơi cần kiểm tra đầu tiên.

---
# So sánh giữa Generic_MIL_Dataset và Generic_MIL_Dataset2

Generic_MIL_Dataset và Generic_MIL_Dataset2 đều phục vụ đọc WSI feature dạng MIL, nhưng dùng cho hai nhóm dữ liệu
  có format khác nhau.

  | Điểm khác | Generic_MIL_Dataset | Generic_MIL_Dataset2 |
  |---|---|---|
  | Dùng cho task | BRCA, RCC, NSCLC | ESCA, TGCT, CESC |
  | Kế thừa | Kế thừa Generic_WSI_Classification_Dataset | Không kế thừa Dataset chuẩn, chỉ là wrapper tạo split |
  | Annotation | Đọc annotation CSV riêng lúc khởi tạo | Không đọc annotation lúc khởi tạo |
  | Split CSV | Split chỉ có train, val, test | Split có cả train_label, val_label, test_label |
  | Label | Map label từ chuỗi như IDC, ILC, CCRCC sang số | Map label số từ CSV qua label_dict |
  | Slide id | Annotation có slide_id dạng .svs; khi đọc file thì bỏ .svs | Dùng trực tiếp slide id trong split CSV
  |
  | File feature | <data_dir>/h5_files/<slide_id_without_svs>.h5 | <data_dir>/h5_files/<slide_id>.h5 |
  | Split object trả về | Generic_Split | Generic_MIL_Dataset2_Split |

  Cụ thể, Generic_MIL_Dataset ở mergeslide_tta/datasets.py:416 phù hợp với dữ liệu có annotation đầy đủ. Nó đọc
  CSV, lọc class không dùng, map label, tạo slide_data, rồi khi split thì đối chiếu slide id trong split với
  annotation.

  Còn Generic_MIL_Dataset2 ở mergeslide_tta/datasets.py:447 đơn giản hơn: nó tin rằng file split CSV đã chứa luôn
  slide id và label. Vì vậy return_splits() chỉ đọc các cột train, train_label, val, val_label, test, test_label,
  rồi tạo Generic_MIL_Dataset2_Split.

  Điểm dễ lỗi nhất là quy ước tên file:

  # Generic_MIL_Dataset
  slide_id.split('.svs')[0]

  # Generic_MIL_Dataset2_Split
  slide_id

  Nghĩa là với Generic_MIL_Dataset, annotation có thể là ABC.svs nhưng file feature là ABC.h5. Với
  Generic_MIL_Dataset2, nếu split ghi ABC, code sẽ tìm đúng ABC.h5; nếu split ghi ABC.svs, code sẽ tìm ABC.svs.h5.