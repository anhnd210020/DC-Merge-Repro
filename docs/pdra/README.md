# PDRA: Phân bổ rank theo nhu cầu dự đoán

**Tóm tắt thí nghiệm dành cho người hướng dẫn nghiên cứu**

> **Tình trạng:** phần triển khai và 19 bài kiểm thử CPU đã hoàn tất. Chưa chạy preflight đầy đủ trên máy chủ, chưa đánh giá bằng GPU và chưa có kết quả thực nghiệm PDRA. Vì vậy, chưa thể kết luận PDRA tốt hơn Uniform.

## 1. Vấn đề

Trong DC-Merge, mỗi task có thể cần lượng rank khác nhau để giữ lại năng lực sau khi hợp nhất. Phân bổ Uniform chia rank bằng nhau cho mọi task. Cách này rõ ràng và giữ nguyên tổng ngân sách, nhưng có thể cấp quá nhiều rank cho một task trong khi task khác cần thêm rank.

PDRA kiểm tra liệu có thể phân bổ cùng tổng rank budget theo nhu cầu rank được dự đoán trước hay không. Đây là giả thuyết cần đo lường; nhu cầu khác nhau và lợi ích của phân bổ không đều chưa được khẳng định là kết quả.

## 2. Giả thuyết

Giả thuyết chính là mean retention của PDRA cao hơn Uniform trong cùng các context và cùng tổng rank budget:

$$
\mathbb{E}_c[Q_{c,\mathrm{PDRA}}-Q_{c,\mathrm{Uniform}}] > 0.
$$

Thí nghiệm ghép cặp hai phương pháp trên từng context. Chỉ sau khi có kết quả GPU mới có thể đánh giá giả thuyết này.

## 3. Phương pháp

### Predictor đã freeze

Predictor là Ridge một biến, đã fit một lần trên 118 nhãn rank-demand thuộc **validation split của thí nghiệm trước**. Đặc trưng được chọn là `energy_rank_90`; không fit lại trên các context của thí nghiệm PDRA và không dùng nhãn final-test để dự đoán hoặc gán rank.

Artifact freeze các bước median imputation và standardization cùng hệ số Ridge (`alpha=1.0`). Với đặc trưng task $x_i$:

$$
z_i = \frac{\operatorname{impute}(x_i;\,167.5)-181.152542}{62.730474},\qquad
\hat D_i = -4.222463 + 3.541401 z_i.
$$

Các hệ số trên là dạng làm tròn để trình bày; [artifact predictor](../../task_demand_prediction/pdra_predictor.json) lưu giá trị đầy đủ và dữ liệu nguồn đã freeze. Mục tiêu rank-demand lúc fit là diện tích hình thang có dấu của độ chênh retention so với baseline rank 16 trên trục rank chuẩn hóa:

$$
D_i=\int_0^1\bigl(R_i(16)-R_i(r(x))\bigr)\,dx,\qquad
x=\frac{r-2}{16-2},\quad r\in\{2,4,8,12,16\}.
$$

### Phương pháp phân bổ và ngân sách

| Số task trong context | Uniform | PDRA theo thứ tự nhu cầu dự đoán giảm dần | Tổng budget |
|---|---|---|---:|
| 4 | `[8, 8, 8, 8]` | `[12, 8, 8, 4]` | 32 |
| 6 | `[8, 8, 8, 8, 8, 8]` | `[12, 12, 8, 8, 4, 4]` | 48 |

Trong mỗi context, task có dự đoán cao hơn nhận vị trí rank lớn hơn trong template PDRA. Nếu dự đoán bằng nhau, code dùng thứ tự task đã đăng ký trong manifest.

| Phương pháp | Cách xếp thứ tự trước khi gán template |
|---|---|
| Uniform | Chia đều rank cho các task. |
| PDRA | Nhu cầu dự đoán giảm dần. |
| Reverse-PDRA | Nhu cầu dự đoán tăng dần, dùng cùng template PDRA. |
| Oracle-Ranked | Nhu cầu oracle của cùng split giảm dần, dùng cùng template PDRA. |

**Ví dụ minh họa, với điểm giả định chứ không phải kết quả:** trong context gồm A, B, C, D, giả sử điểm dự đoán lần lượt là 0.20, 0.90, 0.45, 0.70. Thứ tự PDRA là B, D, C, A, nên rank tương ứng là B=12, D=8, C=8, A=4. Tổng rank vẫn là 32. Uniform gán rank 8 cho cả bốn task.

Oracle-Ranked dùng các đường đáp ứng rank của cùng split để tạo một thứ tự comparator. Oracle-Ranked chỉ là chẩn đoán hậu nghiệm, không tham gia predictor, PDRA, Reverse-PDRA hay việc chọn/tune phương pháp. Đặc biệt, nó **không được bảo đảm là cận trên toán học** cho mọi cách phân bổ rank.

## 4. Thiết lập thí nghiệm

### Task và context

Tập task cố định có tám task: Stanford Cars (`stanford_cars`), DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397 và SVHN. Manifest chứa 45 context mới gồm bốn task và 19 context mới gồm sáu task; các membership đã dùng ở thí nghiệm trước không được đưa vào đánh giá chính.

Các context mới là các tổ hợp của **cùng tám task này**. Đây là đánh giá trên membership mới, không phải tổng quát hóa sang task chưa từng thấy.

### Tách validation và final-test

Predictor được freeze từ validation của thí nghiệm trước. PDRA dùng predictor đó cho cả hai split và không dùng nhãn final-test để xếp hạng task. Runner phải hoàn tất validation với completion marker hợp lệ trước khi chấp nhận final-test.

Oracle-Ranked là ngoại lệ chỉ dành cho chẩn đoán hậu nghiệm: oracle sweep trên validation dùng nhãn validation; oracle sweep trên final-test dùng nhãn final-test chỉ để xếp thứ tự Oracle-Ranked và tính oracle headroom của split đó. Các nhãn này không được dùng để chọn hay tune PDRA.

### Giữ nguyên DC-Merge và số lượt đánh giá

Can thiệp của thí nghiệm chỉ là `task_ranks`. Pipeline DC-Merge hiện tại, xử lý top-k/TIES và các thiết lập merge đã đăng ký được giữ nguyên, bao gồm `alpha=0.8`, `rho=5.0`, `top_percent=0.001`, `smoothing_strategy="linear"`, scale và density của từng task bằng 1.0, cùng code path rank-reduction hiện có.

| Hạng mục | Mỗi split | Validation và final-test |
|---|---:|---:|
| Context | 64 | 128 context-split (64 context trên mỗi split) |
| Allocation: 4 phương pháp × 64 context | 256 | 512 |
| Oracle rank-sweep chẩn đoán | 1.240 | 2.480 |
| **Tổng evaluator units** | **1.496** | **2.992** |

Oracle sweep là phần bổ sung để dựng nhu cầu oracle, không phải một phương pháp phân bổ thứ năm.

## 5. Chỉ số

Gọi $c$ là context, $m$ là phương pháp, $i$ là task trong context $S_c$. Accuracy được biểu diễn theo phần trăm.

**Task retention** so sánh accuracy sau merge với accuracy standalone của cùng task:

$$
R_{c,m,i}=100\times
\frac{\operatorname{Acc}_{c,m,i}(\mathrm{merged})}
{\operatorname{Acc}_{i}(\mathrm{standalone})}.
$$

**Context mean retention** là trung bình đều của retention các task trong context:

$$
Q_{c,m}=\frac{1}{|S_c|}\sum_{i\in S_c}R_{c,m,i}.
$$

**Cải thiện ghép cặp PDRA−Uniform** được tính trong từng context và lấy trung bình trên $N=64$ context:

$$
\Delta_c=Q_{c,\mathrm{PDRA}}-Q_{c,\mathrm{Uniform}},\qquad
\overline{\Delta}=\frac{1}{N}\sum_{c=1}^{N}\Delta_c.
$$

$\Delta$ có đơn vị điểm phần trăm retention. Context là một đơn vị ghép cặp: mỗi phương pháp được so sánh trên cùng membership và cùng split.

**Win rate** là tỷ lệ context có cải thiện dương; hòa không tính là thắng:

$$
\operatorname{WinRate}=\frac{1}{N}\sum_{c=1}^{N}\mathbf{1}[\Delta_c>0].
$$

**Worst-task retention** trong context là retention thấp nhất trong các task của context:

$$
W_{c,m}=\min_{i\in S_c}R_{c,m,i}.
$$

Chỉ số này giúp phát hiện task bị giảm mạnh dù context mean có thể tốt.

**Oracle headroom** là chênh lệch giữa context mean Oracle-Ranked và Uniform:

$$
H_c=Q_{c,\mathrm{Oracle}}-Q_{c,\mathrm{Uniform}}.
$$

$H_c$ là comparator chẩn đoán theo cùng-split oracle, không phải cận trên toán học. Implementation cũng ghi captured gain $(Q_{c,\mathrm{PDRA}}-Q_{c,\mathrm{Uniform}})/H_c$ khi $H_c>0$; nếu headroom bằng hoặc nhỏ hơn 0 thì captured gain được đánh dấu không xác định.

## 6. Tình trạng hiện tại

- Code PDRA và 19 bài kiểm thử CPU trong [`tests/test_pdra.py`](../../tests/test_pdra.py) đã hoàn tất.
- Full server preflight **chưa chạy**.
- GPU evaluation **chưa chạy**.
- Hiện **chưa có kết quả PDRA thực nghiệm**; chưa có cơ sở để kết luận PDRA thắng Uniform.
- Vì mọi context đều ghép từ cùng tám task đã đăng ký, thí nghiệm này không kiểm tra generalization sang task mới.

## 7. Cách tái lập

Các file đã freeze và chi tiết thao tác nằm tại:

- [Predictor validation-only đã freeze](../../task_demand_prediction/pdra_predictor.json)
- [Manifest context và allocation đã freeze](../../task_demand_prediction/pdra_context_manifest.json)
- [Method chi tiết](../pdra_method.md)
- [Runbook: asset, hash, lệnh preflight và chạy](../pdra_runbook.md)
- [Runner implementation](../../task_demand_prediction/pdra.py)

Trình tự tái lập:

1. Giữ nguyên predictor và manifest đã freeze; không refit predictor hoặc thay membership.
2. Dùng runbook để xác minh đủ base model, datasets, adapters, heads và reference files; kiểm tra hash trước và sau khi chuyển lên máy chủ.
3. Chạy full `pdra-preflight` trên máy chủ đã có môi trường CUDA. Lệnh này xác minh dependency, asset, hash và CUDA nhưng không chạy evaluator trên ví dụ dữ liệu.
4. Chạy stage validation trước, kiểm tra output và sealed completion.
5. Chỉ sau khi validation completion được xác minh mới chạy stage final-test. Gate trong runner sẽ từ chối final-test nếu validation chưa hoàn tất hợp lệ.

Runbook giữ nguyên các lệnh đầy đủ và hash asset để tránh lệch giữa hướng dẫn và cấu hình đã freeze.
