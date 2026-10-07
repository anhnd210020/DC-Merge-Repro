# Method — Predicted Demand-Ranked Allocation (PDRA)

## 1. Mục tiêu

Kiểm tra liệu **predicted rank demand** có giúp phân bổ cùng một tổng rank budget tốt hơn **uniform allocation** trong DC-Merge hay không.

Giả thuyết chính:

\[
Q_{\text{PDRA}} > Q_{\text{Uniform}}
\]

với \(Q\) là mean retention của các task trong một merge context.

---

## 2. Phạm vi

- Chỉ thử nghiệm **rank**.
- Giữ nguyên pipeline DC-Merge hiện tại.
- Không thay đổi scale, density, projection, aggregation hoặc các hyperparameter merge khác.
- Dùng predictor đã freeze từ experiment trước:

\[
\hat D_i^{rank}
=
\text{Ridge}(\texttt{energy\_rank\_90}_i)
\]

- Không refit predictor trên allocation-test.

---

## 3. Predicted Demand-Ranked Allocation

Với context:

\[
S=\{T_1,\dots,T_n\}
\]

thực hiện:

1. Tính predicted rank demand \(\hat D_i\) cho từng task.
2. Sort task theo \(\hat D_i\) giảm dần.
3. Gán rank bằng một template cố định theo thứ tự đó.

### 4-task context

Total budget:

\[
B=32
\]

Uniform:

\[
[8,8,8,8]
\]

PDRA:

\[
[12,8,8,4]
\]

Task có predicted demand cao nhất nhận rank 12; thấp nhất nhận rank 4.

### 6-task context

Total budget:

\[
B=48
\]

Uniform:

\[
[8,8,8,8,8,8]
\]

PDRA:

\[
[12,12,8,8,4,4]
\]

Hai task demand cao nhất nhận rank 12; hai task thấp nhất nhận rank 4.

Mọi method phải giữ **cùng total rank budget**.

---

## 4. Baselines

| Method | Allocation |
|---|---|
| Uniform | Chia đều rank |
| **PDRA** | Template theo predicted-demand ranking |
| Reverse-PDRA | Đảo predicted ranking |
| Oracle-Ranked | Cùng template nhưng sort theo oracle demand |

Oracle chỉ dùng làm upper-bound analysis.

---

## 5. Evaluation

Dùng **fresh held-out merge contexts** chưa dùng để thiết kế hoặc tune PDRA.

Retention của task:

\[
R_i
=
100
\frac{Acc_i(\text{merged})}
{Acc_i(\text{standalone})}
\]

Context score:

\[
Q_c
=
\frac{1}{|S_c|}
\sum_i R_{i,c}
\]

Primary metric:

\[
\Delta_c
=
Q_c^{PDRA}-Q_c^{Uniform}
\]

và:

\[
\Delta_{mean}
=
\frac{1}{N}\sum_c\Delta_c
\]

Secondary metrics:

- Win rate: tỷ lệ context có PDRA > Uniform.
- Worst-task retention.
- PDRA vs Reverse-PDRA.
- Oracle headroom.

Oracle headroom:

\[
H
=
Q_{Oracle}-Q_{Uniform}
\]

Captured gain:

\[
\frac{Q_{PDRA}-Q_{Uniform}}
{Q_{Oracle}-Q_{Uniform}}
\]

---

## 6. Success Criterion

Kết quả mong muốn:

\[
\boxed{
Uniform < PDRA \leq Oracle
}
\]

và:

\[
PDRA > Reverse
\]

**GO** nếu:

- Mean retention của PDRA > Uniform.
- Improvement xuất hiện trên phần lớn held-out contexts.
- Không có systematic worst-task collapse.

**Redesign** nếu:

\[
Oracle > Uniform
\quad\text{nhưng}\quad
PDRA \leq Uniform
\]

**Không có allocation headroom** nếu:

\[
Oracle \approx Uniform
\]

---

## 7. Implementation Constraints

- Rank chỉ lấy từ `{2, 4, 8, 12, 16}`.
- Dùng đúng rank-reduction code path của demand experiment hiện tại.
- Không tune allocation template trên test.
- Không dùng oracle demand trong PDRA.
- Log per-task:
  - predicted demand
  - assigned rank
  - merged accuracy
  - standalone accuracy
  - retention

Budget assertion:

```python
assert sum(pdra_ranks) == sum(uniform_ranks)
```

---

## 8. Claim cần kiểm chứng

> **Predicted rank demand có decision utility: dưới cùng một tổng rank budget, demand-ranked allocation cải thiện retention của DC-Merge so với uniform allocation.**
