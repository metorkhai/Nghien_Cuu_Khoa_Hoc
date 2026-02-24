# SoftLogic ViDeBERTa: Phân tích Cảm xúc Ngôn ngữ GenZ Tiếng Việt bằng Suy luận Logic Mờ

Một kiến trúc phân tích cảm xúc tiên tiến dành cho nghiên cứu, đi sâu hơn việc tinh chỉnh (fine-tuning) thông thường bằng cách tích hợp suy luận cơ chế (mechanistic reasoning), nhận diện xung đột ngữ nghĩa và thành kiến quy nạp (inductive bias) — tất cả đều có thể đạo hàm từ đầu đến cuối (end-to-end differentiable).

## Cải tiến Cốt lõi

Các mô hình cảm xúc truyền thống thường thất bại với ngôn ngữ phi chính thống của GenZ vì chúng:
- Chỉ tập trung vào các nhãn chiếm ưu thế (ví dụ: "Enjoyment").
- Bỏ lỡ các mâu thuẫn (bề mặt tích cực + ý định thực tế tiêu cực).
- Dựa vào các từ khóa bề mặt thay vì hiểu ý nghĩa sâu xa.

**SoftLogic ViBERT** (xây dựng trên nền tảng ViDeBERTa) giải quyết các vấn đề này thông qua:

1. **Token-level TF-IDF Gating**: Sử dụng trọng số thống kê để loại bỏ nhiễu từ các token ít quan trọng.
2. **Multi-view Representations**: Trích xuất đặc trưng từ nhiều góc nhìn: Ngữ nghĩa (Semantic), Từ vựng (Lexical), và Ngữ dụng (Pragmatic).
3. **Differentiable Fuzzy Logic**: Sử dụng các vị từ mềm (soft predicates) + quy tắc mờ (fuzzy rules) với khả năng học được độ nhạy của phép toán AND.
4. **Conflict Detection**: Mô hình hóa rõ ràng sự mỉa mai (sarcasm) và các mâu thuẫn ngữ nghĩa.

## Kiến trúc Hệ thống

```
(ngữ cảnh, bình luận)
       ↓
┌─────────────────────────────────┐
│   Token-level TF-IDF Gating     │  ← Tính toán trọng số thống kê cho mỗi token
│   s_i = tfidf(token_id_i)       │
│   E'_i = s_i * E_i              │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│    ViDeBERTa Encoder            │  ← Sử dụng Fsoft-AIC/videberta-base
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Các lớp chiếu Multi-view      │
│   ┌───────┐ ┌───────┐ ┌───────┐ │
│   │z_sem  │ │z_lex  │ │z_prag │ │
│   │(CLS)  │ │(CNN)  │ │(MLP)  │ │
│   └───────┘ └───────┘ └───────┘ │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   Mô đun Suy luận Logic Mờ      │
│                                 │
│   Vị từ (Predicates):           │
│   P_pos_sem, P_neg_sem,         │
│   P_pos_lex, P_neg_lex,         │
│   P_high_int ∈ (0,1)            │
│                                 │
│   Quy tắc (Rules):              │
│   r1 = AND(P_pos_lex, P_neg_sem)│  ← Mỉa mai
│   r2 = AND(P_neg_lex, P_neg_sem)│  ← Tiêu cực mạnh
│   r3 = AND(P_pos_sem, NOT(P_high_int))
│   r4 = AND(P_high_int, P_neg_sem)
│   r5 = |P_pos_lex - P_pos_sem|  │  ← Sự không nhất quán
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│   MLP Suy luận Cảm xúc          │
│   [z_sem, z_lex, z_prag, r1-r5] │
│            ↓                    │
│   Phân loại Đa nhãn (Multi-label)│
└─────────────────────────────────┘
```

## Cài đặt

```bash
# Clone repository
git clone <repository>
cd softlogic_vibert

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Yêu cầu hệ thống
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- (Tùy chọn) CUDA để tăng tốc bằng GPU

## Hướng dẫn nhanh

### Huấn luyện (Training)

```bash
# Huấn luyện với dữ liệu JSON/JSONL cục bộ
python -m softlogic_vibert.train \
    --data-path output_data.json \
    --output-dir softlogic_outputs \
    --epochs 10 \
    --batch-size 16

# Huấn luyện với tập dữ liệu từ HuggingFace
python -m softlogic_vibert.train \
    --hf-dataset tridm/UIT-VSMEC \
    --hf-split train \
    --output-dir softlogic_outputs
```

### Dự đoán (Inference)

```bash
# Dự đoán một câu kèm phân tích chi tiết
python -m softlogic_vibert.inference \
    --ckpt softlogic_outputs/softlogic_vibert_state.pt \
    --comment "em yêu anh quá đi 😍😍😍" \
    --verbose

# Xuất kết quả định dạng JSON
python -m softlogic_vibert.inference \
    --ckpt softlogic_outputs/softlogic_vibert_state.pt \
    --comment "ngon thật đấy 🙄" \
    --json
```

### Sử dụng qua Python API:

```python
from softlogic_vibert import SentimentPredictor

path = "outputs/softlogic_vibert_state.pt"
device = "cpu"  # hoặc "cuda"

model = SentimentPredictor.load(path, device=device)

comment = "ngon thật đấy 🙄"
context = "Đây là nhà hàng mới mở"

# Lưu ý: predict(comment, context). Sử dụng từ khóa để tránh nhầm lẫn thứ tự đối số.
pred = model.predict(comment=comment, context=context)
print("Nhãn dự đoán là:", pred)
```

### Nghiên cứu Bóc tách (Ablation Studies)

```bash
# Chạy các thử nghiệm cốt lõi
python -m softlogic_vibert.ablation run \
    --data-path output_data.json \
    --experiments core \
    --epochs 5

# Chạy tất cả các thử nghiệm
python -m softlogic_vibert.ablation run \
    --data-path output_data.json \
    --experiments all

# So sánh kết quả
python -m softlogic_vibert.ablation compare \
    --study-dir ablation_results/ablation_study_YYYYMMDD_HHMMSS
```

## Cấu hình Bóc tách

| Cấu hình | Masking | Multi-view | Logic | Mô tả |
|--------------|---------|------------|-------|-------------|
| `vibert_only` | Không | Không | Không | Chỉ sử dụng mô hình gốc (ViDeBERTa) |
| `mask_only` | Có | Không | Không | Backbone + TF-IDF gating |
| `multiview_no_logic` | Có | Có | Không | Multi-view nhưng không có suy luận logic |
| `full_model` | Có | Có | Có | Kiến trúc đầy đủ |
| `drop_r1` - `drop_r5` | Có | Có | Có* | Loại bỏ từng quy tắc riêng lẻ |

Các cờ dòng lệnh:
```bash
# Baseline chỉ backbone
--no-mask --no-multiview --no-logic

# Chỉ masking
--use-mask --no-multiview --no-logic

# Multi-view (không logic)
--use-mask --use-multiview --no-logic

# Mô hình đầy đủ
--use-mask --use-multiview --use-logic

# Loại bỏ các quy tắc cụ thể
--drop-rules r1,r5
```

## Định dạng Dữ liệu

Định dạng JSON/JSONL mong muốn:

```json
{
    "comment": "ngon thật đấy 🙄",
    "context": "Đây là nhà hàng mới mở",
    "labels": ["Disgust", "Sarcasm"]
}
```

- `comment`: Bắt buộc. Văn bản người dùng (từ lóng GenZ, emoji, ngôn ngữ phi chính thức).
- `context`: Tùy chọn. Thông tin ngữ cảnh bổ sung.
- `labels`: Danh sách đa nhãn (ví dụ: Enjoyment, Anger, Disgust, Surprise, Fear, Sadness, Other).

## Khả năng Giải thích (Interpretability)

### Phân tích chi tiết qua API

```python
from softlogic_vibert import create_interpreter

# Tạo trình giải thích từ checkpoint
interpreter = create_interpreter("outputs/softlogic_vibert_state.pt")

# Phân tích một mẫu dữ liệu
result = interpreter.analyze(
    comment="ngon thật đấy 🙄",
    context="Nhà hàng này review 5 sao"
)

# Lấy phân tích chi tiết
print(result.predicted_labels)       # ['Disgust']
print(result.rule_activations)       # Kích hoạt luật suy luận: {'r1': 0.82, 'r2': 0.21, ...}
print(result.important_tokens)       # Các token quan trọng: ['ngon', 'thật', 'đấy']
print(result.reasoning_summary)      # Tóm tắt suy luận: {'sarcasm_likely': True, ...}

# Giải thích bằng ngôn ngữ tự nhiên
explanation = interpreter.explain_prediction(
    comment="ngon thật đấy 🙄"
)
print(explanation)
```

### Truy cập Token Masks

```python
from softlogic_vibert import load_model, predict_single

model, tokenizer, ckpt = load_model("outputs/softlogic_vibert_state.pt")

result = predict_single(
    model, tokenizer,
    comment="em yêu anh quá đi",
    return_details=True
)

# Mức độ quan trọng của token
for token_info in result["token_masks"]:
    print(f"{token_info['token']}: {token_info['mask_value']:.3f}")

# Kích hoạt quy tắc
for rule, value in result["rule_activations"].items():
    print(f"{rule}: {value:.3f}")
```

## Mô đun Logic Mờ (Soft Logic Module)

### Các phép toán có thể đạo hàm

```python
# Fuzzy AND (tích t-norm tham số hóa)
# p là một scalar học được (được kẹp trong code)
AND(a, b) = (a * b) ** p

# Fuzzy OR (tổng xác suất)
OR(a, b) = a + b - a*b

# Fuzzy NOT (phần bù)
NOT(a) = 1 - a

# Fuzzy implication (toán tử Reichenbach)
IMPLIES(a, b) = 1 - a + a*b
```

### Quy tắc Suy luận

| Quy tắc | Công thức | Ý nghĩa giải thích |
|------|---------|----------------|
| r1 | `AND(P_pos_lex, P_neg_sem)` | Mỉa mai / Contradiction (Bề mặt tích cực, ý nghĩa tiêu cực) |
| r2 | `AND(P_neg_lex, P_neg_sem)` | Tiêu cực mạnh (Cả từ vựng và ngữ nghĩa đều tiêu cực) |
| r3 | `AND(P_pos_sem, NOT(P_high_int))` | Tích cực nhẹ (Ngữ nghĩa tích cực nhưng cường độ không cao) |
| r4 | `AND(P_high_int, P_neg_sem)` | Tiêu cực dữ dội (Cường độ cao + ngữ nghĩa tiêu cực) |
| r5 | `\|P_pos_lex - P_pos_sem\|` | Sự không nhất quán (Mâu thuẫn giữa cảm xúc bề mặt và ngữ nghĩa) |

## Mẹo Huấn luyện

### Xử lý Mất cân bằng Lớp

```bash
# Sử dụng focal loss cho dữ liệu mất cân bằng
python -m softlogic_vibert.train \
    --loss-type focal \
    --focal-gamma 2.0 \
    ...

# Sử dụng asymmetric loss (khuyến nghị cho đa nhãn)
python -m softlogic_vibert.train \
    --loss-type asymmetric \
    ...
```

### Điều chỉnh Tốc độ Học

```bash
# Tốc độ học khác nhau cho encoder và các head
python -m softlogic_vibert.train \
    --lr 2e-5 \
    --encoder-lr 2e-5 \
    --head-lr 5e-4 \
    ...
```

### Huấn luyện với Độ chính xác Hỗn hợp

```bash
# Bật FP16 để huấn luyện nhanh hơn (yêu cầu CUDA)
python -m softlogic_vibert.train \
    --fp16 \
    ...
```

## Cấu trúc Dự án

```
softlogic_vibert/
├── __init__.py          # Khai báo package
├── config.py            # Các dataclass cấu hình
├── model.py             # Mô hình SoftLogicViBERT cốt lõi
├── train.py             # Kịch bản huấn luyện
├── inference.py         # Kịch bản dự đoán
├── losses.py            # Các hàm loss tùy chỉnh
├── metrics.py           # Các độ đo đánh giá
├── data.py              # Các tiện ích tải dữ liệu
├── utils.py             # Các hàm trợ giúp
├── ablation.py          # Trình chạy nghiên cứu bóc tách
├── interpretability.py  # Công cụ phân tích mô hình
└── README.md            # File này
```

## Các Checkpoint đã Lưu

Quá trình huấn luyện tạo ra:
- `softlogic_vibert_state.pt`: State dict (nhẹ, khuyến nghị)
- `softlogic_vibert_full.pt`: Đối tượng mô hình đầy đủ
- `train_summary.json`: Tóm tắt các độ đo huấn luyện
- `training_history.json`: Các độ đo theo từng epoch
- `config.json`: Cấu hình thử nghiệm

## Sử dụng Nâng cao

### Vòng lặp Huấn luyện Tùy chỉnh

```python
from softlogic_vibert import (
    SoftLogicViBERT, ModelConfig, SoftLogicLoss,
    load_and_prepare_data, prepare_dataloaders
)
from transformers import AutoTokenizer
import torch

# Cấu hình
config = ModelConfig(
    model_name="Fsoft-AIC/videberta-base",
    use_mask=True,
    use_multiview=True,
    use_logic=True,
)

# Tải dữ liệu
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
train_rows, val_rows, label_map, label_list = load_and_prepare_data("data.json")
train_loader, val_loader = prepare_dataloaders(
    train_rows, val_rows, tokenizer, label_map
)

# Create model
config.num_labels = len(label_map)
model = SoftLogicViBERT(config).cuda()

# Custom training...
```

### Extending the Logic Module

```python
class ExtendedLogicModule(SoftLogicModule):
    def __init__(self, proj_size):
        super().__init__(proj_size)
        # Add custom predicates
        self.pred_custom = SoftPredicate(proj_size)
    
    def forward(self, z_sem, z_lex, z_prag):
        base_rules, base_details = super().forward(z_sem, z_lex, z_prag)
        
        # Add custom rule
        p_custom = self.pred_custom(z_sem)
        r_custom = self.AND(p_custom, base_details["p_high_int"])
        
        # Extend outputs...
        return extended_rules, extended_details
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{softlogic_vibert,
    title = {SoftLogic ViDeBERTa (SoftLogic ViBERT): Soft-Logic Reasoning for Vietnamese GenZ Sentiment},
  year = {2024},
  description = {A novel sentiment analysis architecture with differentiable fuzzy logic}
}
```

## License

This project is released for research purposes.

## Acknowledgments

- ViDeBERTa backbone (Fsoft-AIC/videberta-base)
- HuggingFace Transformers
- PyTorch team
