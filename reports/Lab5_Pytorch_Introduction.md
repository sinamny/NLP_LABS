# **Lab 5: PyTorch Introduction**

## **1. Mục tiêu**

Bài lab này giúp sinh viên làm quen với PyTorch – một trong những thư viện Deep Learning phổ biến và mạnh mẽ nhất hiện nay. Sau bài thực hành, sinh viên sẽ:

* Hiểu và thao tác với Tensor – cấu trúc dữ liệu cốt lõi của PyTorch.
* Sử dụng Autograd để tính đạo hàm tự động.
* Biết cách xây dựng một mô hình Neural Network đơn giản bằng `nn.Module`.
* Làm quen với hai lớp quan trọng:

  * `nn.Linear`
  * `nn.Embedding`


## **2. Hướng dẫn chạy code**
### **2.1. Cấu trúc thư mục chính**

```
nlp-labs/
│
├── labs/
│   ├── lab1/                     # Lab 1: Tokenizer
│   ├── lab2/                     # Lab 2: Vectorizer
│   ├── lab4/                     # Lab 4: Word embeddings
│   ├── lab5/                     # Lab 5: Sentiment analysis & text preprocessing
│   ├── lab5_2/                   # Lab 5 phần nâng cao: RNN for NER
│   │   ├── lab5_pytorch_intro.py # Mã chính
│   └── __init__.py
```
> **Chú thích:**
>
> * Tất cả mã nguồn của Lab 5 (bao gồm RNN for NER) nằm trong `labs/lab5_2`.
> * File chính chạy trực tiếp: `lab5_pytorch_intro.py`.
### **2.2. Cài đặt môi trường (sử dụng `requirements.txt`)**

1. Tạo môi trường Python (Python ≥ 3.10):

```bash
python -m venv nlp-lab-env
source nlp-lab-env/bin/activate   # Linux/Mac
nlp-lab-env\Scripts\activate      # Windows
```

2. Cài đặt tất cả thư viện từ `requirements.txt`:

```bash
pip install -r requirements.txt
```
### **2.3. Chạy Lab 5: Pytorch Introduction**

Tất cả mã nguồn được đặt trong:

```
labs/lab5_2/lab5_pytorch_intro.py
```

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy Lab 5
 python -m labs.lab5_2.lab5_pytorch_intro
```

## **3. Các bước thực hiện**
### **PHẦN 1: TENSOR**

Tensor trong PyTorch tương tự như `numpy.ndarray`, nhưng mạnh hơn vì nó có thể chạy trên GPU và có hỗ trợ autograd.


### **Task 1.1 – Tạo Tensor**

#### **Các bước thực hiện**

1. Tạo Tensor từ  list bằng `torch.tensor`.
2. Tạo Tensor từ NumPy array bằng `torch.from_numpy`.
3. Tạo Tensor bằng các hàm khởi tạo (`ones_like`, `rand_like`).
4. In ra:

   * `shape`
   * `dtype`
   * `device`

#### **Code **

```python
import torch
import numpy as np

# Tạo tensor từ list
x_data = torch.tensor([[1, 2], [3, 4]])
print("Tensor từ list:\n", x_data)

# Tạo tensor từ numpy array
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)
print("\nTensor từ Numpy array:\n", tensor_from_np)

# Tensor ones & random
ones_tensor = torch.ones_like(x_data)
rand_tensor = torch.rand_like(x_data, dtype=torch.float)
print("\nOnes Tensor:\n", ones_tensor)
print("\nRandom Tensor:\n", rand_tensor)

# Các thuộc tính của tensor
print("\nShape của tensor:", x_data.shape)
print("Dtype của tensor:", x_data.dtype)
print("Device lưu trữ của tensor:", x_data.device)
```

#### **Kết quả**

```
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])

Tensor từ Numpy array:
 tensor([[1, 2],
        [3, 4]])

Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.7485, 0.3161],
        [0.9889, 0.3378]])

Shape của tensor torch.Size([2, 2])
Dtype của tensor: torch.float32
Device lưu trữ của tensor: cpu
```


### **Task 1.2 – Các phép toán Tensor**

#### **Các bước thực hiện**

1. Cộng hai tensor cùng kích thước.
2. Nhân toàn bộ tensor với một số.
3. Nhân ma trận bằng toán tử `@`.

#### **Code**

```python
print("x_data + x_data =", x_data + x_data)
print("\nx_data * 5 =", x_data * 5)
print("\nx_data @ x_data.T =", x_data @ x_data.T)
```

#### **Kết quả**

```python
x_data + x_data = tensor([[2, 4],
        [6, 8]])

x_data * 5 = tensor([[ 5, 10],
        [15, 20]])

x_data @ x_data.T = tensor([[ 5, 11],
        [11, 25]])
```


### **Task 1.3 – Indexing & Slicing**

#### **Các bước thực hiện**

1. Lấy hàng đầu tiên.
2. Lấy cột thứ hai.
3. Truy xuất phần tử tại vị trí (2,2).

#### **Code**

```python
print("Hàng đầu:", x_data[0])
print("Cột thứ hai:", x_data[:, 1])
print("Phần tử (2, 2): ", x_data[1, 1])
```

#### **Kết quả**

```
Hàng đầu: tensor([1, 2])
Cột thứ hai: tensor([2, 4])
Phần tử (2, 2):  tensor(4)
```


### **Task 1.4 – Reshape Tensor**

#### **Các bước thực hiện**

1. Tạo tensor 4×4 bằng `torch.rand`.
2. Reshape thành tensor 16×1 bằng `.view()` hoặc `.reshape()`.

#### **Code**

```python
tensor4x4 = torch.rand(4, 4)
print("Tensor 4x4:\n", tensor4x4)

reshaped = tensor4x4.reshape(16, 1)
print("\nTensor reshape 16x1:\n", reshaped)
```

#### **Kết quả**
```
Tensor 4x4:
 tensor([[0.9199, 0.0490, 0.9510, 0.4633],
        [0.5430, 0.5306, 0.8856, 0.5500],
        [0.1272, 0.2303, 0.1343, 0.9918],
        [0.4935, 0.9565, 0.5976, 0.8111]])

Tensor reshape 16x1:
 tensor([[0.9199],
        [0.0490],
        [0.9510],
        [0.4633],
        [0.5430],
        [0.5306],
        [0.8856],
        [0.5500],
        [0.1272],
        [0.2303],
        [0.1343],
        [0.9918],
        [0.4935],
        [0.9565],
        [0.5976],
        [0.8111]])
```


### **PHẦN 2: AUTOGRAD**

Autograd cho phép PyTorch tự động tính đạo hàm, cực kỳ quan trọng khi huấn luyện mạng nơ-ron.


### **Task 2.1 – Sử dụng Autograd**

#### **Các bước thực hiện**

1. Tạo tensor `x` với `requires_grad=True`.
2. Tính:

$y = x + 2$
$z = y^2 \times 3$

3. Gọi `z.backward()` để tính gradient.

#### **Code**

```python
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3

print("x =", x)
print("y =", y)
print("grad_fn của y:", y.grad_fn)

z.backward()
print("Đạo hàm dz/dx:", x.grad)
```

#### **Kết quả**

```
x = tensor([1.], requires_grad=True)
y = tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at ...>
Đạo hàm dz/dx: tensor([18.])
```

#### **Giải thích**


$y = x + 2 \quad\Rightarrow\quad y = 3$

$z = 3y^2 = 3 \times 9 = 27$

$\frac{dz}{dx} = 6y = 6 \times 3 = 18$

#### Lưu ý quan trọng

Gọi `z.backward()` lần thứ 2 sẽ báo lỗi vì đồ thị đã bị giải phóng.
Muốn gọi nhiều lần phải dùng:

```
z.backward(retain_graph=True)
```


### **PHẦN 3: XÂY DỰNG MÔ HÌNH**


### **Task 3.1 – Lớp `nn.Linear`**

#### **Các bước thực hiện**

1. Khai báo một layer Linear: input 5 → output 2.
2. Tạo dữ liệu đầu vào kích thước (3, 5).
3. Truyền vào layer để lấy output.

#### **Code**

```python
import torch.nn as nn

linear = nn.Linear(5, 2)
inputs = torch.rand(3, 5)

outputs = linear(inputs)
print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)
print("Output:\n", outputs)
```

#### **Kết quả**

```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
...
```


### **Task 3.2 – Lớp `nn.Embedding`**

#### **Các bước thực hiện**

1. Tạo embedding gồm 10 từ, mỗi từ có vector 3 chiều.
2. Đầu vào là các chỉ số từ (indices).
3. Trích xuất vector embedding.

#### **Code**

```python
embedding = nn.Embedding(10, 3)
indices = torch.tensor([1, 5, 2, 9])
embedded = embedding(indices)

print("Input indices shape:", indices.shape)
print("Embedding output shape:", embedded.shape)
print("Embeddings:\n", embedded)
```

#### **Kết quả**
```
Input indices shape: torch.Size([4])
Embedding output shape: torch.Size([4, 3])
Embeddings:
 tensor([[ 0.1813,  0.1492,  0.6733],
        [ 1.6073, -0.9811, -1.2778],
        [-0.5047, -0.7312,  0.7571],
        [ 0.1957, -0.5834, -0.2559]], grad_fn=<EmbeddingBackward0>)
```

### **Task 3.3 – Xây dựng mô hình bằng `nn.Module`**

#### **Các bước thực hiện**

1. Tạo class kế thừa `nn.Module`.
2. Khởi tạo:

   * Embedding layer
   * Linear layer
3. Xây dựng hàm `forward`.

#### **Code**

```python
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)     # (batch, seq_len, embed_dim)
        x = self.linear(x)        # (batch, seq_len, output_dim)
        return x

model = SimpleModel(10, 3, 4, 2)
input_data = torch.tensor([[1, 3, 5, 2]])
output = model(input_data)

print("Model output shape:", output.shape)
print("Model output:\n", output)
```

#### **Kết quả**

```
Model output shape: torch.Size([1, 4, 2]) Model output: tensor([[[ 0.1115, 0.0493], [-0.0101, 0.0770], [ 0.3323, 0.2246], [-0.0048, 0.0609]]], grad_fn=<ViewBackward0>)
```

## **4. Khó khăn và hướng giải quyết**
## Khó khăn gặp phải

1. **Hiểu cách hoạt động của Tensor và Autograd**  
   - Sinh viên mới thường nhầm lẫn giữa việc thay đổi tensor có `requires_grad=True` và không.  
   - Việc gọi `backward()` nhiều lần dễ gây lỗi do đồ thị gradient đã bị giải phóng.

2. **Xử lý kích thước tensor khi làm việc với nn.Linear và nn.Embedding**  
   - Đầu vào của `nn.Linear` yêu cầu đúng shape `(batch_size, features)`.  
   - Đầu ra của `nn.Embedding` là `(batch_size, seq_len, embedding_dim)` cần reshape khi truyền vào Linear cho các mô hình phức tạp hơn.  

3. **Khởi tạo mô hình nn.Module**  
   - Sinh viên cần nắm vững cách định nghĩa lớp, constructor, và forward function.  
   - Các lỗi phổ biến: quên gọi `super().__init__()`, sai tên layer hoặc shape tensor.

## Hướng giải quyết

- **Autograd**: Luôn kiểm tra `requires_grad` của tensor, dùng `retain_graph=True` khi cần backward nhiều lần.  
- **Shape tensor**: Sử dụng `.view()` hoặc `.reshape()` để điều chỉnh kích thước phù hợp trước khi truyền vào layer.  
- **Debug nn.Module**: In shape các tensor trong `forward` để kiểm tra đúng chiều.  
- Bắt đầu từ mô hình đơn giản, từng bước thêm layer và quan sát output trước khi xây dựng mô hình phức tạp hơn.


## **5. Kết luận**

Qua Lab 5:

* Hiểu cách tạo và thao tác tensor.
* Sử dụng được Autograd để tính gradient tự động.
* Làm quen với các layer cơ bản trong `torch.nn`.
* Xây dựng được một mô hình đơn giản kế thừa từ `nn.Module`.

# **6. Tài liệu tham khảo**
1. **PyTorch Official Documentation**
   *Tensors, Autograd, Neural Networks, Optimizers*
   [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

2. **PyTorch Tutorials – Learn the Basics**
   [https://pytorch.org/tutorials/beginner/basics/intro.html](https://pytorch.org/tutorials/beginner/basics/intro.html)

3. **Deep Learning with PyTorch: A 60 Minute Blitz**
   *Giới thiệu về tensor và autograd*
   [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

4. **Neural Networks with PyTorch**
   *Giải thích về nn.Module, nn.Linear, nn.Embedding*
   [https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

5. **Autograd Mechanics**
   [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)

6. **Machine Learning Mastery – PyTorch Tutorials**
   [https://machinelearningmastery.com/start-here/#pytorch](https://machinelearningmastery.com/start-here/#pytorch)

7. **CS231n Notes**
   [http://cs231n.github.io/](http://cs231n.github.io/)