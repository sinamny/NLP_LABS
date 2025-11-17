import torch
import numpy as np
from torch import nn

print("PHẦN 1. TENSOR")

# Task 1. Tạo tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("Tensor từ list:\n", x_data, "\n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("Tensor từ Numpy array: \n", x_np, "\n")

x_ones = torch.ones_like(x_data) # tạo tensor gồm các số 1 có cùng shape với x_data
print("Ones Tensor: \n", x_ones, "\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print("Random Tensor: \n", x_rand, "\n")

print("Shape của tensor", x_rand.shape)
print("Dtype của tensor:", x_rand.dtype)
print("Device lưu trữ của tensor:", x_rand.device, "\n")


# Task 1.2: Các phép toán
print("x_data + x_data =", x_data + x_data, "\n")
print("x_data * 5 =", x_data * 5, "\n")
print("x_data @ x_data.T =", x_data @ x_data.T, "\n")


# Task 1.3: Indexing & Slicing
print("Hàng đầu:", x_data[0])
print("Cột thứ hai:", x_data[:, 1])
print("Phần tử (2, 2): ", x_data[1, 1], "\n")

# Task 1.4:Reshape Tensor
x = torch.rand(4, 4)
x_reshaped = x.view(16, 1)
print("Tensor 4x4: \n", x)
print("Tensor reshape 16x1: \n", x_reshaped, "\n")

print("PHẦN 2. AUTOGRAD")
x = torch.ones(1, requires_grad=True)
print("x =", x)

y = x + 2
print("y =", y)
print("grad_fn của y:", y.grad_fn)

z = y * y * 3
z.backward()

print("Đạo hàm dz/dx:", x.grad)
print("\nNếu gọi z.backward() lần nữa sẽ lỗi vì đồ thị đã giải phóng.\n")


print("PHẦN 3. XÂY DỰNG MÔ HÌNH")
# Task 3.1: nn.Linear
linear_layer = nn.Linear(5, 2)
input_tensor = torch.randn(3, 5)
output = linear_layer(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
print("Output:\n", output, "\n")

# Task 3.2: nn.Embedding
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
input_indices = torch.LongTensor([1, 5, 0, 8])
embeddings = embedding_layer(input_indices)

print("Input indices shape:", input_indices.shape)
print("Embedding output shape:", embeddings.shape)
print("Embeddings:\n", embeddings, "\n")

# Tassk 3.3: nn.Module model
class MyFirstModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyFirstModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, indices):
        embeds = self.embedding(indices)
        hidden = self.activation(self.linear(embeds))
        output = self.output_layer(hidden)
        return output
model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)

input_data = torch.LongTensor([[1,2,5,9]])
output_data = model(input_data)

print("Model output shape:", output_data.shape)
print("Model output:\n", output_data)