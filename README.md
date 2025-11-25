# NLP LABS

Dự án này là tập hợp các bài lab thực hành về Xử lý Ngôn ngữ Tự nhiên (NLP) và Học sâu (DL) sử dụng Python.

## Cấu trúc thư mục chính

```
nlp-labs/
│
├── labs/                  # Thư mục chứa các bài lab (mỗi lab là một bài thực hành, các test theo lab)
│   ├── lab1/                      # Lab 1 – Giới thiệu NLP cơ bản
│   ├── lab2/                      # Lab 2 – Xử lý văn bản
│   ├── lab4/                      # Lab 4 – Word Embeddings      
│   ├── lab5/                      # Lab 5 – Text Classification
│   ├── lab5_2/                    # Lab 5 – Giới thiệu về RNNs và các bài toán  
│   └── lab6/                      # Lab 6: Transformers    
│   └── ...
│
├── reports/               # File báo cáo tiết cho từng lab
├── src/                   # Mã nguồn chính của dự án
│   ├── core/              
│   ├── preprocessing/     
│   └── representations/  
│   └── ...
│
├── data/                  # Dataset 
│   └── UD_English-EWT/
│   └── sentiments.csv 
│   └── ... 
├── results/               # Kết quả huấn luyện (model, log, biểu đồ,…)
├── models/                # Lưu các model
├── .gitignore             # File gitignore
└── requirements.txt       # Danh sách thư viện Python cần cài đặt
```