# 🤖 AI Chơi Cờ Liên Lục (Connect-6) Sử Dụng Học Tăng Cường Sâu

## 📦 Phụ thuộc Python
Dự án yêu cầu các thư viện Python sau:
- numpy  
- pytorch==0.4.0  
- PyQt5  

## 🚀 Hướng dẫn sử dụng

- Ví dụ chạy giao diện người dùng để đấu với AI:
```bash
python ui.py -s 10 -r 6 -m 800 -i model/10_10_6_best_policy_3.model
```

- Ví dụ chạy chức năng tự chơi cờ:
```bash
python train.py -s 10 -r 6 -m 800 -i model/10_10_6_best_policy_3.model
```


- Để xem tùy chọn hỗ trợ
```bash
python train.py -h
```

- Để tiến hành trích xuất đặc trưng sau quá trình tự chơi   
```bash
python init_feature
```

- Để tiến hành giải thích với SHAP   
```bash
python explain
```