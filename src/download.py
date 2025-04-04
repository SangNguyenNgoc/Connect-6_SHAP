import kaggle as kg
import os

os.environ['KAGGLE_USERNAME'] = 'nguynngcsang'
os.environ['KAGGLE_KEY'] = 'e7eef7088221d6e641c2c4169dadca0b'

kg.api.authenticate()
# Đặt thư mục lưu file tải về
output_dir = "D:/PyCharmproject/AlphaSix/src/kaggle/output"
os.makedirs(output_dir, exist_ok=True)

# Tải dữ liệu từ kernel
kg.api.kernels_output("nguynngcsang/alphasix", path=output_dir)

print("Download completed!")
