# 1. Sử dụng image Python chính thức
FROM python:3.12-slim--bookworm

# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Sao chép toàn bộ code và requirement.txt vào container
COPY . .

# 4. Cài đặt các thư viện cần thiết từ requirement.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. Chạy script chính (thay bằng file bạn muốn, ví dụ main.py)
CMD ["python", "main.py"]


## Bước 1: Build Docker image
#docker build -t ppo-xapp .

# Bước 2: Run container
#docker run --rm -it ppo-xapp