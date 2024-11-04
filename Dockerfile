FROM python:3.10
# Set the working directory
WORKDIR /app
# Copy requiremnt.txt file tp app directory
COPY requirement.txt .
COPY unet_sentinel_resnet.py .
COPY arecanut-unetresnetsent2-128-epoch=333-val_JaccardIndex=0.46474990248680115.ckpt .
COPY main.py .

RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install the libraries
RUN pip install "fastapi[standard]"
RUN pip install -r requirement.txt
# Set the default command to Python3
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]