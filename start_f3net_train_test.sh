



apt install -q tesseract-ocr
pip install -q pytesseract

pip install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX




git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# !cp -a /content/datasets/imgs/pascal/. /content/F3Net/data/PASCAL-S/image/
# !cp -a /content/datasets/masks/pascal/. /content/F3Net/data/PASCAL-S/mask/




wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv /content/resnet50-19c8e357.pth /content/F3Net/res/


python /content/F3Net/src/train.py
python /content/F3Net/src/test.py