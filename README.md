# LUCY: Linguistic Understanding and Control Yielding Early Stage of Her

## Requirements
```
git clone https://github.com/VITA-MLLM/LUCY.git
cd LUCY
conda create -n lucy python=3.10 -y
conda activate lucy
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Stage 1 
Aligned audio encoder is available [here](https://huggingface.co/VITA-MLLM/LUCY-Audio-Encoder-110kh).
### Stage 2 & 3 Training
Run the following scripts to continue training stage 2 and stage 3.
```
./s2p0.sh
./s2p1.sh
./s3.sh
```

## Demo of Emotion Control
https://github.com/user-attachments/assets/80120730-a37b-4ed5-8da6-7584156a6a67

## Demo of Function Calls
https://github.com/user-attachments/assets/1826bad6-207b-426d-8a99-ce7d684e20f2

## Demo of Natural Conversation
https://github.com/user-attachments/assets/86e9995a-998f-4bbb-8c2e-c51311293cb4

