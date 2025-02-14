# BERT Chinese Emotion Classification

A Python notebook demonstrating fine-tuning of BERT model for Chinese emotion classification using the Hugging Face Transformers library. This project shows how to train and use a BERT model to classify Chinese text into 8 different emotional categories.

## Project Structure
```
.
├── finetune.ipynb
├── predict.ipynb
├── output/
│   ├── [trained model files]
│   └── [tokenizer files]
├── requirements.txt
```

## Installing
```bash
pip install -r requirements.txt
```
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face Transformers library
- `datasets` - For dataset handling
- `evaluate` - For model evaluation
- `accelerate` - For optimized training
- `scikit-learn` - For metrics calculation

## Usage
1. Mount Google Drive (if using Google Colab)
2. Run finetune.ipynb to train the model
3. Use predict.ipynb to classify new texts

Example prediction:
```python
texts = [
    我每天都能跟她一起上學，我好開心！ => 開心語調 (0.97)
    最好的朋友要離開臺灣了，以後可能不容易再見面... => 悲傷語調 (0.90)
    我覺得我快不行了... => 悲傷語調 (0.78)
    剛剛收到研究所錄取的通知書！ => 開心語調 (0.94)
    今年的冬天好像比較晚來。 => 平淡語氣 (0.95)
]
```

## Emotion Categories
The model classifies text into 8 emotional categories:
- 平淡語氣 (Neutral)
- 開心語調 (Happy)
- 關切語調 (Caring)
- 憤怒語調 (Angry)
- 驚奇語調 (Surprised)
- 悲傷語調 (Sad)
- 厭惡語調 (Disgusted)
- 疑問語調 (Questioning)

## Data Sources and Model Base
### Training Dataset
- [Datasets:Johnson8187/Chinese Multi-Emotion Dialogue Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)
- A comprehensive dataset for Chinese emotion classification

### Base Model
- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
- Pre-trained Chinese BERT model from Google

## Model Details
- Max sequence length: 512
- Training epochs: 5
- Batch size: 16 (effective batch size: 32 with gradient accumulation)
- Learning rate: 2e-5
- Warmup ratio: 0.1
- Weight decay: 0.01
- Gradient accumulation steps: 2
- Training/validation split: 80/20

## Demo
Click this image to watch the demo video

[![BERT Chinese Emotion Classification - BERT中文情緒分析練習](https://img.youtube.com/vi/_mD591UNjw8/0.jpg)](https://www.youtube.com/watch?v=_mD591UNjw8)

## Versioning
- Python 3.10

## Authors
[solano66](https://github.com/solano66)

## Acknowledgments
- Hugging Face team for the Transformers library
- Google BERT team for the pre-trained model
- Johnson8187 for the Chinese Multi-Emotion Dialogue Dataset
### Thank you to all the people who release code on GitHub