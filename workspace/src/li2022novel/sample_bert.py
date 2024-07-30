import torch
from torch import nn, optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# トークナイザーとモデルのロード
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# モデルをトレーニングモードに
model.train()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# データの準備
texts = ["Hello, how are you?", "I am a student studying machine learning."]  # サンプルデータ
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

# ラベルを作成（入力シーケンスを次のトークンとしてシフト）
labels = inputs['input_ids'].clone()
labels[:, :-1] = inputs['input_ids'][:, 1:]
labels[:, -1] = -100  # 最後のトークンには損失を計算しない
inputs['labels'] = labels

# オプティマイザーの設定
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# トレーニングループ
epochs = 3
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# モデルを保存
model.save_pretrained('./transformer_token_prediction_model')
tokenizer.save_pretrained('./transformer_token_prediction_model')