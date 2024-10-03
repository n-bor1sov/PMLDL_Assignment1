from flask import Flask, request, jsonify
from flask import abort
from typing import Dict
import requests
import os
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        print(x)
        x = self.sequential(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
model = model.to(device)

base_model = BinaryClassifier()
base_model.load_state_dict(torch.load('model/best.pt', map_location=device))
base_model.to(device)
base_model.eval()



def tokenize_and_embed(text):
    # Tokenize the text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get the embeddings
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.pooler_output
    
    return embeddings

@app.route('/sentiment-analisys', methods=['POST'])
def check_video_duplicate():
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        embedding = tokenize_and_embed(text)
        with torch.no_grad():
            embedding = embedding.to(device)
            output = base_model(embedding)
            prediction = torch.round(F.sigmoid(output))
        
        print(f'Prediction: {prediction}')
        answer = 'positive' if prediction == 1 else 'negative'
        return jsonify({'sentiment': str(answer)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="3010")