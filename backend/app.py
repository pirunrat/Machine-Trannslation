from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS from flask_cors
from model import Model
import torch.nn as nn
import nltk




try:
    path = './files/MultiplicativeAttention.pt'
    model_instance = Model(path)
    print("Model has been loaded successfully")

    # Download the "punkt" resource
    nltk.download('punkt')
    
except KeyError:
    print(f'Error:{KeyError}')

app = Flask(__name__)
# Allow requests from 'http://localhost:3000' to the '/make_recommendations' route
CORS(app, origins=["*"])

@app.route('/')
def hello():
    return 'Hello, World!'



@app.route('/test', methods=['POST'])
def Test():
    data = request.get_json()
    print(data)
    try:
        data = request.get_json()
        message = data.get('sentence', '')

        prediction = model_instance.inference(message)
        text = ' '.join(prediction)
       
        print("Prompt message:", message)
        print(f"text:{text}")

        response = {'Answer': text}
        return jsonify(response)
    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"}), 404
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
