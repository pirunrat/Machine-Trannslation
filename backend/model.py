import os
os.environ['TZ'] = 'Asia/Bangkok'

from utils import Utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from pythainlp import word_tokenize as thai_word_tokenize

class Model:
    def __init__(self,model_path) -> None:

        # Loading all necsessary objects
        vocabs = Utils('./files/vocab_transform.pickle')
        self.vocab_transform = vocabs.load()
        self.max_seq_len = 30
        self.seed = 0
        self.temperature = 0.5
        self.device = torch.device('cpu')
        self.src_tokenizer = thai_word_tokenize
    

        # Loading the model
        model_util = Utils(model_path)
        self.model = model_util.load_pytorch_model()
    

    
    def inference(self, src_sentence):

        SRC_LANGUAGE = 'th'
        TRG_LANGUAGE = 'en' 
        sos_idx=2
        eos_idx=3
        

        # Source Language
        src_tokens = self.src_tokenizer(src_sentence)
        src = self.vocab_transform[SRC_LANGUAGE](src_tokens)
        src = torch.tensor([src]).to(self.device)

        with torch.no_grad():
            # Initialize the trg tensor with sos token
            trg = torch.tensor([[sos_idx]]).to(self.device)

           
            output, _ = self.model(src, trg)

            print(f'output:{output.shape}')

            pred_token = output.argmax(dim=-1)

            # pred_token = output.argmax(2)[:,-1].item()

            print(f'pred_token:{pred_token}')

            trg = torch.cat((trg, torch.tensor([[pred_token]]).to(self.device)), dim=1)

            print(f'length:{len(self.vocab_transform[SRC_LANGUAGE])}')

        

        # Convert the predicted tensor to a list of tokens using trg_tokenizer
        predicted_sentence = [self.vocab_transform[TRG_LANGUAGE].get_itos()[token] for token in trg.squeeze().tolist() if token not in [sos_idx, eos_idx]]
        
        output = ''.join(predicted_sentence)
        return output
        
        

        