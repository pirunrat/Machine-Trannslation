import pickle
import torch
from Transformer import Seq2SeqTransformer, Decoder, Encoder
import torch.nn as nn

class Utils:
    def __init__(self, path) -> None:
        self.path = path
        try:
            with open('./files/vocab_transform.pickle', 'rb') as f:
                loaded_variable = pickle.load(f)
            self.vocab_transform = loaded_variable
        except Exception as e:
            print(f"Error loading vocab_transform {self.path}: {e}")
            return None

    def load(self):
        with open(self.path, 'rb') as f:
            loaded_variable = pickle.load(f)
        return loaded_variable
    
    def initialize_weights(self,m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    
    def load_pytorch_model(self):
        try:
            input_dim   = len(self.vocab_transform['th'])
            output_dim  = len(self.vocab_transform['en'])
            hid_dim = 16
            enc_layers = 3
            dec_layers = 3
            enc_heads = 8
            dec_heads = 8
            enc_pf_dim = 32
            dec_pf_dim = 32
            enc_dropout = 0.1
            dec_dropout = 0.1

            SRC_PAD_IDX = 1
            TRG_PAD_IDX = 1
            device = torch.device('cpu')

            enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)

            dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)
                # Create a new instance of the Skipgram model
            loaded_model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
            loaded_model.apply(self.initialize_weights)

            # Load the saved model parameters into the new instance
            loaded_model.load_state_dict(torch.load(self.path))

             # Set the model to evaluation mode
            loaded_model.eval()
            print("Model has been loaded successfully")
            return loaded_model
        except Exception as e:
            print(f"Error loading PyTorch model from {self.path}: {e}")
            return None