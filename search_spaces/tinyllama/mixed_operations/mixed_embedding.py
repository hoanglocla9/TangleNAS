import torch.nn as nn
import torch.nn.functional as F
import torch 
from torch.nn.modules.sparse import Embedding
class MixedEmbedding(Embedding):
    def __init__(self, hidden_size_list, embedding) -> None:
        super().__init__(embedding.num_embeddings, embedding.embedding_dim)
        self.hidden_size_list = hidden_size_list
        self.weight =  embedding.weight

        self.max_hidden_size = max(self.hidden_size_list)

    def sample_weights_and_bias(self, hidden_size):
        weight = self.weight[:,:hidden_size]
        return weight
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            hidden_size_weight = self.sample_weights_and_bias(self.hidden_size_list[weights_max])
            emb_weight_padded = weights[weights_max]*hidden_size_weight
            out = F.embedding(x, emb_weight_padded)
            return out
        else:
            embedding_mixture = 0
            i=0
            for hidden_size in self.hidden_size_list:
                hidden_size_weight = self.sample_weights_and_bias(hidden_size)
                emb_weight_padded = F.pad(hidden_size_weight, (0,self.max_hidden_size - hidden_size_weight.shape[-1]), "constant", 0)
                embedding_mixture+=weights[i] * emb_weight_padded
                i+=1
            #print("embedding_mixture:", embedding_mixture)
            out = F.embedding(x, embedding_mixture)
            return out
        
# test mixed linear emb
'''if __name__ == "__main__":
    emb_dim_list = [10, 20, 30]
    max_out_dim = 30
    linear_layer = nn.Embedding(2, 30)
    linear_layer.weight.data = torch.ones_like(linear_layer.weight.data)
    mixed_linear_emb = MixedEmbeddingV2(emb_dim_list, max_out_dim, linear_layer)
    x = torch.tensor([0, 1, 0, 1]).reshape(2, 2)
    weights = [0.1, 0.7, 0.2]
    out = mixed_linear_emb(x, weights, use_argmax=True)
    print(out)
    print(out.shape)'''