import torch.nn as nn
import torch.nn.functional as F
import torch 
class MixedEmbeddingV2(nn.Module):
    def __init__(self, embed_dim_list, max_embed_dim, embedding) -> None:
        super().__init__()
        self.embed_dim_list = embed_dim_list
        self.embedding = embedding
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, embed_dim):
        weight = self.embedding.weight[:,:embed_dim]
        return weight
    
    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights_max = torch.argmax(torch.tensor(weights), dim=-1)
            emb_weight = self.sample_weights_and_bias(self.embed_dim_list[weights_max])
            emb_weight_padded = weights[weights_max]*emb_weight
            out = F.embedding(x, emb_weight)
            return out
        else:
            embedding_mixture = 0
            i=0
            for embed_dim in self.embed_dim_list:
                #print(self.embed_dim_list)
                emb_weight = self.sample_weights_and_bias(embed_dim)
                emb_weight_padded = F.pad(weights[i]*emb_weight, (0,self.max_embed_dim - emb_weight.shape[-1]), "constant", 0)
                #print(weights)
                embedding_mixture+= emb_weight_padded
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