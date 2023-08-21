# imports
import torch 
from torch import nn 
from math import sqrt
import copy

# 1. Embedding
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size -- size of vocabulary, int
            d_model -- size of embedding, int
        Returns:
            -- nothing --
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_table = nn.Embedding(vocab_size, d_model)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor, [batch, max_len]
        Returns:
            embedded_x -- the embeddings of x, [batch, max_len, d_model]
        """
        return self.embedding_table(x)
    
# 2. Positional Embeddings
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model : int, max_len : int):
        """
        Args:
            d_model -- size of embedding, int
            max_len -- maximum size of input/output
        Returns:
            -- nothing --
        """
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.rand([max_len, d_model]), requires_grad = True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x -- input without positional encoding, [batch, max_len, d_model]
        Returns:
            x + positional encoding
        """
        # broadcasting occurs here
        return  x + self.positional_encoding
    
# 3. Multi-Head Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model : int, num_heads: int, drop_prob: float):
        """
        Args:
            d_model -- size of embeddings, int
            num_heads -- number of heads for MHA, int
            drop_prob -- probability for dropout layer
        Returns:
            -- nothing --
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.out_layer = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(drop_prob)
        self.softmax_layer = nn.Softmax(dim = -1)

    def forward(self, pre_query : torch.Tensor, pre_key : torch.Tensor, pre_value : torch.Tensor, mask : torch.Tensor = None ) -> torch.Tensor:
        """
        Args:
            pre_query, pre_key, pre_value -- the MHA input vectors, [batch, max_len, d_model]
            mask -- the MHA mask, [max_len, max_len]
        Returns:
            mha_tokens -- [batch, max_len, d_model]
        """
        batch_size = pre_query.shape[0]

        query = self.dropout_layer( self.query_layer(pre_query) ).view([batch_size, -1 , self.num_heads, self.head_dim])
        key = self.dropout_layer( self.key_layer(pre_key) ).view([batch_size, -1 , self.num_heads, self.head_dim])
        value = self.dropout_layer( self.value_layer(pre_value) ).view([batch_size, -1 , self.num_heads, self.head_dim])
        # [batch, max_len, num_head, head_dim]

        scores = torch.einsum("blhd, bLhd -> bhlL", query, key) / sqrt(self.head_dim)
        # [batch, num_head, max_len, max_len]

        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
        
        attn_scores = self.softmax_layer(scores)
        # [batch, num_head , max_len, max_len]

        mha_tokens = torch.einsum("bhlL, bLhd -> blhd", attn_scores, value)
        # [batch, max_len, num_head, head_dim]

        return self.out_layer( self.dropout_layer(mha_tokens.view(batch_size, -1, self.d_model)) )
        # [batch, max_len, d_model]


# 4. Encoder
class Encoder(nn.Module):

    def __init__(self, d_model: int, num_heads: int, drop_prob: float, expansion_factor: int):
        """
        Args:
            d_model -- size of embeddings, int
            num_heads -- number of heads for MHA, int
            drop_prob -- probability for dropout layer, float
            expansion_factor -- expansion factor for linear layer, int
        Returns:
            -- nothing --
        """
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, drop_prob)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, expansion_factor*d_model),
            nn.ReLU(),
            nn.Linear(expansion_factor*d_model, d_model),
            nn.Dropout(drop_prob)
        )
    
    def forward(self, x : torch.Tensor, mask : torch.Tensor) -> torch.Tensor :
        """
        Args:
            x -- input vector, [batch, max_len, d_model]
            mask -- mask for MHA, [max_len, max_len]
        Returns:
            encoder output [batch, max_len, d_model]
        """ 
        mha_tokens = self.mha(x, x, x, mask)
        # [batch, max_len, d_model]

        norm_1 = self.layer_norm_1(mha_tokens + x)
        # [batch, max_len, d_model]

        norm_2 = self.layer_norm_2(norm_1 + self.feed_forward(norm_1) )
        # [batch, max_len, d_model]

        return norm_2
    
# 5. Decoder

class Decoder(nn.Module):

    def __init__(self, d_model: int, num_heads: int, drop_prob: float, expansion_factor: int):
        """
        Args:
            d_model -- size of embeddings, int
            num_heads -- number of heads for MHA, int
            drop_prob -- probability for dropout layer, float
            expansion_factor -- expansion factor for linear layer, int
        Returns:
            -- nothing --
        """
        super().__init__()

        self.masked_mha = MultiHeadAttention(d_model, num_heads, drop_prob)
    def forward(self, x : torch.Tensor, enc_tokens : torch.Tensor, mask : torch.Tensor):
        """
        Args:
            x -- decoder input, [batch, max_len, d_model]
            enc_tokens -- encoder output, [batch, max_len, d_model]
            mask -- mask for masked_mha in decoder, [max_len, max_len]
        Returns:
            decoder logits [batch, max_len, d_model]
        """
        masked_mha_tokens = self.masked_mha(x, x, x, mask)

        norm_1 = self.layer_norm_1(masked_mha_tokens + x)
        
        norm_2 = self.layer_norm_2(norm_1 + self.enc_dec_mha(norm_1, enc_tokens, enc_tokens))
        
        return self.layer_norm_3( self.dropout_layer(norm_2) + self.feed_forward(norm_2) )
    

# 6. Transformer
class Transformer(nn.Module):
    
    def __init__(self, num_enc_dec:int,  enc_vocab_size: int, dec_vocab_size: int, d_model: int, max_len: int, num_heads: int, drop_prob: float, expansion_factor: int):
        """
        Args:
            num_enc_dec -- the number of encoders and decoders, int
            enc_vocab_size -- encoder vocab size, int
            dec_vocab_size -- decoder vocab size, int
            d_model -- size of embeddings, int
            max_len -- maximum input/output length, int
            num_heads -- num of heads for MHA, int
            drop_prob -- probability for dropout layers, float
            expansion_factor -- expansion factor for linear layer, int
        Returns:
            -- nothing -- 
        """
        super().__init__()

        self.encoder_embedding = Embedding(enc_vocab_size, d_model)
        self.decoder_embedding = Embedding(dec_vocab_size, d_model)

        self.encoder_positional_encoding = PositionalEmbedding(d_model, max_len)
        self.decoder_positional_encoding = PositionalEmbedding(d_model, max_len)

        self.encoders = nn.ModuleList([Encoder(d_model, num_heads, drop_prob, expansion_factor) for _ in range(num_enc_dec) ])
        self.decoders = nn.ModuleList([Decoder(d_model, num_heads, drop_prob, expansion_factor) for _ in range(num_enc_dec) ])

        self.linear_stack = nn.Sequential(
            nn.Linear(max_len * d_model, max_len),
            nn.ReLU(),
            nn.Linear(max_len, dec_vocab_size)
        )

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor, enc_mask: torch.Tensor, dec_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_x -- input for encoder, [batch, max_len]
            enc_y -- input for decoder, [batch, max_len]
            enc_mask -- mask for encoder, [max_len, max_len]
            dec_mask -- mask for decoder, [max_len, max_len]
        Returns:
            Transformer Logits
        """
        batch_size = enc_tokens.shape[0]
        enc_tokens = copy.deepcopy(enc_x)
        # [batch, max_len]

        enc_tokens = self.encoder_embedding(enc_tokens)
        enc_tokens = self.encoder_positional_encoding(enc_tokens)
        # [batch, max_len, d_model]

        for encoder in self.encoders:
            enc_tokens = encoder(enc_tokens, enc_mask)
        # [batch, max_len, d_model]

        dec_tokens = copy.deepcopy(dec_x)
        dec_tokens = self.decoder_embedding(dec_tokens)
        dec_tokens = self.decoder_positional_encoding(dec_tokens)

        for decoder in self.decoders:
            dec_tokens = decoder(dec_tokens, enc_tokens, dec_mask) # y = decoder(y, enc_tokens, mask)
        # [batch, max_len, d_model]
        
        return self.linear_stack(dec_tokens.view(batch_size, -1))
