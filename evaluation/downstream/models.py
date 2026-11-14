import torch
import torch.nn as nn
import math


class AudioEmbedding(nn.Module):
    def __init__(self, feature_dim, d_model, max_position_embeddings, dropout=0.1):
        super(AudioEmbedding, self).__init__()
        
        # Audio feature embedding (projecting MFCC/Mel-Spectrogram to d_model size)
        self.feature_embedding = nn.Linear(feature_dim, d_model)
        
        # Positional embedding to capture the temporal information in the sequence
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_length, feature_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Create positional encodings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # (batch_size, seq_length)
        
        # Project audio features to the model dimension
        audio_embedding = self.feature_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional embeddings
        embeddings = audio_embedding + self.position_embedding(position_ids)
        
        # Add [CLS] token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # (batch_size, seq_length + 1, d_model)
        
        # Apply LayerNorm and Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_position_embeddings, dropout=0.1):
        super(PhonemeEmbedding, self).__init__()
        
        # Phoneme embedding (embedding phoneme IDs to d_model size)
        self.phoneme_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding to capture the temporal information in the sequence
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_length), input is phoneme IDs (integers)
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Create positional encodings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # (batch_size, seq_length)
        
        # Embed the phoneme IDs
        phoneme_embedding = self.phoneme_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional embeddings
        embeddings = phoneme_embedding + self.position_embedding(position_ids)
        
        # Add [CLS] token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # (batch_size, seq_length + 1, d_model)
        
        # Apply LayerNorm and Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(query, key, value, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        output = self.fc_out(attn_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feedforward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x


class BERTForAudioClassification(nn.Module):
    def __init__(self, feature_dim, d_model, num_heads, d_ff, num_layers, num_classes, max_position_embeddings, dropout=0.1):
        super(BERTForAudioClassification, self).__init__()
        
        # Audio embedding
        self.embedding = AudioEmbedding(feature_dim, d_model, max_position_embeddings, dropout)
        
        # Stacking multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, feature_dim)
        x = self.embedding(x)
        
        # Pass through all transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Extract the [CLS] token's representation (first token)
        cls_output = x[:, 0, :]  # (batch_size, d_model), taking the first token's output
        
        # Classification layer
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        logits = torch.sigmoid(logits)
        return logits


class BERTForPhonemeClassification(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_position_embeddings, dropout=0.1):
        super(BERTForPhonemeClassification, self).__init__()
        
        # Phoneme embedding
        self.embedding = PhonemeEmbedding(vocab_size, d_model, max_position_embeddings, dropout)
        
        # Stacking multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        # x: (batch_size, seq_length), input is phoneme IDs
        x = self.embedding(x)
        
        # Pass through all transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Extract the [CLS] token's representation (first token)
        cls_output = x[:, 0, :]  # (batch_size, d_model), taking the first token's output
        
        # Classification layer
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        logits = torch.sigmoid(logits)  # Since it's a multi-label classification task
        return logits
