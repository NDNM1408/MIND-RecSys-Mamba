import logging
import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class MambaBlock(nn.Module):
    def __init__(self, config):
        super(MambaBlock, self).__init__()
        self.config = config

        # Convolutional layer with padding to maintain sequence length
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.d_conv,
            padding=(config.d_conv - 1) // 2,  # Ensure sequence length is preserved
            groups=config.hidden_size,
        )

        # Feed-forward layer
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states

        # Ensure hidden_states has the correct shape: (batch_size, seq_len, hidden_size)
        print("AAAAA", hidden_states.shape)

        # Apply 1D convolution
        hidden_states = hidden_states.transpose(1, 2)  # Shape: (batch_size, hidden_size, seq_len)
        hidden_states = self.conv(hidden_states)       # Shape: (batch_size, hidden_size, seq_len)
        hidden_states = hidden_states.transpose(1, 2)  # Shape: (batch_size, seq_len, hidden_size)

        # Apply feed-forward layer
        hidden_states = self.fc(hidden_states)

        # Apply dropout and LayerNorm with residual connection
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual)

        return hidden_states


class MambaEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(MambaEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList([MambaBlock(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Support multiple different poolers with shared encoder
        self.poolers = nn.ModuleList()
        if config.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_embs, attention_mask=None, pooler_index=0):
        # input_embs: batch_size, seq_len, emb_dim
        # attention_mask: batch_size, seq_len (not used in Mamba, but kept for compatibility)

        embeddings = input_embs
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        all_hidden_states = [embeddings]

        for layer_module in self.encoders:
            layer_outputs = layer_module(all_hidden_states[-1], attention_mask)
            all_hidden_states.append(layer_outputs)

        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output


class MambaModel(nn.Module):
    def __init__(self, config):
        super(MambaModel, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.mamba_encoder = MambaEncoder(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, inputs, mask):
        text_vec = self.mamba_encoder(inputs, mask)  # Shape: (batch_size, hidden_size)
        
        return text_vec