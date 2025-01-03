# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import math
import clip

class PositionalEncoding(nn.Module):
    """为Transformer添加位置信息的模块"""

    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embed_size]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class ImageEncoder(nn.Module):
    def __init__(self, embed_size, freeze=True, dropout=True):
        super(ImageEncoder, self).__init__()
        self.freeze = freeze
        self.ResNet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 只保留ResNet的卷积层部分
        self.ResNet = nn.Sequential(*list(self.ResNet.children())[:-2]) # output: [batch_size, 512, 7, 7]
        # 冻结ResNet参数
        if self.freeze:
            for param in self.ResNet.parameters():
                param.requires_grad = False    
        self.dropout = nn.Dropout(0.2)
        self.should_drop = dropout
        self.fc = nn.Linear(512, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        features = self.ResNet(x)   # output: [batch_size, 512, 7, 7]
        features = features.view(features.size(0), features.size(1), -1)  # output: [batch_size, 512, 49]
        features = features.permute(0, 2, 1)  # output: [batch_size, 49, 512]
        features = self.fc(features)  # output: [batch_size, 49, embed_size]
        if self.should_drop:
            features = self.dropout(features)
        return features

class ImageEncoderCLIP(nn.Module):
    def __init__(self, embed_size, freeze=True, dropout=True, device='cpu'):
        super(ImageEncoderCLIP, self).__init__()
        self.freeze = freeze
        self.device = device
        # 加载预训练的 CLIP 模型
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()  # 设置为评估模式
        
        if self.freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.2)
        self.should_drop = dropout
        # CLIP 的图像编码器输出维度，ViT-B/32 的 embed_size 是 512
        self.fc = nn.Linear(self.clip_model.visual.output_dim, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, 3, H, W]
        Returns:
            features: Tensor, shape [batch_size, embed_size]
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # [batch_size, clip_embed_size]
        if self.should_drop:
            features = self.dropout(self.fc(features))
        else:
            features = self.fc(features)
        return features

        
class TransformerDecoder(nn.Module):
    """使用Transformer解码器生成文本描述"""

    def __init__(self, embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=0.1, encoder_type='ResNet'):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.positional_encoding = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.encoder_type = encoder_type
        
    def forward(self, captions, memory):
        """
        Args:
            captions: Tensor, shape [batch_size, seq_len]
            memory: Tensor, shape [batch_size, seq_len=49, embed_size]
        Returns:
            outputs: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        embeddings = self.embed(captions) * math.sqrt(self.embed_size)  # [batch_size, seq_len, embed_size]
        embeddings = self.positional_encoding(embeddings)  # [batch_size, seq_len, embed_size]
        embeddings = embeddings.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]

        if self.encoder_type == 'CLIP':
            # memory shape: [batch_size, embed_size]
            memory = memory.unsqueeze(0)    # [seq_len=1, batch_size, embed_size]
        else:
            # memory shape: [batch_size, 49, embed_size]
            memory = memory.permute(1, 0, 2)  # [seq_len=49, batch_size, embed_size]

        #print(f'captions: {captions}')
        tgt_mask, tgt_padding_mask = self.generate_square_subsequent_mask(captions.size(1), captions)
        #print(f'target mask: {tgt_mask}, {tgt_mask.size()}')
        #print(f'target padding mask: {tgt_padding_mask},{tgt_padding_mask.size()}')
        tgt_mask = tgt_mask.to(embeddings.device)
        tgt_padding_mask = tgt_padding_mask.to(embeddings.device)

        outputs = self.transformer_decoder(tgt=embeddings, 
                                           memory=memory, 
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_padding_mask)
        outputs = outputs.permute(1, 0, 2)  # [batch_size, seq_len, embed_size]
        outputs = self.fc_out(outputs)  # [batch_size, seq_len, vocab_size]
        return outputs

    def generate_square_subsequent_mask(self, sz, input):
        tgt_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (input == 0)
        tgt_padding_mask = tgt_padding_mask.to(dtype=torch.float32)
        return tgt_mask, tgt_padding_mask

class CaptionModel(nn.Module):
    """结合ImageEncoder和TransformerDecoder的完整模型"""
    def __init__(self, embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=True, encoder_type='ResNet'):
        super(CaptionModel, self).__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'ResNet':
            self.encoder = ImageEncoder(embed_size=embed_size, freeze=True, dropout=True)
        elif encoder_type == 'CLIP':
            self.encoder = ImageEncoderCLIP(embed_size=embed_size, freeze=True, dropout=True)
        else:
            raise ValueError("Invalid encoder type. Please choose from ['ResNet', 'CLIP']")
        
        self.decoder = TransformerDecoder(embed_size, vocab_size, num_layers, nhead, dim_feedforward, dropout=0.1 if dropout else 0.0, encoder_type=self.encoder_type)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        

    def forward(self, images, captions):
        """
        前向传播
        Args:
            images: Tensor, shape [batch_size, 3, H, W]
            captions: Tensor, shape [batch_size, seq_len]
        Returns:
            outputs: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        features = self.encoder(images)  # [batch_size, seq_len=49, embed_size]
        outputs = self.decoder(captions, features)  # [batch_size, seq_len, vocab_size]
        return outputs

    def captionBatch(self, images, vocabulary, max_length=50):
        """
        生成批量图片的描述
        Args:
            images: Tensor, shape [batch_size, 3, H, W]
            vocabulary: Vocabulary 对象
            max_length: 最大生成长度
        Returns:
            results: List[List[str]], shape [batch_size]
        """
        device = images.device
        batch_size = images.size(0)
        results = [[] for _ in range(batch_size)]

        with torch.no_grad():
            features = self.encoder(images)  # [batch_size, seq_len=49, embed_size]

        # 初始化输入为 <SOS>
        inputs = torch.tensor([vocabulary.stoi["<SOS>"]] * batch_size, dtype=torch.long).to(device)  # [batch_size]
        inputs = inputs.unsqueeze(1)  # [batch_size, 1]

        for _ in range(max_length):
            outputs = self.decoder(inputs, features)  # [batch_size, seq_len, vocab_size]
            outputs = outputs[:, -1, :]  # [batch_size, vocab_size]
            _, predicted = outputs.max(1)  # [batch_size]

            predicted_words = [vocabulary.itos[p.item()] for p in predicted]
            for i, word in enumerate(predicted_words):
                if word != "<EOS>":
                    results[i].append(word)
            inputs = torch.cat([inputs, predicted.unsqueeze(1)], dim=1)  # [batch_size, seq_len+1]

        return results


    def captionImage(self, image, vocabulary, max_length=50):
        """
        生成单张图片的描述
        Args:
            image: Tensor, shape [3, H, W]
            vocabulary: Vocabulary 对象
            max_length: 最大生成长度
        Returns:
            result: List[str]
        """
        device = image.device
        result = []
        with torch.no_grad():
            features = self.encoder(image)  # [1, seq_len=49, embed_size]

        # 初始化输入为 <SOS>
        inputs = torch.tensor([vocabulary.stoi["<SOS>"]], dtype=torch.long).to(device)  # [1]
        inputs = inputs.unsqueeze(0)  # [1, 1]

        for _ in range(max_length):
            outputs = self.decoder(inputs, features)  # [1, seq_len, vocab_size]
            outputs = outputs[:, -1, :]  # [1, vocab_size]
            _, predicted = outputs.max(1)  # [1]

            predicted_word = vocabulary.itos[predicted.item()]
            if predicted_word == "<EOS>":
                break
            result.append(predicted_word)
            inputs = torch.cat([inputs, predicted.unsqueeze(0)], dim=1)  # [1, seq_len+1]

        return result