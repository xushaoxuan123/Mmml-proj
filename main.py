import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.models import CaptionModel
from src.dataset import FlickrDataset, split_train_val_dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import nltk
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

####################################################################
# Functions
####################################################################

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_model_bleu(model,val_loader,args, vocab):
    smoothing_fn = SmoothingFunction().method1
    references, hypotheses = [], []
    count = 0
    with torch.no_grad():
        eval_loader = tqdm(val_loader, desc="Evaluation", total=len(val_loader))
        for imgs, captions in eval_loader:
            imgs = imgs.to(args.device)
            captions = captions.to(args.device)

            generated_captions = model.captionBatch(imgs, vocab)
            for i in range(len(imgs)):
                ref = captions[i].tolist()
                if count == 0:
                    print(ref)
                    print([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                    print(generated_captions[i])
                    count+=1
                references.append([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                hypotheses.append(generated_captions[i])
    # print(references[0], hypotheses[0])
    avg_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
    print(f"Validation Set Average BLEU Score: {avg_bleu:.4f}")
def test_model_single(model, dataset, args):
    for image_name in ["1000268201_693b08cb0e.jpg", "1001773457_577c3a7d70.jpg"]:
        image_path = "/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k/mmml-proj/flickr8k/Images/" + image_name
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        # plt.imshow(img.permute(1,2,0))
        image_input = img.to(device=args.device).unsqueeze(0) # check here
        caption = model.captionImage(image=image_input, vocabulary=dataset.vocab)
        print(f"{image_name}: {' '.join(caption)}")

class Collate:
    def __init__(self, pad_value):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch], dim=0)  # [batch_size, C, H, W]
        captions = torch.stack([item[1] for item in batch], dim=1).permute(1, 0)  # [batch_size, max_length]
        return imgs, captions


def get_loader(root_dir="/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k/Images", 
               caption_path="/root/mmml-proj/flickr8k/captions.txt", 
               transform=None, 
               batch_size=48, 
               freq_threshold = 5, 
               num_workers=8, 
               shuffle=True, 
               pin_memory=True, 
               selecting_samples = 10000):
    dataset = FlickrDataset(root_dir=root_dir,caption_path=caption_path, transform=transform, freq_threshold = freq_threshold, selecting_samples = selecting_samples)
    pad_value = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        shuffle=shuffle, 
                        pin_memory=pin_memory,
                        collate_fn=Collate(pad_value), 
                        generator=torch.Generator(device='cpu'))
    return loader, dataset

def train_and_eval(args, model, 
                  train_loader, val_loader,
                  optimizer, scheduler, filename, vocab):
    """
    训练并评估模型。

    参数：
    - args: 包含训练参数的命名空间，例如 num_epochs 和 device。
    - model: 要训练的模型。
    - train_loader: 训练数据的 DataLoader。
    - val_loader: 验证数据的 DataLoader。
    - optimizer: 优化器。
    - scheduler: 学习率调度器。
    - filename: 用于保存模型检查点的文件名。
    - vocab: 词汇表对象，包含词到索引的映射（vocab.stoi）。
    """

    # 定义损失函数，忽略填充项的损失
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    best_val_loss = float('inf')
    BLEU = False
    for epoch in range(args.num_epochs):
        ########################
        #        训练阶段       #
        ########################
        model.train()
        train_loss = 0.0
        train_total = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}", total=len(train_loader))
        for imgs, captions in train_loader_tqdm:
            imgs = imgs.to(args.device)
            captions = captions.to(args.device)
            optimizer.zero_grad()
            outputs = model(imgs, captions[:, :-1])  # 输入为字幕的前一部分
            targets = captions[:, 1:]

            outputs = outputs.reshape(-1, outputs.size(2))  # 形状：[batch_size * (seq_len-1), vocab_size]
            targets = targets.reshape(-1)                   # 形状：[batch_size * (seq_len-1)]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)
            train_loader_tqdm.set_postfix(loss=train_loss / train_total, lr=optimizer.param_groups[0]['lr'])

        # 计算平均训练损失
        avg_train_loss = train_loss / train_total

        ########################
        #        验证阶段       #
        ########################
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_total = 0.0
        
        val_loader_tqdm = tqdm(val_loader, desc="Validation", total=len(val_loader))
        for imgs, captions in val_loader_tqdm:
            imgs = imgs.to(args.device)          # 将图像数据移动到设备
            captions = captions.to(args.device)  # 将字幕数据移动到设备
            outputs = model(imgs, captions[:, :-1])
            targets = captions[:, 1:]
            
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            val_total += targets.size(0)
            val_loader_tqdm.set_postfix(loss=val_loss / val_total)

        # 计算平均验证损失
        avg_val_loss = val_loss / val_total

        # 打印本轮训练和验证的损失
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        if BLEU:
            smoothing_fn = SmoothingFunction().method1
            references, hypotheses = [], []
            count = 0
            with torch.no_grad():
                eval_loader = tqdm(val_loader, desc="Evaluation", total=len(val_loader))
                for imgs, captions in eval_loader:
                    imgs = imgs.to(args.device)
                    captions = captions.to(args.device)

                    generated_captions = model.captionBatch(imgs, vocab)
                    for i in range(len(imgs)):
                        ref = captions[i].tolist()
                        if count == 0:
                            print(ref)
                            print([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                            print(generated_captions[i])
                            count+=1
                        references.append([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                        hypotheses.append(generated_captions[i])

            # 计算 BLEU 分数
            print(references[0], hypotheses[0])
            avg_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
            print(f"Validation Set Average BLEU Score: {avg_bleu:.4f}")
        # 如果当前验证损失优于最佳验证损失，则保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(checkpoint, filename)
            print(f"Validation loss decreased. Model saved to {filename}")

        # 调整学习率
        scheduler.step()


def save_checkpoint(state, filename):
    print("saving checkpoint!")
    torch.save(state, filename)

def load_checkpoint(name, model):
    print("loading checkpoint!")
    checkpoint = torch.load(name, map_location=model.device if hasattr(model, 'device') else 'cpu')
    model.load_state_dict(checkpoint["state_dict"])

if __name__ == '__main__':

    ####################################################################
    # Arguments
    ####################################################################

    parser = argparse.ArgumentParser(description='Flickr8k')
    parser.add_argument('-f', default='', type=str)

    parser.add_argument('--embed_size', type=int, default=512, metavar='N', help='embedding size (default: 256)')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='batch size (default: 24)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs (default: 50)')
    parser.add_argument('--when_decay', type=int, default=30, help='when to decay learning rate (default: 30)')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--num_layers', type=int, default=6, help='number of Transformer decoder layers (default: 6)')
    parser.add_argument('--nhead', type=int, default=8, help='number of heads in the Transformer decoder (default: 8)')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of the feedforward network in Transformer (default: 2048)')

    args = parser.parse_args()
    setup_seed(args.seed)

    ####################################################################
    # Load the dataset 
    ####################################################################

    transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]) 
            ]
        )

    # 加载训练和验证集
    path = "/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k"
    if not os.path.exists(path + "/train_captions.csv") or not os.path.exists(path + "/val_captions.csv") or not os.path.exists(path + "/all_captions.csv"):
        print('Spliting dataset')
        split_train_val_dataset()

    train_loader, train_set = get_loader(caption_path=path + "/train_captions.csv", transform=transform, freq_threshold=2, selecting_samples='all', batch_size=args.batch_size)
    val_loader, val_set = get_loader(caption_path=path + "/val_captions.csv", transform=transform, freq_threshold=2, selecting_samples='all', batch_size=args.batch_size, shuffle= False)
    # loader, dataset = get_loader(transform=transform, freq_threshold=4, selecting_samples='all', batch_size=args.batch_size)
    loader, dataset = get_loader(caption_path=path + "/all_captions.csv", transform=transform, freq_threshold=2, selecting_samples='all', batch_size=args.batch_size)

    args.vocab_size = len(dataset.vocab)
    # print(dataset.vocab.stoi)
    print("VocabSize:", args.vocab_size)
    print("train_set:", len(train_set))
    print("val_set:", len(val_set))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####################################################################
    # Training 
    ####################################################################

    model = CaptionModel(
        embed_size=args.embed_size,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=True,
        encoder_type='CLIP'
    ).to(device=args.device)
    print(f'Using Encoder: {model.encoder_type}, Decoder: TransformerDecoder')
    if args.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.when_decay, 0.1)
    filename = "caption_model_lr{}_decay{}_bsize{}.pth.tar".format(args.lr, args.when_decay, args.batch_size)
    train_and_eval(args, model, train_loader, val_loader, optimizer, scheduler, filename, dataset.vocab)

    ####################################################################
    # Test on single images
    ####################################################################
    
    test_model = CaptionModel(
        embed_size=args.embed_size,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=False,
        encoder_type='CLIP'
    ).to(device=args.device)

    load_checkpoint(filename, test_model)
    test_model_single(test_model, dataset, args)
    test_model_bleu(test_model, val_loader, args, dataset.vocab)

