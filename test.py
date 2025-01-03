import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.models import CaptionModel  # 修改为导入 CNN2Transformer
from src.dataset import FlickrDataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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
    for image_name in ["359082432_c1fd5aa2d6.jpg", "3702607829_2b8b3e65ab.jpg"]:
        image_path = "/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k/Images/" + image_name
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
               caption_path="/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k/captions.txt", 
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])  # 忽略填充项的损失
    best_loss = 1e5
    best_bleu = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        _loss = 0
        _total = 0

        flickr_loader = tqdm(train_loader, desc="Training Epoch {}".format(epoch), total=len(train_loader))
        for imgs, captions in flickr_loader:
            imgs = imgs.to(args.device)
            captions = captions.to(args.device)

            # 前向传播，captions 从<SOS>开始到最后一个词
            outputs = model(imgs, captions[:, :-1])  # [batch_size, seq_len-1, vocab_size]
            optimizer.zero_grad()

            # ground truth，从第一个词开始到<EOS>
            targets = captions[:, 1:]  # [batch_size, seq_len-1]

            # 计算损失
            outputs = outputs.reshape(-1, outputs.size(2))  # [(batch_size * (seq_len-1)), vocab_size]
            targets = targets.reshape(-1)  # [(batch_size * (seq_len-1))]

            loss = criterion(outputs, targets)
            loss.backward()

            _loss += loss.item() * targets.size(0)
            _total += targets.size(0)
            optimizer.step()
            flickr_loader.set_postfix(loss=_loss / _total, lr=optimizer.param_groups[0]['lr'])
        scheduler.step()
        _loss /= _total

        model.eval()
        smoothing_fn = SmoothingFunction().method1
        references, hypotheses = [], []

        with torch.no_grad():
            eval_loader = tqdm(val_loader, desc="Evaluation", total=len(val_loader))
            for imgs, captions in eval_loader:
                imgs = imgs.to(args.device)
                captions = captions.to(args.device)

                generated_captions = model.captionBatch(imgs, vocab)
                for i in range(len(imgs)):
                    ref = captions[i].tolist()
                    references.append([[vocab.itos[idx] for idx in ref if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}]])
                    hypotheses.append(generated_captions[i])

        # 计算 BLEU 分数
        avg_bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing_fn)
        print(f"Validation Set Average BLEU Score: {avg_bleu:.4f}")

        print(f"Loss for epoch {epoch}: {_loss}")
        if _loss < best_loss:
            best_loss = _loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_checkpoint(checkpoint, filename)


def save_checkpoint(state, filename):
    print("saving checkpoint!")
    torch.save(state, filename)

def load_checkpoint(name, model):
    checkpoint = torch.load(name, map_location=model.device if hasattr(model, 'device') else 'cpu', weights_only=True)
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
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='dimension of the feedforward network in Transformer (default: 2048)')

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

    loader, dataset = get_loader(caption_path="/home/shaoxuan_xu/Temp/test/mmml-proj/flickr8k/val_captions.csv", transform=transform, freq_threshold=4, selecting_samples='all', batch_size=args.batch_size)

    args.vocab_size = len(dataset.vocab)
    # print(dataset.vocab.stoi)
    print("VocabSize:", args.vocab_size)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filename = "caption_model_lr{}_decay{}_bsize{}.pth.tar".format(args.lr, args.when_decay, args.batch_size)

    ####################################################################
    # Test on single images
    ####################################################################
    
    # 实例化新的模型用于测试
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
    # test_model_single(test_model, dataset, args)
    test_model_bleu(test_model, loader, args, dataset.vocab)
