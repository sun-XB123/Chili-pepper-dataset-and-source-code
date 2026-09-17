import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base
from transformers import AutoTokenizer, AutoModel

class MultiModalDataset(Dataset):
    def __init__(self, txt_path, img_dir, tokenizer, img_transform):
        self.samples = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                img_name, text = line.split("|", 1)
                img_path = os.path.join(img_dir, img_name)
                if not os.path.exists(img_path):
                    continue
                self.samples.append({"img_path": img_path, "text": text})
        self.tokenizer = tokenizer
        self.img_transform = img_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        img = self.img_transform(img)
        encoding = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        return img, input_ids, attn_mask

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = convnext_base(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(1024, 256)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean([-2, -1])
        x = self.fc(x)
        return F.normalize(x, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("BAAI/bge-large-zh")
        self.fc = nn.Linear(1024, 256)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        x = self.fc(pooled)
        return F.normalize(x, dim=-1)

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.txt_encoder = TextEncoder()

    def forward(self, imgs, input_ids, attn_masks):
        img_vecs = self.img_encoder(imgs)
        txt_vecs = self.txt_encoder(input_ids, attn_masks)
        return img_vecs, txt_vecs

def contrastive_loss(image_embeds, text_embeds, temperature=0.07, margin=0.2):
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    pos_sim = logits[torch.arange(logits.size(0)), labels]
    neg_sim = logits.masked_fill(torch.eye(logits.size(0), device=logits.device).bool(), -1e9)
    hardest_neg = neg_sim.max(dim=1)[0]
    margin_loss = F.relu(margin - pos_sim + hardest_neg).mean()
    return loss + 0.5 * margin_loss

def train(model, dataloader, optimizer, epochs=20, save_path="best_model.pth", temperature=0.07):
    device = next(model.parameters()).device
    best_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        total_loss, avg_pos_sim, avg_neg_sim, count = 0, 0, 0, 0
        for imgs, input_ids, attn_masks in dataloader:
            imgs, input_ids, attn_masks = imgs.to(device), input_ids.to(device), attn_masks.to(device)
            optimizer.zero_grad()
            img_vecs, txt_vecs = model(imgs, input_ids, attn_masks)
            loss = contrastive_loss(img_vecs, txt_vecs, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sim_matrix = F.cosine_similarity(img_vecs.unsqueeze(1), txt_vecs.unsqueeze(0), dim=-1)
            avg_pos_sim += sim_matrix.diag().mean().item()
            mask = ~torch.eye(sim_matrix.size(0), dtype=bool, device=device)
            avg_neg_sim += sim_matrix[mask].mean().item()
            count += 1
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | PosSim: {avg_pos_sim/count:.4f} | NegSim: {avg_neg_sim/count:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model")
    print(f"Training finished. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
    txt_path = r"C:\Users\14100\Desktop\朝天椒.txt"
    img_dir = r"C:\Users\14100\Desktop\朝天椒"
    dataset = MultiModalDataset(txt_path, img_dir, tokenizer, img_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = MultimodalModel().to(device)
    optimizer = torch.optim.Adam([
        {"params": model.img_encoder.fc.parameters(), "lr":5e-4},
        {"params": model.img_encoder.feature_extractor.parameters(), "lr":1e-4},
        {"params": model.txt_encoder.fc.parameters(), "lr":2e-4},
        {"params": model.txt_encoder.encoder.parameters(), "lr":1e-5}
    ])
    train(model, dataloader, optimizer,
          epochs=20,
          save_path=r"C:\Users\14100\Desktop\best_model_convnext_bge_large.pth",
          temperature=0.07)
