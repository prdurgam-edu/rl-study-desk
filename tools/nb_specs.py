"""Guided Google Colab notebooks for the parts that need PyTorch or Gymnasium."""
from labs_warmup import S

COMMON_HOW = ("**How to use this notebook:** run the cells from top to bottom (Shift+Enter). "
              "In each step, replace every `___` in the code cell, run it, then run the **Check** cell under it. "
              "Stuck? Expand the hidden **Solution** cell (click the ▸ arrow). "
              "For speed, turn on a GPU first: *Runtime → Change runtime type → T4 GPU*.")

NOTEBOOKS = [
dict(id='NB_DL10', file='DL10_cnn_cifar10.ipynb', title='Train a CNN on CIFAR-10', intro=
"You built convolution by hand in the browser lab. Now train a real convolutional network on CIFAR-10 (60,000 small colour photos in 10 classes) with PyTorch, then look inside it. Aim: over 75% test accuracy in about 10 minutes on a GPU.",
setup='''
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
TEST = os.environ.get("NB_TEST") == "1"          # used only by the automatic tests; ignore it
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tf = T.Compose([T.ToTensor(), T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])
if TEST:
    train_set = torchvision.datasets.FakeData(512, (3, 32, 32), 10, transform=T.ToTensor())
    test_set = torchvision.datasets.FakeData(256, (3, 32, 32), 10, transform=T.ToTensor())
else:
    train_set = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("device:", device, "| training images:", len(train_set))
''',
steps=[
S('Define the network',
"Three conv blocks (conv 3x3, ReLU, 2x2 max-pool) take the image from 32x32 to 16, 8, then 4 pixels per side, with 32, 64, 128 channels. Then flatten and two linear layers. What's the size of the flattened vector?",
'''
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1), nn.Conv2d(64, 128, 3, padding=1)])
        self.fc1 = nn.Linear(___, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))

model = SmallCNN().to(device)
print(model)
''','''
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1), nn.Conv2d(64, 128, 3, padding=1)])
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(x)), 2)
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))

model = SmallCNN().to(device)
print(model)
''','''
out = model(torch.randn(2, 3, 32, 32, device=device))
assert out.shape == (2, 10), "128 channels x 4 x 4 pixels = 2048 inputs to fc1"
print("Correct!")
''', "128 channels of 4x4: 128 * 4 * 4"),
S('The training loop',
"Every PyTorch training loop has the same five lines: forward, loss, zero the old gradients, backward, step. Fill in the last two.",
'''
EPOCHS = 1 if TEST else 10
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        ___
        ___
        losses.append(loss.item())
    print(f"epoch {epoch + 1}: loss {np.mean(losses[-100:]):.3f}")
plt.plot(losses); plt.xlabel("batch"); plt.ylabel("loss")
''','''
EPOCHS = 1 if TEST else 10
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1}: loss {np.mean(losses[-100:]):.3f}")
plt.plot(losses); plt.xlabel("batch"); plt.ylabel("loss")
''','''
assert len(losses) > 0 and np.isfinite(losses).all()
if not TEST:
    assert np.mean(losses[-100:]) < np.mean(losses[:100]), "Loss should fall during training"
print("Correct!")
''', "loss.backward() then opt.step()"),
S('Test accuracy',
"Switch to evaluation mode, turn off gradient tracking, and count correct predictions.",
'''
def accuracy(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += ___
            total += len(y)
    return correct / total

test_acc = accuracy(model, test_loader)
print(f"test accuracy: {test_acc:.1%}")
''','''
def accuracy(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

test_acc = accuracy(model, test_loader)
print(f"test accuracy: {test_acc:.1%}")
''','''
assert 0 <= test_acc <= 1
if not TEST:
    assert test_acc > 0.75, f"Aim for over 75%; you have {test_acc:.1%}. Train a few more epochs."
print("Correct!")
''', "(pred == y).sum().item()"),
S('Look inside: filters and an occlusion map',
"First-layer filters often look like colour blobs and edge detectors. Then slide a grey square over a test image and record how much the network's confidence in the true class drops: the bright regions are what it looks at.",
'''
w = model.convs[0].weight.detach().cpu()
w = (w - w.min()) / (w.max() - w.min())
fig, axes = plt.subplots(4, 8, figsize=(8, 4))
for ax, f in zip(axes.ravel(), w):
    ax.imshow(f.permute(1, 2, 0)); ax.axis("off")
plt.show()

x, y = test_set[0]
model.eval()
with torch.no_grad():
    base = F.softmax(model(x[None].to(device)), dim=1)[0, y].item()
    heat = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            occluded = x.clone()
            occluded[:, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = 0
            p = F.softmax(model(occluded[None].to(device)), dim=1)[0, y].item()
            heat[i, j] = ___          # how much confidence dropped
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow((x.permute(1, 2, 0) * torch.tensor([0.25, 0.24, 0.26]) + torch.tensor([0.49, 0.48, 0.45])).clamp(0, 1)); axes[0].set_title(classes[y])
axes[1].imshow(heat, cmap="hot"); axes[1].set_title("confidence drop")
''','''
w = model.convs[0].weight.detach().cpu()
w = (w - w.min()) / (w.max() - w.min())
fig, axes = plt.subplots(4, 8, figsize=(8, 4))
for ax, f in zip(axes.ravel(), w):
    ax.imshow(f.permute(1, 2, 0)); ax.axis("off")
plt.show()

x, y = test_set[0]
model.eval()
with torch.no_grad():
    base = F.softmax(model(x[None].to(device)), dim=1)[0, y].item()
    heat = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            occluded = x.clone()
            occluded[:, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = 0
            p = F.softmax(model(occluded[None].to(device)), dim=1)[0, y].item()
            heat[i, j] = base - p          # how much confidence dropped
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow((x.permute(1, 2, 0) * torch.tensor([0.25, 0.24, 0.26]) + torch.tensor([0.49, 0.48, 0.45])).clamp(0, 1)); axes[0].set_title(classes[y])
axes[1].imshow(heat, cmap="hot"); axes[1].set_title("confidence drop")
''','''
assert heat.shape == (8, 8) and np.isfinite(heat).all() and heat.max() <= 1
print("Correct! Save this notebook to your GitHub repo (File -> Save a copy in GitHub).")
''', "base - p"),
]),

dict(id='NB_DL11', file='DL11_char_lstm.ipynb', title='A character-level LSTM that writes Shakespeare', intro=
"Train an LSTM to predict the next character of Shakespeare's plays, then let it write. In the browser lab you saw why LSTMs keep gradients alive; here you'll see what that buys you. About 10 minutes on a GPU.",
setup='''
import os, urllib.request, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
TEST = os.environ.get("NB_TEST") == "1"          # used only by the automatic tests; ignore it
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
if TEST:
    text = "To be, or not to be, that is the question.\\nWhether tis nobler in the mind to suffer\\n" * 200
else:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(len(text), "characters,", V, "distinct")
''',
steps=[
S('Make training batches',
"Pick random starting points. The input is a chunk of L characters; the target is the same chunk shifted one character to the right (the next character at every position).",
'''
def get_batch(L=100, B=64):
    starts = torch.randint(0, len(data) - L - 1, (B,))
    x = torch.stack([data[s:s + L] for s in starts])
    y = torch.stack([data[___] for s in starts])
    return x.to(device), y.to(device)

x, y = get_batch()
print(x.shape, y.shape)
''','''
def get_batch(L=100, B=64):
    starts = torch.randint(0, len(data) - L - 1, (B,))
    x = torch.stack([data[s:s + L] for s in starts])
    y = torch.stack([data[s + 1:s + L + 1] for s in starts])
    return x.to(device), y.to(device)

x, y = get_batch()
print(x.shape, y.shape)
''','''
x, y = get_batch(L=20, B=4)
assert x.shape == y.shape == (4, 20) and torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted left by one"
print("Correct!")
''', "s + 1:s + L + 1"),
S('The model',
"Embedding (each character becomes a 64-number vector) → LSTM with 256 hidden units → linear layer back to one score per character.",
'''
class CharLSTM(nn.Module):
    def __init__(self, V, emb=64, hidden=256, cell=nn.LSTM):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.rnn = cell(___, ___, batch_first=True)
        self.out = nn.Linear(hidden, V)
    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state

model = CharLSTM(V).to(device)
logits, _ = model(x)
print(logits.shape)
''','''
class CharLSTM(nn.Module):
    def __init__(self, V, emb=64, hidden=256, cell=nn.LSTM):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.rnn = cell(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, V)
    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state

model = CharLSTM(V).to(device)
logits, _ = model(x)
print(logits.shape)
''','''
assert model(x)[0].shape == (x.shape[0], x.shape[1], V)
print("Correct!")
''', "Input size emb, hidden size hidden"),
S('Train with gradient clipping',
"Recurrent networks occasionally produce huge gradients. Clipping the gradient norm to 1.0 before each step keeps training stable.",
'''
STEPS = 30 if TEST else 3000
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for step in range(STEPS):
    x, y = get_batch()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(___, 1.0)
    opt.step(); losses.append(loss.item())
    if step % 500 == 0:
        print(step, round(loss.item(), 3))
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
''','''
STEPS = 30 if TEST else 3000
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
losses = []
for step in range(STEPS):
    x, y = get_batch()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); losses.append(loss.item())
    if step % 500 == 0:
        print(step, round(loss.item(), 3))
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
''','''
assert np.mean(losses[-10:]) < np.mean(losses[:10]), "Loss should fall"
if not TEST:
    assert np.mean(losses[-100:]) < 1.7, "Aim for a loss below 1.7"
print("Correct!")
''', "Clip the model's parameters: model.parameters()"),
S('Let it write',
"Feed a prompt, then repeatedly sample the next character from softmax(logits / temperature). Low temperature is safe and repetitive; high temperature is creative and messy. Try 0.5, 0.8 and 1.2.",
'''
def generate(prompt="ROMEO:", n=400, temperature=0.8):
    model.eval()
    x = torch.tensor([[stoi[c] for c in prompt if c in stoi] or [0]], dtype=torch.long, device=device)
    out, state = prompt, None
    with torch.no_grad():
        logits, state = model(x, state)
        for _ in range(n):
            probs = F.softmax(___, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out += itos[nxt.item()]
            logits, state = model(nxt.view(1, 1), state)
    model.train()
    return out

print(generate())
''','''
def generate(prompt="ROMEO:", n=400, temperature=0.8):
    model.eval()
    x = torch.tensor([[stoi[c] for c in prompt if c in stoi] or [0]], dtype=torch.long, device=device)
    out, state = prompt, None
    with torch.no_grad():
        logits, state = model(x, state)
        for _ in range(n):
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out += itos[nxt.item()]
            logits, state = model(nxt.view(1, 1), state)
    model.train()
    return out

print(generate())
''','''
g = generate("To", n=50)
assert isinstance(g, str) and len(g) >= 50
print("Correct! Try swapping nn.LSTM for nn.RNN (cell=nn.RNN) and compare the loss.")
''', "The last position's scores, scaled: logits[0, -1] / temperature"),
]),

dict(id='NB_P08', file='P08_resnet_and_tiny_gpt.ipynb', title='Project: fine-tune ResNet and build a tiny GPT', intro=
"Two modern workhorses. Part A: take ResNet-18, trained on a million ImageNet photos, and adapt it to CIFAR-10 (transfer learning). Part B: build a small GPT (a decoder-only transformer) from the attention you wrote in DL12, and train it on Shakespeare. Use a GPU runtime; allow about 30 minutes.",
setup='''
import os, urllib.request, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
TEST = os.environ.get("NB_TEST") == "1"          # used only by the automatic tests; ignore it
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tf = T.Compose([T.Resize(96), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
if TEST:
    train_set = torchvision.datasets.FakeData(64, (3, 32, 32), 10, transform=tf)
    test_set = torchvision.datasets.FakeData(64, (3, 32, 32), 10, transform=tf)
else:
    full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
    train_set = torch.utils.data.Subset(full, range(10000))
    test_set = torch.utils.data.Subset(torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf), range(2000))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)

def accuracy(model, loader):
    model.eval(); c = n = 0
    with torch.no_grad():
        for x, y in loader:
            c += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return c / n

def train(model, params, epochs, lr):
    opt = torch.optim.Adam(params, lr=lr); model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
''',
steps=[
S('Part A: swap the head',
"ResNet-18's last layer (`fc`) predicts 1,000 ImageNet classes. Replace it with a fresh linear layer for 10 classes, keeping the same number of inputs.",
'''
net = torchvision.models.resnet18(weights=None if TEST else "IMAGENET1K_V1")
net.fc = nn.Linear(___, 10)
net = net.to(device)
print(net.fc)
''','''
net = torchvision.models.resnet18(weights=None if TEST else "IMAGENET1K_V1")
net.fc = nn.Linear(net.fc.in_features, 10)
net = net.to(device)
print(net.fc)
''','''
assert net.fc.out_features == 10 and net.fc.in_features == 512
print("Correct!")
''', "net.fc.in_features (it's 512)"),
S('Train only the head, then everything',
"First freeze the pretrained backbone and train only the new head (fast, safe). Then unfreeze everything and fine-tune with a small learning rate.",
'''
for p in net.parameters():
    p.requires_grad = False
for p in net.fc.parameters():
    p.requires_grad = ___
train(net, net.fc.parameters(), epochs=1 if TEST else 2, lr=1e-3)
acc_head = accuracy(net, test_loader); print("head only:", acc_head)

for p in net.parameters():
    p.requires_grad = True
train(net, net.parameters(), epochs=1 if TEST else 2, lr=1e-4)
acc_full = accuracy(net, test_loader); print("fine-tuned:", acc_full)
''','''
for p in net.parameters():
    p.requires_grad = False
for p in net.fc.parameters():
    p.requires_grad = True
train(net, net.fc.parameters(), epochs=1 if TEST else 2, lr=1e-3)
acc_head = accuracy(net, test_loader); print("head only:", acc_head)

for p in net.parameters():
    p.requires_grad = True
train(net, net.parameters(), epochs=1 if TEST else 2, lr=1e-4)
acc_full = accuracy(net, test_loader); print("fine-tuned:", acc_full)
''','''
assert 0 <= acc_head <= 1 and 0 <= acc_full <= 1
if not TEST:
    assert acc_full > 0.85 and acc_full > acc_head, "Fine-tuning everything should reach over 85%"
print("Correct! Compare: your CNN from scratch needed 50,000 images to reach ~75%.")
''', "The head must be trainable: True"),
S('Part B: causal self-attention',
"The heart of GPT. Project the input to queries, keys and values, compute softmax(QK^T / sqrt(d)) with a causal mask so no position sees the future, and mix the values.",
'''
if TEST:
    text = "To be, or not to be, that is the question.\\n" * 300
else:
    text = urllib.request.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").read().decode()
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}; itos = dict(enumerate(chars))
data = torch.tensor([stoi[c] for c in text])
CTX, D, HEADS = 128, 192, 6

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).bool())
    def forward(self, x):
        B, Tn, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = [t.view(B, Tn, HEADS, D // HEADS).transpose(1, 2) for t in (q, k, v)]
        att = q @ k.transpose(-2, -1) / math.sqrt(D // HEADS)
        att = att.masked_fill(~self.mask[:Tn, :Tn], ___)
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, Tn, D)
        return self.proj(y)
''','''
if TEST:
    text = "To be, or not to be, that is the question.\\n" * 300
else:
    text = urllib.request.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").read().decode()
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}; itos = dict(enumerate(chars))
data = torch.tensor([stoi[c] for c in text])
CTX, D, HEADS = 128, 192, 6

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).bool())
    def forward(self, x):
        B, Tn, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = [t.view(B, Tn, HEADS, D // HEADS).transpose(1, 2) for t in (q, k, v)]
        att = q @ k.transpose(-2, -1) / math.sqrt(D // HEADS)
        att = att.masked_fill(~self.mask[:Tn, :Tn], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, Tn, D)
        return self.proj(y)
''','''
attn = CausalSelfAttention()
a = torch.randn(1, 10, D); b = a.clone(); b[0, 7:] = torch.randn(3, D)
assert torch.allclose(attn(a)[0, :7], attn(b)[0, :7], atol=1e-5), "Changing future tokens must not change earlier outputs"
print("Correct!")
''', 'Masked positions get minus infinity before the softmax: float("-inf")'),
S('Assemble GPT and train it',
"A transformer block is attention and an MLP, each wrapped as x = x + layer(norm(x)) (a residual connection). Stack four blocks between a token+position embedding and an output layer. Fill in the residual for the MLP.",
'''
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = ___
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(4)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        return self.head(self.ln(self.blocks(x)))

gpt = TinyGPT().to(device)
opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
losses = []
for step in range(20 if TEST else 2500):
    s = torch.randint(0, len(data) - CTX - 1, (32,))
    x = torch.stack([data[i:i + CTX] for i in s]).to(device); y = torch.stack([data[i + 1:i + CTX + 1] for i in s]).to(device)
    loss = F.cross_entropy(gpt(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if step % 500 == 0: print(step, round(loss.item(), 3))

idx = torch.tensor([[stoi[c] for c in "ROMEO:" if c in stoi] or [0]], device=device)
gpt.eval()
with torch.no_grad():
    for _ in range(300):
        p = F.softmax(gpt(idx[:, -CTX:])[0, -1] / 0.8, dim=-1)
        idx = torch.cat([idx, torch.multinomial(p, 1)[None]], dim=1)
print("".join(itos[i] for i in idx[0].tolist()))
''','''
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(4)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        return self.head(self.ln(self.blocks(x)))

gpt = TinyGPT().to(device)
opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
losses = []
for step in range(20 if TEST else 2500):
    s = torch.randint(0, len(data) - CTX - 1, (32,))
    x = torch.stack([data[i:i + CTX] for i in s]).to(device); y = torch.stack([data[i + 1:i + CTX + 1] for i in s]).to(device)
    loss = F.cross_entropy(gpt(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if step % 500 == 0: print(step, round(loss.item(), 3))

idx = torch.tensor([[stoi[c] for c in "ROMEO:" if c in stoi] or [0]], device=device)
gpt.eval()
with torch.no_grad():
    for _ in range(300):
        p = F.softmax(gpt(idx[:, -CTX:])[0, -1] / 0.8, dim=-1)
        idx = torch.cat([idx, torch.multinomial(p, 1)[None]], dim=1)
print("".join(itos[i] for i in idx[0].tolist()))
''','''
assert np.mean(losses[-5:]) < np.mean(losses[:5]), "Loss should fall"
if not TEST:
    assert np.mean(losses[-100:]) < 1.8, "Aim for a loss below 1.8"
print("Correct! Same architecture as ChatGPT's ancestors, just much smaller.")
''', "x + self.mlp(self.ln2(x))"),
]),

dict(id='NB_P13', file='P13_dqn_cartpole.ipynb', title='Project: DQN from scratch on CartPole', intro=
"Deep Q-Networks (Mnih et al., 2015) put a neural network in place of the Q-table, made stable by two tricks: a replay buffer and a target network. Build it, solve CartPole-v1 (average return 475+ out of 500), then remove each trick and see what breaks. Training takes about 10-15 minutes (a GPU doesn't help much here; CartPole is tiny).",
setup='''
!pip -q install gymnasium
import os, random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
TEST = os.environ.get("NB_TEST") == "1"          # used only by the automatic tests; ignore it
torch.manual_seed(0); random.seed(0); np.random.seed(0)
env = gym.make("CartPole-v1")
print(env.observation_space, env.action_space)
''',
steps=[
S('The Q-network',
"A small MLP maps the 4 state numbers to one Q-value per action.",
'''
def make_qnet():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, ___))

qnet = make_qnet()
print(qnet(torch.zeros(1, 4)))
''','''
def make_qnet():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

qnet = make_qnet()
print(qnet(torch.zeros(1, 4)))
''','''
assert make_qnet()(torch.zeros(3, 4)).shape == (3, 2), "One output per action: 2"
print("Correct!")
''', "CartPole has 2 actions"),
S('A replay buffer',
"Store transitions (s, a, r, s', done) and train on random mini-batches of them. Random sampling breaks the strong correlation between consecutive steps.",
'''
class Replay:
    def __init__(self, cap=50000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, *transition):
        self.buf.append(transition)
    def sample(self, n):
        batch = ___
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32), torch.tensor(a), torch.tensor(r, dtype=torch.float32),
                torch.tensor(s2, dtype=torch.float32), torch.tensor(d, dtype=torch.float32))
    def __len__(self):
        return len(self.buf)
''','''
class Replay:
    def __init__(self, cap=50000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, *transition):
        self.buf.append(transition)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.tensor(s, dtype=torch.float32), torch.tensor(a), torch.tensor(r, dtype=torch.float32),
                torch.tensor(s2, dtype=torch.float32), torch.tensor(d, dtype=torch.float32))
    def __len__(self):
        return len(self.buf)
''','''
rb = Replay()
for i in range(100):
    rb.push(np.zeros(4), i % 2, 1.0, np.ones(4), False)
s, a, r, s2, d = rb.sample(32)
assert s.shape == (32, 4) and a.shape == (32,) and len(set(a.tolist())) == 2
print("Correct!")
''', "random.sample(self.buf, n)"),
S('The DQN loss',
"The target for Q(s, a) is r + gamma max_a' Q_target(s', a'), with no future value when the episode ended. Compute it with no gradient, from the separate target network.",
'''
gamma = 0.99
def dqn_loss(qnet, target_net, batch):
    s, a, r, s2, d = batch
    q = qnet(s).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        target = ___
    return F.smooth_l1_loss(q, target)
''','''
gamma = 0.99
def dqn_loss(qnet, target_net, batch):
    s, a, r, s2, d = batch
    q = qnet(s).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        target = r + gamma * (1 - d) * target_net(s2).max(dim=1).values
    return F.smooth_l1_loss(q, target)
''','''
tn = make_qnet()
batch = (torch.zeros(4, 4), torch.tensor([0, 1, 0, 1]), torch.ones(4), torch.zeros(4, 4), torch.tensor([0., 0., 1., 1.]))
q = qnet(batch[0]).gather(1, batch[1].view(-1, 1)).squeeze(1)
want = 1 + gamma * torch.tensor([1., 1., 0., 0.]) * tn(batch[3]).max(1).values
assert torch.allclose(dqn_loss(qnet, tn, batch), F.smooth_l1_loss(q, want.detach())), "target = r + gamma * (1 - done) * max over next Q-values"
print("Correct!")
''', "r + gamma * (1 - d) * target_net(s2).max(dim=1).values"),
S('Train it',
"Epsilon decays from 1 to 0.05 over the first 3,000 steps. After 500 warm-up steps, do one gradient step per environment step; copy the online network into the target network every 250 steps. Watch the curve: DQN often reaches 500, then wobbles or even collapses for a while. That instability is real and famous, and it is why the later algorithms (Double DQN, PPO) exist.",
'''
def train_dqn(episodes=300, use_replay=True, target_every=250, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    qnet, target_net = make_qnet(), make_qnet()
    target_net.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=1e-3)
    rb, steps, returns = Replay(), 0, []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done, total = False, 0.0
        while not done:
            eps = max(0.05, 1 - steps / 3000)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(qnet(torch.tensor(s, dtype=torch.float32)).argmax())
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            rb.push(s, a, r, s2, float(term)); s, total, steps = s2, total + r, steps + 1
            if len(rb) >= 500:
                batch = rb.sample(64) if use_replay else Replay.sample(type("B", (), {"buf": list(rb.buf)[-64:]})(), 64)
                loss = dqn_loss(qnet, target_net, batch)
                opt.zero_grad(); loss.backward(); opt.step()
            if steps % target_every == 0:
                target_net.load_state_dict(___)
        returns.append(total)
        if ep % 25 == 0:
            print(ep, np.mean(returns[-25:]))
    return returns

returns = train_dqn(episodes=5 if TEST else 300)
plt.plot(returns); plt.plot(np.convolve(returns, np.ones(20) / 20, mode="valid")); plt.xlabel("episode"); plt.ylabel("return")
''','''
def train_dqn(episodes=300, use_replay=True, target_every=250, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    qnet, target_net = make_qnet(), make_qnet()
    target_net.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=1e-3)
    rb, steps, returns = Replay(), 0, []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done, total = False, 0.0
        while not done:
            eps = max(0.05, 1 - steps / 3000)
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(qnet(torch.tensor(s, dtype=torch.float32)).argmax())
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            rb.push(s, a, r, s2, float(term)); s, total, steps = s2, total + r, steps + 1
            if len(rb) >= 500:
                batch = rb.sample(64) if use_replay else Replay.sample(type("B", (), {"buf": list(rb.buf)[-64:]})(), 64)
                loss = dqn_loss(qnet, target_net, batch)
                opt.zero_grad(); loss.backward(); opt.step()
            if steps % target_every == 0:
                target_net.load_state_dict(qnet.state_dict())
        returns.append(total)
        if ep % 25 == 0:
            print(ep, np.mean(returns[-25:]))
    return returns

returns = train_dqn(episodes=5 if TEST else 300)
plt.plot(returns); plt.plot(np.convolve(returns, np.ones(20) / 20, mode="valid")); plt.xlabel("episode"); plt.ylabel("return")
''','''
assert len(returns) > 0
if not TEST:
    assert max(np.convolve(returns, np.ones(20) / 20, mode="valid")) > 400, "A 20-episode average should get above 400 at some point"
print("Correct!")
''', "Copy the online network's weights: qnet.state_dict()"),
dict(free=True, **S('Ablation: take the tricks away',
"Run three more agents: no replay buffer (train only on the latest transitions), no target network (copy every step), and neither. Compare learning curves. This is the experiment from the DQN paper; expect the stripped-down agents to be slower and much less stable. (Takes 10-20 minutes.)",
'''
variants = {
    "full DQN": dict(),
    "no replay": dict(use_replay=False),
    "no target net": dict(target_every=1),
    "neither": dict(use_replay=False, target_every=1),
}
curves = {name: train_dqn(episodes=3 if TEST else 300, **kw) for name, kw in variants.items()}
for name, r in curves.items():
    plt.plot(np.convolve(r, np.ones(min(20, len(r))) / min(20, len(r)), mode="valid"), label=name)
plt.xlabel("episode"); plt.ylabel("return (20-episode average)"); plt.legend()
''','''
variants = {
    "full DQN": dict(),
    "no replay": dict(use_replay=False),
    "no target net": dict(target_every=1),
    "neither": dict(use_replay=False, target_every=1),
}
curves = {name: train_dqn(episodes=3 if TEST else 300, **kw) for name, kw in variants.items()}
for name, r in curves.items():
    plt.plot(np.convolve(r, np.ones(min(20, len(r))) / min(20, len(r)), mode="valid"), label=name)
plt.xlabel("episode"); plt.ylabel("return (20-episode average)"); plt.legend()
''','''
assert set(curves) == set(variants)
print("Done! Write two sentences in your repo on what each trick buys you.")
''', "Nothing to fill in: run it and compare the curves.")),
]),
]
