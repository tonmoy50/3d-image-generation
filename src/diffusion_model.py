import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Diffusion parameters
T = 200
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)


# 2. Model
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x, t):
        t = t[:, None, None, None].float() / T
        t_embed = t.expand_as(x)
        x = torch.cat([x, t_embed], dim=1)
        return self.net(x[:, :1])


# 3. Forward diffusion q(x_t | x_0)
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])[
        :, None, None, None
    ]
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


# 4. Dataset
transform = transforms.Compose([transforms.ToTensor(), lambda x: (x * 2) - 1])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 5. Train
model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1):  # one epoch demo
    for x0, _ in tqdm(dataloader):
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise)
        pred_noise = model(xt, t)

        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch done — Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "simple_diffusion_model.pth")
print("✅ Model saved!")
