from models import (
    Generator,
    Discriminator,
)

import os

import matplotlib
matplotlib.use("Agg")

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import torchutils

from .dataset_utils import ShapeNet
from .render_utils import render_orthographic_faces

def get_argparser():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Data root directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["shapenet"],
        help="Dataset to train on. Must be one of: `shapenet`.",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="ShapeNet category to train on. Must be one of: `airplane`.",
    )
    parser.add_argument(
        "--voxel_resolution",
        type=int,
        default=64,
        help="Voxel resolution (default: 64).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train for (default: 20)",
    )
    parser.add_argument(
        "--generator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the generator.",
    )
    parser.add_argument(
        "--discriminator_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the discriminator.",
    )
    parser.add_argument(
        "--use_spectral_norm",
        action="store_true",
        help="Whether to use spectral normalization (default: False).",
    )
    parser.add_argument(
        "--use_hinge_loss",
        action="store_true",
        help="Whether to use hinge loss (default: False).",
    )
    parser.add_argument(
        "--use_instance_noise",
        action="store_true",
        help="Whether to use instance noise (default: False).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=200,
        help="Latent space dimensionality (default: 200).",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=0.0025,
        help="Learning rate for the generator (default: 0.0025).",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=0.00001,
        help="Learning rate for the discriminator (default: 0.00001).",
    )
    parser.add_argument(
        "--generator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--generator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--discriminator_adam_beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5).",
    )
    parser.add_argument(
        "--discriminator_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=20,
        help="Interval (in epochs) at which to save model checkpoints (default: 20).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save intermediate output (default: ./output).",
    )
    parser.add_argument(
        "--save_output_interval",
        type=int,
        default=5,
        help="Number of epochs between saving intermediate output (default: 5).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser

def main():
    args = get_argparser().parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Training on {args.dataset}")
    if args.dataset == "shapenet":
        dataset = ShapeNet(
            root=args.root,
            category=args.category,
            voxel_resolution=args.voxel_resolution,
            image_set="train",
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            drop_last=True,
        )

    G = Generator(
        in_channels=args.generator_in_channels,
        latent_dim=args.latent_dim,
    )
    D = Discriminator(
        in_channels=args.discriminator_in_channels,
        use_spectral_norm=args.use_spectral_norm,
    )
    G = G.to(device)
    D = D.to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=args.generator_lr, betas=(args.generator_adam_beta1, args.generator_adam_beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.discriminator_lr, betas=(args.discriminator_adam_beta1, args.discriminator_adam_beta2))

    criterion = torch.nn.BCEWithLogitsLoss()

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.category)
    if not (os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir)):
        os.makedirs(checkpoint_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset, args.category, "epoch")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
    
    G.train()
    D.train()

    dtype = torchutils.get_torch_dtype(args.mixed_precision)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    use_scaler = (device.type == "cuda") and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    test_noise = torch.randn(1, args.latent_dim, device=device)
    for epoch in range(args.num_epochs):
        running_G_loss = 0.0
        running_D_loss = 0.0
        running_real_loss = 0.0
        running_fake_loss = 0.0
        
        running_D_acc = 0.0
        running_real_score = 0.0
        running_fake_score = 0.0

        if args.use_hinge_loss:
            running_real_margin = 0.0
            running_fake_margin = 0.0
        with tqdm(dataloader, desc=f"Training", unit="batch") as pbar:
            for idx, images in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                images = images * 2.0 - 1.0
                batch_size = images.size(0)

                noise_std = max(0.0, 0.05 * (1.0 - epoch / 100)) if args.use_instance_noise else 0.0

                # === Train Discriminator ===
                noise = torch.randn(batch_size, args.latent_dim, device=device)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    fake_images = G(noise)
                    if args.use_hinge_loss:
                        fake_images = torch.tanh(fake_images)
                    else:
                        fake_images = torch.sigmoid(fake_images)

                    real_out = D(images + torch.randn_like(images) * noise_std)
                    fake_out = D(fake_images.detach() + torch.randn_like(fake_images) * noise_std)

                    real_logits = real_out.view(-1)
                    fake_logits = fake_out.view(-1)

                    real_labels = torch.ones_like(real_logits)
                    fake_labels = torch.zeros_like(fake_logits)

                    if args.use_hinge_loss:
                        real_loss = torch.mean(torch.clamp(1.0 - real_out, min=0.0))
                        fake_loss = torch.mean(torch.clamp(1.0 + fake_out, min=0.0))
                    else:
                        real_loss = criterion(real_logits, real_labels)
                        fake_loss = criterion(fake_logits, fake_labels)

                D_loss = real_loss + fake_loss

                with torch.no_grad():
                    real_score = torch.sigmoid(real_logits).mean().item()
                    fake_score = torch.sigmoid(fake_logits).mean().item()

                    real_pred = real_logits >= 0
                    fake_pred = fake_logits < 0
                    D_accuracy = torch.cat([real_pred, fake_pred]).float().mean()

                D_optimizer.zero_grad(set_to_none=True)
                scaler.scale(D_loss).backward()
                scaler.step(D_optimizer)

                running_real_score += real_score
                running_fake_score += fake_score

                if args.use_hinge_loss:
                    running_real_margin += real_logits.mean().item()
                    running_fake_margin += fake_logits.mean().item()

                running_D_loss += D_loss.item()
                running_real_loss += real_loss.item()
                running_fake_loss += fake_loss.item()
                running_D_acc += D_accuracy.item()

                # === Train Generator ===
                G_optimizer.zero_grad(set_to_none=True)

                noise = torch.randn(batch_size, args.latent_dim, device=device)

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    fake_images = G(noise)
                    if args.use_hinge_loss:
                        fake_images = torch.tanh(fake_images)
                    else:
                        fake_images = torch.sigmoid(fake_images)
                    fake_out = D(fake_images + torch.randn_like(fake_images) * noise_std)

                    fake_logits = fake_out.view(-1)
                    real_labels = torch.ones_like(fake_logits)

                    if args.use_hinge_loss:
                        G_loss = -torch.mean(fake_out)
                    else:
                        G_loss = criterion(fake_logits, real_labels)
        
                scaler.scale(G_loss).backward()
                scaler.step(G_optimizer)
                scaler.update()

                running_G_loss += G_loss.item()

                pbar.set_postfix({
                    "G Loss": f"{G_loss.item():.2f}",
                    "D Loss": f"{D_loss.item():.2f}",
                    "D(real) Loss": f"{real_loss.item():.2f}",
                    "D(fake) Loss": f"{fake_loss.item():.2f}",
                })
            
        average_G_loss = running_G_loss / len(dataloader)
        average_D_loss = running_D_loss / len(dataloader)
        average_real_loss = running_real_loss / len(dataloader)
        average_fake_loss = running_fake_loss / len(dataloader)
        average_D_acc = running_D_acc / len(dataloader)
        average_real_score = running_real_score / len(dataloader)
        average_fake_score = running_fake_score / len(dataloader)
        
        print(f"Epoch: {epoch + 1}/{args.num_epochs}")
        print(f"G Loss: {average_G_loss:.5f}, D Loss: {average_D_loss:.5f}, D(real) Loss: {average_real_loss:.5f}, D(fake) Loss: {average_fake_loss:.5f}")
        print(f"D Accuracy: {average_D_acc:.2f}, Real Score: {average_real_score:.2f}, Fake Score: {average_fake_score:.2f}")

        if args.use_hinge_loss:
            average_real_margin = running_real_margin / len(dataloader)
            average_fake_margin = running_fake_margin / len(dataloader)
            print(f"Real Margin: {average_real_margin:.2f}, Fake Margin: {average_fake_margin:.2f}")

        if (epoch + 1) % args.save_output_interval == 0:
            print("Saving fake images")
            G.eval()
            with torch.no_grad():
                fake_images = G(test_noise)
                if args.use_hinge_loss:
                    fake_images = torch.tanh(fake_images)
                    level = 0.0
                else:
                    fake_images = torch.sigmoid(fake_images)
                    level = 0.5
                fake_images = fake_images.squeeze().cpu().numpy()
            G.train()

            min_val = float(fake_images.min())
            max_val = float(fake_images.max())

            if not (min_val < level < max_val):
                level = (min_val + max_val) / 2.0

            faces = render_orthographic_faces(fake_images=fake_images, level=level)
            faces.save(os.path.join(output_dir, f"{epoch + 1:0{pad_length}d}.png"))

        if (epoch + 1) % args.save_checkpoint_interval == 0:
            checkpoint = {
                "dataset": args.dataset,
                "category": args.category,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "discriminator_in_channels": args.discriminator_in_channels,
                "generator_lr": args.generator_lr,
                "discriminator_lr": args.discriminator_lr,
                "G_optimizer": G_optimizer.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                "latent_dim": args.latent_dim,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.dataset}-{args.category}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
            generator = {
                "dataset": args.dataset,
                "category": args.category,
                "G": G.state_dict(),
                "generator_in_channels": args.generator_in_channels,
                "voxel_resolution": args.voxel_resolution,
                "latent_dim": args.latent_dim,
            }
            torch.save(generator, os.path.join(checkpoint_dir, f"{args.dataset}-{args.category}_{epoch + 1:0{pad_length}d}.pth"))

    print("Saving final model checkpoints")
    checkpoint = {
        "dataset": args.dataset,
        "category": args.category,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "discriminator_in_channels": args.discriminator_in_channels,
        "generator_lr": args.generator_lr,
        "discriminator_lr": args.discriminator_lr,
        "G_optimizer": G_optimizer.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "latent_dim": args.latent_dim,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"{args.dataset}-{args.category}_checkpoint_{epoch + 1:0{pad_length}d}.pth"))
    generator = {
        "dataset": args.dataset,
        "category": args.category,
        "G": G.state_dict(),
        "generator_in_channels": args.generator_in_channels,
        "voxel_resolution": args.voxel_resolution,
        "latent_dim": args.latent_dim,
    }
    torch.save(generator, os.path.join(checkpoint_dir, f"{args.dataset}-{args.category}_{epoch + 1:0{pad_length}d}.pth"))

if __name__ == "__main__":
    main()
