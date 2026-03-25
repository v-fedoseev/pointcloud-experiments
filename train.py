import argparse
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import schedulefree

from src.models.pointnet_cls import get_model, get_loss
from src.data_utils.ModelNetDatDataset import ModelNetDatDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dat", default="data/modelnet40_train_1024pts.dat")
    parser.add_argument("--test_dat", default="data/modelnet40_test_1024pts.dat")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_category", type=int, default=40)
    parser.add_argument("--log_dir", default="runs/")
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.transpose(2, 1).float().to(device)
            labels = labels.long().to(device)
            logits, trans_feat = model(points)
            loss = criterion(logits, labels, trans_feat)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cpu")

    train_dataset = ModelNetDatDataset(args.train_dat, npoints=1024, use_normals=True)
    test_dataset = ModelNetDatDataset(args.test_dat, npoints=1024, use_normals=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model(k=args.num_category, normal_channel=True).to(device)
    criterion = get_loss()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr)

    # Hook to capture the 3×3 input STN output (trans) for determinant logging
    stn_output = {}
    def stn_hook(module, input, output):
        stn_output["trans"] = output.detach()
    model.feat.stn.register_forward_hook(stn_hook)

    run_name = "pointnet_cls_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    os.makedirs("checkpoints/pointnet_cls", exist_ok=True)
    best_test_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        det_list = []

        optimizer.train()
        for points, labels in train_loader:
            points = points.transpose(2, 1).float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            logits, trans_feat = model(points)
            loss = criterion(logits, labels, trans_feat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if "trans" in stn_output:
                dets = torch.det(stn_output["trans"])
                det_list.append(dets)

        train_loss = total_loss / total
        train_acc = correct / total

        optimizer.eval()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)

        if det_list:
            all_dets = torch.cat(det_list)
            writer.add_scalar("tnet/det_mean", all_dets.mean().item(), epoch)
            writer.add_scalar("tnet/det_std", all_dets.std().item(), epoch)

        print(
            f"Epoch {epoch}/{args.num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/pointnet_cls/best_model.pth")

    writer.close()
    print(f"Done. Best test acc: {best_test_acc:.4f}. TensorBoard: {os.path.join(args.log_dir, run_name)}")


if __name__ == "__main__":
    main()
