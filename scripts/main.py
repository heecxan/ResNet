import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import matplotlib.pyplot as plt
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

from utils.data_preprocess import get_loaders
from utils.loss_fn import get_loss
from model.resnet50 import resnet50
from utils.train_utils import train, evaluate

T = TypeVar("T")

def parse_config(config_path: str, cls: Type[T]) -> T:
    with open(config_path, "r") as f:
        config_data = json.load(f)

    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}

    return cls(**filtered_data)


@dataclass
class Config:
    model_name: str = field(
        default="mlp", metadata={"help": "The name of the model to use."}
    )

    hidden_size: int = field(
        default=768, metadata={"help": "The size of the hidden layer."}
    )

    n_classes: int = field(
        default=10, metadata={"help": "The number of classes to classify."}
    )

    batch_size: int = field(default=128, metadata={"help": "The batch size to use."})

    opt_name: str = field(
        default="adam", metadata={"help": "The name of the optimizer to use."}
    )

    lr: float = field(default=1e-3, metadata={"help": "The learning rate to use."})

    num_epochs: int = field(
        default=100, metadata={"help": "The number of epochs to train for."}
    )

    early_stopping: int = field(
        default=None,
        metadata={"help": "The number of epochs to wait before early stopping."},
    )

    p: float = field(default=0.1, metadata={"help": "The dropout rate to use."})


def main():
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        config_path = sys.argv[1]
    else:
        config_path = "config/config.json"

    config = parse_config(config_path, Config)
    os.makedirs(f"{config.model_name}_logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로드
    train_loader, val_loader, test_loader = get_loaders(config.batch_size)

    # 모델 정의
    model = resnet50(num_classes=config.n_classes).to(device)

    # 손실함수, 옵티마이저
    criterion = get_loss("cross_entropy")

    if config.opt_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.opt_name}")

    # 결과를 리스트에 저장
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 학습 루프
    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, mode="val")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"[Epoch {epoch+1}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # 테스트 평가
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, mode="test")
    print(f"[Test] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "Resnet_logs/best_model.pth")

    # 학습 결과 plot
    log_dir = f"{config.model_name}_logs"

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "loss_plot.png"))
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "accuracy_plot.png"))
    plt.show()

if __name__ == "__main__":
    main()
