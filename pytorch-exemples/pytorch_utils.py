"""Fonctions utilitaires partagées pour les travaux pratiques de Deep Learning.

Ce module regroupe les fonctions réutilisées d'un notebook à l'autre :
entraînement, évaluation, visualisation des résultats, et un Dataset
optimisé pour le chargement d'images depuis le disque.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class FastImageFolder(ImageFolder):
    """ImageFolder qui charge un batch d'images en parallèle (threads) via
    __getitems__, une méthode appelée par le DataLoader à la place de plusieurs
    appels successifs à __getitem__ lorsqu'elle est définie. La lecture disque
    et le décodage JPEG libèrent le GIL, donc ce parallélisme accélère le
    chargement d'un batch.
    """

    def __getitems__(self, indices):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.__getitem__, indices))


def train_net(model, train_loader, val_loader, num_epochs,
              optimizer, criterion, scheduler=None, device=None):
    """
    Entraîne un modèle PyTorch et retourne le modèle et l'historique.

    Paramètres
    ----------
    model        : nn.Module  --> modèle déjà instancié
    train_loader : DataLoader --> données d'entraînement
    val_loader   : DataLoader --> données de validation
    num_epochs   : int        --> nombre d'époques
    optimizer    : Optimizer  --> optimiseur construit sur les paramètres du modèle
    criterion    : nn.Module  --> fonction de coût
    scheduler    : LRScheduler | None --> ajustement du taux d'apprentissage (optionnel)
    device       : torch.device | None --> auto-détecté si None

    Retours
    -------
    model   : nn.Module --> modèle entraîné
    history : dict      --> clés 'loss', 'accuracy', 'val_loss', 'val_accuracy'
    """

    # detecter et utiliser l'accelerateur pour votre machine
    if device is None:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Chaque pas d'entraînement comprend une étape d'entraînement et une étape de validation
    for epoch in range(num_epochs):

        # Phase entraînement
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Phase validation
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                val_loss_sum += criterion(logits, y_batch).item() * X_batch.size(0)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # ReduceLROnPlateau attend la métrique surveillée, les autres schedulers non
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d}/{num_epochs}  '
              f'loss={train_loss:.4f}  acc={train_acc:.4f}  '
              f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  '
              f'LR={current_lr:.2e}')

    return model, history


def evaluate(model: nn.Module, dataset,
             device=None, batch_size: int = 256, num_workers: int = 0) -> tuple:
    """
    Évalue un modèle sur un Dataset complet.
    Équivalent de model.evaluate() en Keras.

    Retourne (loss, accuracy).
    """
    if device is None:
        device = next(model.parameters()).device   # même device que le modèle

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)

    loss = total_loss / total
    acc = correct / total
    print(f'loss={loss:.4f}  accuracy={acc:.4f}')
    return loss, acc


def plot_training_history(history, save_fig=False, fig_name=''):
    """Affiche l'évolution de la performance et de la fonction de coût."""
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if save_fig:
        plt.savefig('accuracy' + fig_name + '.png')
    plt.show()

    plt.figure()
    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if save_fig:
        plt.savefig('loss' + fig_name + '.png')
    plt.show()


def _denormalize(img: torch.Tensor, mean, std) -> torch.Tensor:
    """Annule la normalisation pour retrouver des couleurs correctes à l'affichage."""
    img = img.cpu()
    if mean is None or std is None:
        return img
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (img * std_t + mean_t).clamp(0, 1)


def model_predict(model, img: torch.Tensor, true_label: int,
                  class_names, mean=None, std=None) -> None:
    """
    Affiche une image et la prédiction du modèle.

    Paramètres
    ----------
    model       : nn.Module --> modèle entraîné
    img         : Tensor (C,H,W) --> image issue du Dataset (format PyTorch CHW)
    true_label  : int --> label correct
    class_names : list[str] --> noms des classes, indexés par label
    mean, std   : tuple | None --> statistiques de normalisation, pour l'affichage
    """
    device = next(model.parameters()).device

    # plt.imshow attend (H, W, C) — on permute les axes CHW → HWC
    plt.imshow(_denormalize(img, mean, std).permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    # Prédiction
    # img.unsqueeze(0) : (C,H,W) → (1,C,H,W)
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        y_hat = logits.argmax(dim=1).item()

    print(f"The image is a      : {class_names[true_label]}")
    print(f"The model predicted : {class_names[y_hat]}")


def plot_predictions_grid(model, dataset, indices, class_names,
                          mean=None, std=None, ncols=4):
    """
    Affiche une grille d'images avec, pour chacune, la classe prédite, la
    confiance du modèle (softmax) et la classe réelle. Le titre est vert si
    la prédiction est correcte, rouge sinon.

    Paramètres
    ----------
    model       : nn.Module --> modèle entraîné
    dataset     : Dataset   --> dataset d'où proviennent les images (ex: test_set)
    indices     : list[int] --> indices des exemples à afficher
    class_names : list[str] --> noms des classes, indexés par label
    mean, std   : tuple | None --> statistiques de normalisation, pour l'affichage
    ncols       : int       --> nombre de colonnes de la grille
    """
    device = next(model.parameters()).device

    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            img, true_label = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)
            confidence, y_hat = probs.max(dim=1)
            y_hat, confidence = y_hat.item(), confidence.item()

            ax.imshow(_denormalize(img, mean, std).permute(1, 2, 0))
            ax.axis('off')

            color = 'green' if y_hat == true_label else 'red'
            ax.set_title(
                f"{class_names[y_hat]} ({confidence:.0%})\nvrai : {class_names[true_label]}",
                color=color, fontsize=9,
            )

    # Cacher les axes restants si le nombre d'images ne remplit pas la grille
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
