# Servir le modèle ONNX en C++

Petit programme en C++ qui charge le modèle `beans_resnet18.onnx` — exporté par le
notebook [`pytorch-exemples/Transfer_learning_pytorch.ipynb`](../pytorch-exemples/Transfer_learning_pytorch.ipynb) —
et classe une image de feuille de haricot passée en argument.

C'est la démonstration concrète de l'intérêt d'ONNX : **ni Python ni PyTorch ne sont
installés ici**. Le serveur n'a besoin que d'ONNX Runtime et d'une bibliothèque pour
lire les images.

```
$ ./beans_classifier feuille.jpg
Image  : feuille.jpg (500x500)
Modèle : ../pytorch-exemples/beans_resnet18.onnx

Prédiction : bean_rust (92.4 %)

Détail des probabilités :
  angular_leaf_spot : 4.1 %
  bean_rust : 92.4 %
  healthy : 3.5 %
```

> Système visé : **Ubuntu / Linux**. Les commandes ci-dessous ont été écrites pour
> Ubuntu (22.04 ou plus récent).

---

## 1. Prérequis

### Outils de compilation et OpenCV

OpenCV sert uniquement à lire l'image sur le disque et à la redimensionner.

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

### ONNX Runtime

ONNX Runtime n'est pas disponible dans les dépôts Ubuntu : on télécharge l'archive
officielle et on la décompresse (ici dans le dossier personnel).

**Vérifiez d'abord l'architecture de votre machine**, car l'archive en dépend :

```bash
uname -m
```

* `x86_64` → PC Intel/AMD classique, machine virtuelle sur PC, serveur cloud ;
* `aarch64` → **machine virtuelle sur Mac Apple Silicon (Parallels, UTM, VMware)**,
  Raspberry Pi, serveur ARM (AWS Graviton…).

Téléchargez ensuite l'archive correspondante :

```bash
cd ~

# Architecture x86_64 (Intel/AMD)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz
tar -xzf onnxruntime-linux-x64-1.30.0.tgz

# Architecture aarch64 (ARM, dont les VM sur Mac Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-aarch64-1.30.0.tgz
tar -xzf onnxruntime-linux-aarch64-1.30.0.tgz
```

Vous obtenez un dossier `~/onnxruntime-linux-<arch>-1.30.0` contenant `include/` et
`lib/`. C'est ce chemin qui sera donné à CMake.

> Se tromper d'architecture ne provoque **aucune erreur au téléchargement ni à la
> compilation** : l'erreur n'apparaît qu'à l'édition de liens, sous la forme
> `libonnxruntime.so: error adding symbols: file in wrong format` (voir la section
> « En cas de problème »).

> La version **1.30.0** correspond à celle utilisée dans le notebook. Une version plus
> ancienne qu'ONNX Runtime 1.19 ne saura pas lire ce modèle (il utilise l'IR version 10).

### Le modèle `.onnx`

Le fichier `beans_resnet18.onnx` **n'est pas versionné dans le dépôt** (il pèse 45 Mo
et figure dans le `.gitignore`). Il faut donc l'obtenir avant de lancer le programme,
en exécutant la section « Servir le modèle avec ONNX » du notebook, qui le crée dans
`pytorch-exemples/`.

---

## 2. Compilation

Depuis ce dossier (`example_cpp_onnx/`) :

```bash
# Remplacez <arch> par x64 ou aarch64 selon le résultat de `uname -m`
cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-<arch>-1.30.0
cmake --build build
```

Le binaire est produit dans `build/beans_classifier`.

---

## 3. Exécution

```bash
# Le modèle est cherché par défaut dans ../pytorch-exemples/beans_resnet18.onnx
./build/beans_classifier une_image.jpg

# Ou en indiquant explicitement un autre modèle
./build/beans_classifier une_image.jpg /chemin/vers/mon_modele.onnx
```

Pour tester rapidement, n'importe quelle image du jeu de test fait l'affaire :

```bash
./build/beans_classifier ../pytorch-exemples/data/beans/test/bean_rust/0003.jpg
```

---

## 4. Le contrat du modèle (le point le plus important)

Le fichier `.onnx` ne contient **que le réseau**. Tout le reste doit être reproduit
à l'identique côté C++, et une erreur ici ne provoque **aucun message** : le programme
répond simplement n'importe quoi.

| Élément | Valeur attendue | Où dans le code |
|---|---|---|
| Taille d'entrée | 224 × 224 | `kImageSize` |
| Ordre des canaux | **RGB** (OpenCV lit en BGR) | `cv::cvtColor(..., COLOR_BGR2RGB)` |
| Échelle des pixels | divisés par 255 → `[0, 1]` | `convertTo(..., 1.0/255.0)` |
| Normalisation | moyenne/écart-type ImageNet | `kMean`, `kStd` |
| Disposition | `CHW` et non `HWC` | boucle de `PreprocessImage` |
| Ordre des classes | **alphabétique** : `angular_leaf_spot`, `bean_rust`, `healthy` | `kClassNames` |

L'ordre des classes vient d'`ImageFolder`, qui trie les dossiers par ordre
alphabétique. Le réseau ne renvoie que des indices (0, 1, 2) : c'est cette liste,
transportée séparément, qui leur donne un sens.

### Une différence attendue avec le notebook

Le notebook redimensionne les images avec PIL, ce programme avec OpenCV. Les deux
bibliothèques n'interpolent pas exactement de la même façon, donc les logits ne sont
pas identiques au bit près.

La comparaison a été mesurée sur des images du jeu de test : l'écart maximal observé
sur les logits est de **0,185**, et la **classe prédite reste la même** (10 images
sur 10). À titre de comparaison, oublier l'inversion BGR→RGB change déjà la classe
prédite, et oublier la division par 255 décale les logits de plus de **41**. Autrement
dit, un écart de l'ordre de 0,1 est normal ; un écart de plusieurs unités signale une
erreur de prétraitement.

---

## 5. En cas de problème

**`libonnxruntime.so: error adding symbols: file in wrong format`** (à l'édition de liens)

L'archive ONNX Runtime téléchargée ne correspond pas à l'architecture de la machine.
C'est le cas typique d'une VM Linux sur Mac Apple Silicon : la VM est en `aarch64`,
alors que l'archive `...-linux-x64-...` est prévue pour du Intel/AMD. Vérifiez :

```bash
uname -m                                             # architecture de la machine
file onnxruntime-linux-*/lib/libonnxruntime.so       # architecture de la bibliothèque
```

Les deux doivent concorder (`aarch64` avec `ARM aarch64`, `x86_64` avec `x86-64`).
Si ce n'est pas le cas, retéléchargez la bonne archive (section 1), puis
**reconfigurez entièrement** — le cache de CMake garde l'ancien chemin :

```bash
rm -rf build
cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-aarch64-1.30.0
cmake --build build
```

**`error while loading shared libraries: libonnxruntime.so`**
Le chemin de la bibliothèque est normalement inscrit dans le binaire par CMake
(`RPATH`). Si le dossier d'ONNX Runtime a été déplacé depuis la compilation :

```bash
export LD_LIBRARY_PATH=$HOME/onnxruntime-linux-<arch>-1.30.0/lib:$LD_LIBRARY_PATH
```

**`ONNX Runtime introuvable` lors du `cmake`**
Le chemin donné à `-DONNXRUNTIME_ROOT` est incorrect. Il doit pointer vers le dossier
qui contient `include/` et `lib/` :

```bash
ls $HOME/onnxruntime-linux-<arch>-1.30.0   # doit afficher include  lib  ...
```

**`Could NOT find OpenCV`**
Le paquet de développement manque : `sudo apt install libopencv-dev`.

**`impossible de lire l'image`**
Chemin incorrect, ou format non géré par OpenCV. Le JPEG et le PNG fonctionnent.

---

## 6. Confort de développement : clangd (optionnel)

[clangd](https://clangd.llvm.org/) fournit l'autocomplétion, la navigation dans le
code et les diagnostics en direct. Il s'appuie sur le fichier
`build/compile_commands.json`, que CMake génère **automatiquement** ici (voir
`CMAKE_EXPORT_COMPILE_COMMANDS` dans `CMakeLists.txt`). Il suffit donc d'avoir lancé
`cmake -B build` une fois.

Sous Ubuntu :

```bash
sudo apt install clangd
```

Dans VS Code, installez l'extension **clangd**
(`llvm-vs-code-extensions.vscode-clangd`), puis désactivez le moteur IntelliSense de
l'extension Microsoft C/C++ : les deux analysent le code en parallèle et se marchent
dessus. Dans les réglages de l'espace de travail (`.vscode/settings.json`) :

```json
{
  "C_Cpp.intelliSenseEngine": "disabled"
}
```

> Ce fichier `.vscode/settings.json` n'est volontairement pas versionné : il
> désactiverait IntelliSense chez quelqu'un qui n'a pas installé clangd, le laissant
> sans aucune assistance. Le fichier `.clangd`, lui, est versionné car il est utile
> à tous les éditeurs (VS Code, Zed, Neovim…).

### Si vous éditez depuis une autre machine que celle qui compile

clangd a besoin des en-têtes d'ONNX Runtime et d'OpenCV pour résoudre les `#include`.
Si vous éditez le code sur un poste où ces bibliothèques ne sont pas installées (par
exemple sous macOS, en compilant sur une VM Linux), les deux `#include` seront
signalés comme introuvables et tous les symboles `cv::` et `Ort::` apparaîtront en
erreur. Deux solutions :

* utiliser **VS Code Remote-SSH** pour éditer directement sur la machine Linux :
  clangd s'y exécute et dispose alors de tous les en-têtes ;
* ou masquer uniquement ce bruit en décommentant la section `Diagnostics` du fichier
  `.clangd` (voir les commentaires qu'il contient). À utiliser en connaissance de
  cause : cela masque aussi un `#include` réellement mal écrit.

---

## 7. Pour aller plus loin

* Les fournisseurs d'exécution (CUDA, TensorRT…) permettent d'accélérer l'inférence
  sans changer le code : https://onnxruntime.ai/docs/execution-providers/
* API C++ d'ONNX Runtime : https://onnxruntime.ai/docs/api/c/
* Le format ONNX : https://onnx.ai/
