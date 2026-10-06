"""
QuantumPath: hybrid quantum vision benchmark (Track 8)

Compares four models on BreastMNIST (ultrasound, malignant vs benign/normal):
  1. Classical CNN baseline (the number to beat)
  2. QSVC  - quantum kernel + classical SVM, on CNN features
  3. VQC   - CNN -> variational quantum circuit, trained end to end
  4. QCNN  - CNN -> quantum convolution/pooling circuit, trained end to end

Run:   python quantum_diagnostics.py --n-train 600 --epochs 5
Output: results.json (load it into the website dashboard)

Research prototype only. Not for clinical use.
"""
import argparse, copy, json, math, time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from medmnist import BreastMNIST
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

NQ = 4  # qubits. Keep small: simulator cost doubles with every qubit.


# ---------------------------------------------------------------- data
def load(n_train, n_test, seed=0):
    rng = np.random.default_rng(seed)

    def get(split, n):
        ds = BreastMNIST(split=split, download=True)
        x = ds.imgs.astype("float32")[:, None] / 255.0          # (N,1,28,28)
        y = 1 - ds.labels.ravel().astype("float32")             # 1 = malignant (check medmnist INFO)
        idx = rng.permutation(len(x))[:n]
        return torch.tensor(x[idx]), torch.tensor(y[idx])

    return get("train", n_train), get("test", n_test)


# -------------------------------------------------------------- models
class Encoder(nn.Module):
    """Small CNN that turns an image into 16 features."""
    def __init__(self, out=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Scale(nn.Module):
    """Maps tanh output (-1..1) to rotation angles (-pi..pi)."""
    def forward(self, x):
        return x * math.pi


class Net(nn.Module):
    def __init__(self, enc, mid, head):
        super().__init__()
        self.enc, self.mid, self.head = enc, mid, head

    def forward(self, x):
        return self.head(self.mid(self.enc(x)))


def vqc_layer(layers=2):
    dev = qml.device("default.qubit", wires=NQ)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(NQ))
        qml.StronglyEntanglingLayers(weights, wires=range(NQ))
        return [qml.expval(qml.PauliZ(i)) for i in range(NQ)]

    return qml.qnn.TorchLayer(circuit, {"weights": (layers, NQ, 3)}), NQ


def _conv(p, a, b):
    qml.RY(p[0], wires=a); qml.RY(p[1], wires=b)
    qml.CNOT(wires=[a, b]); qml.RZ(p[2], wires=b); qml.CNOT(wires=[a, b])
    qml.RY(p[3], wires=a)


def _pool(p, src, dst):
    qml.CRZ(p[0], wires=[src, dst])
    qml.PauliX(wires=src); qml.CRX(p[1], wires=[src, dst]); qml.PauliX(wires=src)


def qcnn_layer():
    """4 qubits: conv on pairs -> conv across pairs -> pool 4->2 -> conv -> read qubit 3."""
    dev = qml.device("default.qubit", wires=NQ)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(NQ))
        _conv(weights[0], 0, 1); _conv(weights[0], 2, 3)   # shared weights = translation invariance
        _conv(weights[1], 1, 2)
        _pool(weights[2], 0, 1); _pool(weights[2], 2, 3)   # keep qubits 1 and 3
        _conv(weights[3], 1, 3)
        return [qml.expval(qml.PauliZ(3))]

    return qml.qnn.TorchLayer(circuit, {"weights": (4, 4)}), 1


def quantum_net(base_enc, layer_fn):
    qlayer, n_out = layer_fn()
    mid = nn.Sequential(nn.Linear(16, NQ), nn.Tanh(), Scale(), qlayer)
    return Net(copy.deepcopy(base_enc), mid, nn.Linear(n_out, 1))


# ------------------------------------------------------------ training
def fit(model, X, y, epochs, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward(); opt.step(); total += loss.item()
        print(f"  epoch {ep + 1}/{epochs}  loss {total / len(loader):.4f}")


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X).squeeze(1)).numpy()


def metrics(y, p, secs):
    y = y.numpy().astype(int)
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 3),
            "auc": round(roc_auc_score(y, p), 3),
            "sensitivity": round(recall_score(y, pred), 3),   # malignant cases caught
            "seconds": round(secs, 1)}


# ---------------------------------------------------------------- QSVC
def qsvc(enc, Xtr, ytr, Xte, n_kernel):
    enc.eval()
    with torch.no_grad():
        Ftr, Fte = enc(Xtr[:n_kernel]).numpy(), enc(Xte).numpy()
    pca = PCA(NQ).fit(Ftr)
    sc = MinMaxScaler((0, math.pi)).fit(pca.transform(Ftr))
    Ftr, Fte = sc.transform(pca.transform(Ftr)), np.clip(sc.transform(pca.transform(Fte)), 0, math.pi)

    dev = qml.device("default.qubit", wires=NQ)

    @qml.qnode(dev)
    def overlap(a, b):
        qml.IQPEmbedding(a, wires=range(NQ), n_repeats=2)
        qml.adjoint(qml.IQPEmbedding(b, wires=range(NQ), n_repeats=2))
        return qml.probs(wires=range(NQ))

    kern = lambda a, b: overlap(a, b)[0]          # probability of returning to |0000>
    Ktr = qml.kernels.kernel_matrix(Ftr, Ftr, kern)
    Kte = qml.kernels.kernel_matrix(Fte, Ftr, kern)
    svc = SVC(kernel="precomputed", probability=True).fit(Ktr, ytr[:n_kernel].numpy())
    return svc.predict_proba(Kte)[:, 1]


# ------------------------------------------------------------ Grad-CAM
def gradcam(model, X, path="gradcam.png", k=4):
    """Heatmaps of where the CNN encoder looked (last conv layer)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conv, store = model.enc.net[3], {}
    h1 = conv.register_forward_hook(lambda m, i, o: store.update(a=o))
    h2 = conv.register_full_backward_hook(lambda m, gi, go: store.update(g=go[0]))
    model.eval()
    fig, ax = plt.subplots(2, k, figsize=(3 * k, 6))
    for j in range(k):
        x = X[j:j + 1]
        model.zero_grad()
        model(x).squeeze().backward()
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store["a"]).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=28, mode="bilinear")[0, 0].detach().numpy()
        cam = cam / (cam.max() + 1e-8)
        ax[0, j].imshow(x[0, 0], cmap="gray"); ax[0, j].set_title("image")
        ax[1, j].imshow(x[0, 0], cmap="gray"); ax[1, j].imshow(cam, cmap="jet", alpha=0.45); ax[1, j].set_title("Grad-CAM")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    h1.remove(); h2.remove()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=156)
    ap.add_argument("--n-kernel", type=int, default=100, help="train samples for QSVC (cost grows with the square)")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(0)

    (Xtr, ytr), (Xte, yte) = load(a.n_train, a.n_test)
    results = {}

    print("Classical baseline")
    t = time.time()
    base = Net(Encoder(), nn.Identity(), nn.Linear(16, 1))
    fit(base, Xtr, ytr, a.epochs)
    results["Classical CNN"] = metrics(yte, predict(base, Xte), time.time() - t)
    gradcam(base, Xte)

    print("QSVC (quantum kernel on CNN features)")
    t = time.time()
    results["QSVC"] = metrics(yte, qsvc(base.enc, Xtr, ytr, Xte, a.n_kernel), time.time() - t)

    for name, fn in [("VQC", vqc_layer), ("QCNN", qcnn_layer)]:
        print(f"{name} (hybrid, end to end)")
        t = time.time()
        net = quantum_net(base.enc, fn)
        fit(net, Xtr, ytr, a.epochs)
        results[name] = metrics(yte, predict(net, Xte), time.time() - t)

    print(json.dumps(results, indent=2))
    json.dump(results, open("results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
