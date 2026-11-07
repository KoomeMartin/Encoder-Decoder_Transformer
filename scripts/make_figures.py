"""Generate the README figures (light + dark variants).

    python scripts/make_figures.py

Run it from any directory; it works relative to the repository root. Every figure is built
from real material in this repository:

* the handout subset in ``hw3_data_subset/`` (utterance 103-1240-0000), processed with the
  project's own ``hw3lib`` code (ASRDataset normalisation and SpecAugment, the 10k BPE tokenizer,
  the Conv2D speech embedding, the attention masks);
* the final configuration in ``configs/final.yaml``;
* the W&B run export and Kaggle metadata in ``submission_artifacts/``.

Outputs (written to ./assets):
    architecture-{light,dark}.svg   end-to-end pipeline and encoder-decoder layout
    features-{light,dark}.svg       raw filterbank -> global MVN -> SpecAugment
    subsampling-{light,dark}.svg    Conv2D speech embedding: tensor shapes and receptive field
    tokens-{light,dark}.svg         tokenisation and teacher forcing
    masks-{light,dark}.svg          encoder pad mask and decoder causal mask
    training-{light,dark}.svg       losses, dev error rates and learning rate per epoch
    kaggle-{light,dark}.svg         Kaggle public-leaderboard history
"""

import base64
import contextlib
import datetime as dt
import io
import json
import os
import pickle
import pickletools
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
ARTIFACTS = os.path.join(ROOT, "submission_artifacts")
SUBSET = os.path.join(ROOT, "hw3_data_subset", "hw3p2_data")
UTT = "103-1240-0000"
sys.path.insert(0, ROOT)

THEMES = {
    "light": {
        "surface": "#fcfcfb", "panel": "#f4f3ef", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "s1": "#2a78d6", "s1_soft": "#cde2fb", "s2": "#eb6834", "s2_soft": "#fbe0d4",
        "s3": "#1baf7a", "s3_soft": "#d3f1e5", "neutral": "#e9e8e3",
        "heat": ["#f4f3ef", "#8ab6ea", "#2a78d6", "#0b2f5e"],
    },
    "dark": {
        "surface": "#1a1a19", "panel": "#232322", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "s1": "#3987e5", "s1_soft": "#16375e", "s2": "#d95926", "s2_soft": "#4a2614",
        "s3": "#199e70", "s3_soft": "#123d2e", "neutral": "#2f2f2d",
        "heat": ["#1a1a19", "#1d4d86", "#3987e5", "#cfe4ff"],
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace'


def heat_cmap(t):
    return LinearSegmentedColormap.from_list("heat", t["heat"])


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def show_token(piece):
    """Byte-level BPE marks a leading space with 'Ġ'; show it as a middle dot."""
    return piece.replace("Ġ", "·")


# --------------------------------------------------------------------------- #
# Data: everything comes from the repo's own code and files
# --------------------------------------------------------------------------- #
def load_inputs():
    import torch
    import yaml

    from hw3lib.data import ASRDataset, H3Tokenizer
    from hw3lib.model.masks import CausalMask, PadMask
    from hw3lib.model.speech_embedding import SpeechEmbedding

    os.chdir(ROOT)  # tokenizer paths in the config are relative to the repo root
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "final.yaml"), encoding="utf-8"))
    data_cfg = dict(cfg["data"], root=SUBSET, feat_dtype="float32", load_threads=8, subset=1.0)

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        tok_map = cfg["tokenization"]["token_map"]
        tokenizers = {k: H3Tokenizer(tok_map, k) for k in ["char", "1k", "5k", "10k"]}
        ds = ASRDataset(partition="train-clean-100", config=data_cfg,
                        tokenizer=tokenizers[cfg["tokenization"]["token_type"]],
                        isTrainPartition=True, global_stats=None)

    idx = ds.fbank_files.index(f"{UTT}.npy")
    raw = np.load(os.path.join(SUBSET, "train-clean-100", "fbank", f"{UTT}.npy"))
    text = "".join(np.load(os.path.join(SUBSET, "train-clean-100", "text", f"{UTT}.npy")))
    sample = ds[idx]
    norm = sample[0].numpy()  # (80, T), global MVN with subset statistics

    # SpecAugment exactly as collate_fn applies it; pick a seed whose masks are all visible
    def runs(flags):
        out, start = [], None
        for i, f in enumerate(list(flags) + [False]):
            if f and start is None:
                start = i
            elif not f and start is not None:
                out.append((start, i))
                start = None
        return out

    for seed in range(200):
        torch.manual_seed(seed)
        aug = ds.collate_fn([sample])[0][0].numpy().T  # (80, T)
        zero = aug == 0
        f_runs = runs(zero.all(axis=1))
        t_runs = runs(zero.all(axis=0))
        if (len(f_runs) == 2 and len(t_runs) == 4 and min(b - a for a, b in f_runs) >= 8
                and min(b - a for a, b in t_runs) >= 20):
            break

    # Conv2D speech embedding, built from the final config, run on this utterance
    m = cfg["model"]
    emb = SpeechEmbedding(input_dim=80, output_dim=m["d_model"], time_reduction=m["time_reduction"],
                          reduction_method=m["reduction_method"], dropout=0.0).eval()
    x = torch.from_numpy(norm.T.copy()).unsqueeze(0)
    shapes = [tuple(x.unsqueeze(1).shape[1:])]
    with torch.no_grad():
        h = x.unsqueeze(1)
        for layer in emb.cnn.conv:
            h = layer(h)
            if isinstance(layer, torch.nn.Conv2d):
                shapes.append(tuple(h.shape[1:]))
        out, out_len = emb(x, torch.tensor([x.shape[1]]))
    conv = {"shapes": shapes, "out": tuple(out.shape[1:]), "out_len": int(out_len[0]),
            "linear_in": emb.cnn.linear_out.in_features}

    # Tokens: the same transcript under every vocabulary
    tok10 = tokenizers["10k"]
    ids = tok10.encode(text)
    pieces = [tok10.tokenizer.id_to_token(i) for i in ids]
    counts = {k: len(v.encode(text)) for k, v in tokenizers.items()}

    # Masks from hw3lib.model.masks on a real 3-utterance batch
    names = ["103-1240-0000", "103-1240-0010", "103-1240-0002"]
    lens = torch.tensor([np.load(os.path.join(SUBSET, "train-clean-100", "fbank", f"{n}.npy")).shape[1]
                         for n in names])
    sub_lens = emb.calculate_downsampled_length(lens)
    pad = PadMask(torch.zeros(len(names), int(sub_lens.max()), 1), sub_lens).numpy()
    causal = CausalMask(torch.zeros(1, 10, 1)).numpy()
    masks = {"names": names, "lens": lens.tolist(), "sub_lens": sub_lens.tolist(), "pad": pad,
             "causal": causal, "tokens": ["[SOS]"] + [show_token(p) for p in pieces[:9]]}

    return {"cfg": cfg, "raw": raw, "norm": norm, "aug": aug, "f_runs": f_runs, "t_runs": t_runs,
            "seed": seed, "text": text, "pieces": pieces, "counts": counts, "conv": conv,
            "masks": masks}


def load_history():
    data = open(os.path.join(ARTIFACTS, "wandb_export.pkl"), "rb").read()
    unsafe = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"}
    if unsafe & {op.name for op, _, _ in pickletools.genops(data)}:
        raise SystemExit("wandb_export.pkl imports code; refusing to load it")
    run = pickle.loads(data)[0]
    hist = sorted(run["history"], key=lambda r: r["_step"])
    return run, hist


def load_kaggle():
    meta = json.load(open(os.path.join(ARTIFACTS, "kaggle_metadata.json"), encoding="utf-8"))
    subs = [s for s in meta["submissions"] if s.get("public_score") is not None]
    subs.sort(key=lambda s: s["submitted_at"])
    return subs


# --------------------------------------------------------------------------- #
# Small SVG toolkit for the hand-built diagrams
# --------------------------------------------------------------------------- #
class Svg:
    def __init__(self, t, w, h, title, desc):
        self.t, self.w, self.h = t, w, h
        self.out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f"font-family='{FONT}' role=\"img\" aria-labelledby=\"title desc\">",
            f'<title id="title">{esc(title)}</title>',
            f'<desc id="desc">{esc(desc)}</desc>',
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{t["muted"]}"/></marker></defs>',
            f'<rect width="{w}" height="{h}" rx="16" fill="{t["surface"]}"/>',
        ]

    def text(self, x, y, s, size=13, fill=None, weight=400, anchor="middle", mono=False):
        fam = f" font-family='{MONO}'" if mono else ""
        self.out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                        f'fill="{fill or self.t["ink"]}" text-anchor="{anchor}"{fam}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill=None, stroke=None, rx=10, dash=False, sw=1.5, opacity=None):
        d = ' stroke-dasharray="5 4"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                        f'fill="{fill or self.t["panel"]}" stroke="{stroke or self.t["axis"]}" '
                        f'stroke-width="{sw}"{d}{o}/>')

    def line(self, x1, y1, x2, y2, arrow=True, color=None, sw=1.5, dash=False):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color or self.t["muted"]}" stroke-width="{sw}"{m}{d}/>')

    def path(self, d, arrow=True, color=None, sw=1.5, fill="none", dash=False):
        m = ' marker-end="url(#arrow)"' if arrow else ""
        da = ' stroke-dasharray="4 4"' if dash else ""
        self.out.append(f'<path d="{d}" fill="{fill}" stroke="{color or self.t["muted"]}" '
                        f'stroke-width="{sw}"{m}{da}/>')

    def image(self, x, y, w, h, png_bytes):
        b64 = base64.b64encode(png_bytes).decode("ascii")
        self.out.append(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                        f'preserveAspectRatio="none" style="image-rendering:pixelated" '
                        f'href="data:image/png;base64,{b64}"/>')

    def chip(self, x, y, label, fill=None, stroke=None, size=12, mono=True, pad=9, h=24, ink=None):
        w = pad * 2 + len(label) * size * (0.61 if mono else 0.56)
        self.rect(x, y, w, h, fill=fill or self.t["panel"], stroke=stroke or self.t["axis"], rx=6, sw=1.2)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size, ink or self.t["ink"], 600, mono=mono)
        return w

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.out + ["</svg>"]))


def png_of(arr, cmap, vmin, vmax):
    """Encode a 2-D array (rows = mel bins, low frequency first) as a PNG, low bins at the bottom."""
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(arr), cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 1. Architecture
# --------------------------------------------------------------------------- #
def architecture_svg(t, path):
    W, H = 960, 870
    s = Svg(t, W, H, "Encoder-decoder Transformer for speech recognition: data pipeline and model",
            "Audio lane: 80-bin log-mel filterbank, global mean-variance normalisation, SpecAugment during "
            "training, padding into a batch. Text lane: transcript, 10k BPE tokens, shifted and golden copies. "
            "Model: Conv2D speech embedding that shortens time 4 times, 12 pre-norm encoder layers with "
            "bidirectional self-attention, a final LayerNorm, and a CTC head; a tied token embedding, 6 pre-norm "
            "decoder layers with causal self-attention, cross-attention into the encoder output and SwiGLU, a final "
            "LayerNorm and a tied output projection. Loss is cross-entropy plus 0.3 times CTC. 26.47 million parameters.")

    # ---- 1 · data ---------------------------------------------------------- #
    s.text(40, 40, "1 · DATA PIPELINE", 12, t["muted"], 700, "start")
    s.text(40, 64, "audio", 11, t["muted"], 600, "start")
    s.text(40, 168, "text", 11, t["muted"], 600, "start")
    bw, bh, gap = 178, 74, 34
    audio = [("Log-mel filterbank", "(80, T)", "dB, clipped to [−80, 0]", None),
             ("Global MVN", "(x − μ) / σ", "per mel bin, train stats", None),
             ("SpecAugment", "2 F-masks · 4 T-masks", "training batches only", "dash"),
             ("Pad + batch", "(B, T_max, 80)", "+ true lengths", None)]
    text = [("Transcript", "201 characters", "e.g. CHAPTER ONE MISSUS …", None),
            ("10k BPE tokenizer", "48 tokens", "PAD 0 · SOS 3 · EOS 4", None),
            ("Shift for teacher forcing", "(B, L)", "[SOS]+y   /   y+[EOS]", None)]
    for row, (items, y0) in enumerate([(audio, 72), (text, 176)]):
        for i, (title, big, sub, style) in enumerate(items):
            x = 40 + i * (bw + gap)
            dash = style == "dash"
            s.rect(x, y0, bw, bh, fill=t["s2_soft"] if dash else t["panel"],
                   stroke=t["s2"] if dash else t["axis"], dash=dash)
            s.text(x + bw / 2, y0 + 23, title, 12.5, t["ink"], 600)
            s.text(x + bw / 2, y0 + 46, big, 12.5, t["ink"], 700, mono=True)
            s.text(x + bw / 2, y0 + 64, sub, 11, t["ink2"])
            if i < len(items) - 1:
                s.line(x + bw + 3, y0 + bh / 2, x + bw + gap - 4, y0 + bh / 2)

    # ---- 2 · model panel --------------------------------------------------- #
    # The decoder sits on the left and the encoder on the right, so the text lane (which ends
    # further left) and the audio lane (which ends on the right) reach their columns without crossing.
    py = 290
    s.rect(24, py, W - 48, H - py - 24, fill="none", stroke=t["grid"], rx=14)
    s.text(W / 2, py + 28, "2 · MODEL  ·  26.47 M parameters", 12, t["muted"], 700)

    dx, ex, cw = 64, 520, 376          # decoder / encoder column x, column width
    ax_ = 40 + 3 * (bw + gap) + bw / 2  # centre of "Pad + batch"
    s.path(f"M{ax_},{72 + bh + 2} L{ax_},{py - 14} L{ex + cw / 2},{py - 14} L{ex + cw / 2},{py + 48}")
    tx_ = 40 + 2 * (bw + gap) + bw / 2  # centre of "Shift"
    s.path(f"M{tx_},{176 + bh + 2} L{tx_},{py - 4} L{dx + cw / 2},{py - 4} L{dx + cw / 2},{py + 48}")

    def block(x, y, w, h, title, lines, params, accent, mult=None):
        s.rect(x, y, w, h, fill=t["panel"], stroke=accent, sw=1.6)
        s.text(x + 14, y + 22, title, 13, t["ink"], 700, "start")
        if mult:
            s.text(x + w - 14, y + 22, mult, 13, accent, 700, "end", mono=True)
        for k, ln in enumerate(lines):
            s.text(x + 14, y + 44 + k * 19, ln, 11.5, t["ink2"], 400, "start")
        if params:
            s.text(x + w - 14, y + h - 10, params, 11, t["muted"], 600, "end", mono=True)

    enc, dec = t["s1"], t["s2"]
    s.text(dx, py + 42, "DECODER · writes the text", 11, dec, 700, "start")
    s.text(ex, py + 42, "ENCODER · reads the audio", 11, enc, 700, "start")

    def down(x, y0, y1):
        s.line(x, y0 + 2, x, y1 - 4)

    # encoder column (right)
    e1 = py + 52
    block(ex, e1, cw, 92, "Conv2D speech embedding",
          ["2 × [Conv2d 3×3, stride (2, 1) → GELU], 1 → 256 ch",
           "flatten 256 × 76 → Linear 19,456 → 256 · T → T/4"], "5,573,632", enc)
    e2 = e1 + 120
    block(ex, e2, cw, 110, "Encoder layer (pre-norm)",
          ["x + MHSA(RMSNorm(x))   bidirectional, pad mask",
           "x + SwiGLU(RMSNorm(x)) hidden 682",
           "4 heads × 64 · RoPE · QK-norm · dropout 0.1"], "12 × 787,584", enc, "× 12")
    e3 = e2 + 138
    block(ex, e3, cw, 52, "Final LayerNorm → memory (B, T/4, 256)", [], "512", enc)
    e4 = e3 + 80
    block(ex, e4, cw, 70, "CTC head: Linear 256 → 10,000 + LogSoftmax",
          ["frame-wise log-probs (T/4, B, 10,000), blank = id 0"], "2,570,000", enc)
    down(ex + cw / 2, e1 + 92, e2)
    down(ex + cw / 2, e2 + 110, e3)
    down(ex + cw / 2, e3 + 52, e4)

    # decoder column (left)
    d1 = py + 52
    block(dx, d1, cw, 92, "Token embedding",
          ["10,000 × 256 lookup of the shifted transcript",
           "weights shared with the output projection"], "2,560,000", dec)
    d2 = d1 + 120
    block(dx, d2, cw, 150, "Decoder layer (pre-norm)",
          ["x + MHSA(RMSNorm(x))   causal + pad mask",
           "x + CrossAttn(RMSNorm(x), memory)",
           "    Q from the text · K, V from the encoder memory",
           "x + SwiGLU(RMSNorm(x)) hidden 682",
           "4 heads × 64 · RoPE · QK-norm · dropout 0.1"], "6 × 1,051,136", dec, "× 6")
    d3 = d2 + 178
    block(dx, d3, cw, 70, "Final LayerNorm → Linear 256 → 10,000",
          ["logits (B, L, 10,000); weight tied to the embedding"], "512 + 10,000", dec)
    down(dx + cw / 2, d1 + 92, d2)
    down(dx + cw / 2, d2 + 150, d3)

    # encoder memory -> decoder cross-attention (keys and values)
    mx = (dx + cw + ex) / 2
    my, cy = e3 + 26, d2 + 59
    s.path(f"M{ex - 2},{my} L{mx},{my} L{mx},{cy} L{dx + cw + 4},{cy}", color=enc, sw=1.8)
    s.text(mx + 6, (my + cy) / 2 + 4, "K, V", 11, enc, 700, "start", mono=True)

    # losses
    ly = e4 + 92
    down(ex + cw / 2, e4 + 70, ly)
    down(dx + cw / 2, d3 + 70, ly)
    wce = 2 * 8 + len("Cross-entropy, smoothing 0.1") * 12.5 * 0.61
    s.chip(dx + cw / 2 - wce / 2, ly, "Cross-entropy, smoothing 0.1", fill=t["s2_soft"], stroke=dec,
           size=12.5, h=28, pad=8)
    wctc = 2 * 8 + len("CTC loss × 0.3") * 12.5 * 0.61
    s.chip(ex + cw / 2 - wctc / 2, ly, "CTC loss × 0.3", fill=t["s1_soft"], stroke=enc, size=12.5, h=28,
           pad=8)
    s.text(W / 2, ly + 20, "+", 18, t["ink"], 700)
    s.text(W / 2, ly + 58, "Total loss = CE + 0.3 · CTC.   Inference: encode once, then beam search "
           "(width 5) from [SOS] until [EOS].", 12.5, t["ink2"], 600)
    s.save(path)


# --------------------------------------------------------------------------- #
# 2. Features: raw -> global MVN -> SpecAugment
# --------------------------------------------------------------------------- #
def mpl_style():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})


def style_axes(ax, t, grid_y=False):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9.5)
    if grid_y:
        ax.grid(axis="y", color=t["grid"], lw=1)
        ax.set_axisbelow(True)


def features_svg(t, d, path):
    mpl_style()
    raw, norm, aug = d["raw"], d["norm"], d["aug"]
    T = raw.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.4), dpi=100, sharex=True)
    fig.patch.set_facecolor(t["surface"])
    cmap = heat_cmap(t)
    rows = [
        (raw, -80, 0, "Raw log-mel filterbank", "dB"),
        (norm, -2.5, 2.5, "After global mean–variance normalisation (every mel bin → mean 0, std 1)", "z"),
        (aug, -2.5, 2.5, "Training view after SpecAugment: 2 frequency masks (≤ 20 bins) and 4 time masks "
                         "(≤ 50 frames)", "z"),
    ]
    for ax, (arr, lo, hi, title, unit) in zip(axes, rows):
        style_axes(ax, t)
        ax.spines["bottom"].set_visible(False)
        # average frame pairs for display only (the figure is narrower than 1,761 px); keeps files small
        even = arr[:, : arr.shape[1] // 2 * 2]
        shown = 0.5 * (even[:, 0::2] + even[:, 1::2])
        im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="none", extent=(0, T, 0, 80))
        ax.set_yticks([0, 40, 80])
        ax.set_ylabel("mel bin", color=t["ink2"], fontsize=10)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=11, fontweight="bold", pad=6)
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=t["muted"], length=0, labelsize=8.5)
        cb.set_label(unit, color=t["muted"], fontsize=9)
    ax = axes[2]
    for a, b in d["f_runs"]:
        ax.add_patch(plt.Rectangle((0, a), T, b - a, fill=False, ec=t["s2"], lw=1.6, ls=(0, (4, 3))))
    for a, b in d["t_runs"]:
        ax.add_patch(plt.Rectangle((a, 0), b - a, 80, fill=False, ec=t["s2"], lw=1.6, ls=(0, (4, 3))))
    axes[2].set_xlabel("frame", color=t["ink2"], fontsize=10)
    words = d["text"].split()
    fig.suptitle(f"Utterance {UTT}: {T:,} frames × 80 mel bins", x=0.012, y=0.995, ha="left",
                 color=t["ink"], fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.955, "“" + " ".join(words[:14]) + " …”", ha="left", color=t["ink2"], fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Conv2D subsampling
# --------------------------------------------------------------------------- #
def box3d(s, x, y, w, h, depth, t, accent):
    """Front face w×h at (x, y) with a depth offset up-right (channels)."""
    dxy = depth
    s.path(f"M{x},{y} L{x + dxy},{y - dxy} L{x + w + dxy},{y - dxy} L{x + w},{y} Z", arrow=False,
           color=accent, fill=t["s1_soft"], sw=1.2)
    s.path(f"M{x + w},{y} L{x + w + dxy},{y - dxy} L{x + w + dxy},{y + h - dxy} L{x + w},{y + h} Z",
           arrow=False, color=accent, fill=t["s1_soft"], sw=1.2)
    s.rect(x, y, w, h, fill=t["panel"], stroke=accent, rx=2, sw=1.2)


def subsampling_svg(t, d, path):
    c = d["conv"]
    (c0, t0, f0), (c1, t1, f1), (c2, t2, f2) = c["shapes"]
    W, H = 960, 640
    s = Svg(t, W, H, "Conv2D speech embedding: four times shorter in time",
            f"The normalised filterbank, 1 channel by {t0} frames by {f0} bins, passes through two 3 by 3 "
            f"convolutions with stride 2 in time and 1 in frequency, each followed by GELU: {c1} by {t1} by {f1}, "
            f"then {c2} by {t2} by {f2}. Channels and remaining frequency positions are flattened to "
            f"{c['linear_in']} values per step and projected to 256, giving {c['out'][0]} vectors. Each output "
            f"step moves 4 input frames and sees a window of 7 frames.")
    s.text(40, 40, "SHAPES FOR UTTERANCE " + UTT + "  (channels, time, frequency)", 12, t["muted"], 700, "start")
    cmap = heat_cmap(t)
    norm = d["norm"]
    accent = t["s1"]

    scale = 200 / t0
    gap = 128
    y_mid = 150
    # input spectrogram
    x0, w0, h0 = 40, t0 * scale, 80
    s.image(x0, y_mid - h0 / 2, w0, h0, png_of(norm[:, ::4], cmap, -2.5, 2.5))
    s.rect(x0, y_mid - h0 / 2, w0, h0, fill="none", stroke=t["axis"], rx=2, sw=1)
    s.text(x0 + w0 / 2, y_mid + 66, f"({c0}, {t0:,}, {f0})", 12.5, t["ink"], 700, mono=True)
    s.text(x0 + w0 / 2, y_mid + 84, "normalised filterbank", 11, t["ink2"])

    # conv outputs (box width ∝ time, height ∝ frequency, depth = channels)
    x1, w1 = x0 + w0 + gap, t1 * scale
    box3d(s, x1, y_mid - f1 / 2, w1, f1, 18, t, accent)
    s.text(x1 + w1 / 2 + 9, y_mid + 66, f"({c1}, {t1:,}, {f1})", 12.5, t["ink"], 700, mono=True)
    s.text(x1 + w1 / 2 + 9, y_mid + 84, "after conv 1", 11, t["ink2"])
    x2, w2 = x1 + w1 + 18 + gap, t2 * scale
    box3d(s, x2, y_mid - f2 / 2, w2, f2, 18, t, accent)
    s.text(x2 + w2 / 2 + 9, y_mid + 66, f"({c2}, {t2:,}, {f2})", 12.5, t["ink"], 700, mono=True)
    s.text(x2 + w2 / 2 + 9, y_mid + 84, "after conv 2", 11, t["ink2"])
    x3, w3, h3 = x2 + w2 + 18 + gap, c["out"][0] * scale, 46
    s.rect(x3, y_mid - h3 / 2, w3, h3, fill=t["s1_soft"], stroke=accent, rx=2, sw=1.2)
    s.text(x3 + w3 / 2, y_mid + 66, f"({c['out'][0]}, {c['out'][1]})", 12.5, t["ink"], 700, mono=True)
    s.text(x3 + w3 / 2, y_mid + 84, "to the encoder", 11, t["ink2"])

    def arrow_label(xa, xb, l1, l2, l3):
        s.line(xa + 8, y_mid, xb - 10, y_mid)
        s.text((xa + xb) / 2, y_mid - 38, l1, 11, t["ink"], 600)
        s.text((xa + xb) / 2, y_mid - 23, l2, 11, t["ink2"], 400)
        s.text((xa + xb) / 2, y_mid - 9, l3, 11, t["ink2"], 400)

    arrow_label(x0 + w0, x1, "Conv2d 3×3", "1 → 256 channels", "stride (2, 1), GELU")
    arrow_label(x1 + w1 + 18, x2, "Conv2d 3×3", "256 → 256 channels", "stride (2, 1), GELU")
    arrow_label(x2 + w2 + 18, x3, "flatten + Linear", f"{c2} × {f2} = {c['linear_in']:,}", "→ 256 per step")

    s.text(40, 262, f"Time: {t0:,} → {t1:,} → {t2:,} frames (no padding, so each conv gives ⌊(T − 3) / 2⌋ + 1)."
           f"  Frequency: {f0} → {f1} → {f2}, then folded into the channels.", 11.5, t["ink2"], 400, "start")
    s.text(40, 281, "Self-attention cost grows with T², so a 4× shorter sequence makes every encoder layer "
           "about 16× cheaper.", 11.5, t["ink2"], 400, "start")

    # ---- receptive field zoom ---------------------------------------------- #
    s.text(40, 324, "RECEPTIVE FIELD  ·  one encoder step = 4 frames apart, 7 frames wide", 12, t["muted"],
           700, "start")
    nsteps = 5
    span = 4 * (nsteps - 1) + 7
    energy = d["raw"].mean(axis=0)          # zoom on the loudest (most voiced) stretch
    j0 = max(range(0, (len(energy) - span) // 4), key=lambda j: energy[4 * j:4 * j + span].mean())
    f_start, f_end = 4 * j0, 4 * j0 + span
    nframes = f_end - f_start
    zx, zy, zw, zh = 40, 352, 880, 110
    fw = zw / nframes
    s.image(zx, zy, zw, zh, png_of(norm[:, f_start:f_end], cmap, -2.5, 2.5))
    s.rect(zx, zy, zw, zh, fill="none", stroke=t["axis"], rx=2, sw=1)
    for k in range(nframes + 1):
        if (f_start + k) % 4 == 0:
            s.line(zx + k * fw, zy + zh, zx + k * fw, zy + zh + 6, arrow=False, color=t["axis"], sw=1)
            s.text(zx + k * fw, zy + zh + 19, str(f_start + k), 10, t["muted"], 400, mono=True)
    for i in range(nsteps):
        j = j0 + i
        a, b = 4 * j - f_start, 4 * j + 7 - f_start
        yb = zy + zh + 36 + (i % 2) * 40
        col = t["s2"] if i == 2 else accent
        s.path(f"M{zx + a * fw + 2},{yb - 7} L{zx + a * fw + 2},{yb} L{zx + b * fw - 2},{yb} "
               f"L{zx + b * fw - 2},{yb - 7}", arrow=False, color=col, sw=1.8)
        s.text(zx + (a + b) / 2 * fw, yb + 16, f"output step {j}", 11, t["ink"] if i == 2 else t["ink2"],
               700 if i == 2 else 500)
    s.text(zx, zy - 8, f"frames {f_start}–{f_end - 1} of the normalised utterance (all 80 bins)", 11,
           t["ink2"], 400, "start")
    s.text(40, H - 28, "Each bracket is the span of input frames one output vector depends on: conv 1 sees 3 "
           "frames, conv 2 sees 3 conv-1 outputs that are 2 frames apart, so 3 + 2·2 = 7 frames.", 11.5,
           t["ink2"], 400, "start")
    s.save(path)


# --------------------------------------------------------------------------- #
# 4. Tokens and teacher forcing
# --------------------------------------------------------------------------- #
def tokens_svg(t, d, path):
    pieces = [show_token(p) for p in d["pieces"]]
    counts = d["counts"]
    W, H = 960, 560
    s = Svg(t, W, H, "From transcript to decoder targets",
            f"The 201-character transcript of {UTT} becomes {counts['char']} character tokens, {counts['1k']} with "
            f"the 1k BPE vocabulary, {counts['5k']} with 5k and {counts['10k']} with 10k. The 10k pieces are shown, "
            "followed by teacher forcing: the decoder reads [SOS] plus the tokens and must predict the tokens "
            "plus [EOS], one position ahead.")
    s.text(40, 40, "SAME TRANSCRIPT, FOUR VOCABULARIES  ·  tokens needed", 12, t["muted"], 700, "start")
    order = [("char", "characters"), ("1k", "1k BPE"), ("5k", "5k BPE"), ("10k", "10k BPE (used)")]
    bx, maxw = 170, 560
    for i, (k, label) in enumerate(order):
        y = 58 + i * 30
        used = k == "10k"
        s.text(bx - 12, y + 15, label, 12, t["ink"] if used else t["ink2"], 700 if used else 500, "end")
        w = maxw * counts[k] / counts["char"]
        s.rect(bx, y, w, 20, fill=t["s1"] if used else t["neutral"], stroke=t["s1"] if used else t["axis"],
               rx=4, sw=1)
        s.text(bx + w + 8, y + 15, f"{counts[k]}", 12, t["ink"], 700, "start", mono=True)
    s.text(520, 126, "Fewer tokens mean shorter decoder sequences", 11.5, t["ink2"], 400, "start")
    s.text(520, 144, "and fewer decoding steps at inference.", 11.5, t["ink2"], 400, "start")

    s.text(40, 212, f"THE {counts['10k']} TOKENS  ·  “·” marks a token that starts a new word", 12, t["muted"],
           700, "start")
    x, y = 40, 226
    for p in pieces:
        w = 16 + len(p) * 12 * 0.61
        if x + w > W - 40:
            x, y = 40, y + 32
        s.chip(x, y, p, size=12, pad=8)
        x += w + 6

    y0 = y + 62
    s.text(40, y0, "TEACHER FORCING  ·  the decoder reads one row and is trained to output the other", 12,
           t["muted"], 700, "start")
    shifted = ["[SOS]"] + pieces
    golden = pieces + ["[EOS]"]
    show = list(range(6)) + [None] + [len(shifted) - 2, len(shifted) - 1]
    cx, colw = 196, 80
    s.text(cx - 14, y0 + 38, "decoder input", 12, t["ink"], 600, "end")
    s.text(cx - 14, y0 + 54, "shifted", 11, t["muted"], 400, "end", mono=True)
    s.text(cx - 14, y0 + 112, "target", 12, t["ink"], 600, "end")
    s.text(cx - 14, y0 + 128, "golden", 11, t["muted"], 400, "end", mono=True)
    for k, pos in enumerate(show):
        xc = cx + k * colw
        if pos is None:
            s.text(xc + colw / 2 - 6, y0 + 84, "…", 16, t["muted"], 700)
            continue
        for row, (seq, yy, fill, stroke) in enumerate(
                [(shifted, y0 + 26, t["s2_soft"], t["s2"]), (golden, y0 + 100, t["panel"], t["axis"])]):
            lab = seq[pos]
            w = 16 + len(lab) * 12 * 0.61
            s.chip(xc + (colw - 8 - w) / 2, yy, lab, fill=fill, stroke=stroke, size=12, pad=8)
        s.line(xc + (colw - 8) / 2, y0 + 54, xc + (colw - 8) / 2, y0 + 96)
        s.text(xc + (colw - 8) / 2, y0 + 150, f"t={pos}", 10, t["muted"], 400, mono=True)
    s.text(40, H - 26, f"Both rows have {len(shifted)} positions. Padding uses PAD (id 0), which the "
           "cross-entropy ignores; a causal mask stops position t from seeing inputs after t.", 11.5,
           t["ink2"], 400, "start")
    s.save(path)


# --------------------------------------------------------------------------- #
# 5. Masks
# --------------------------------------------------------------------------- #
def masks_svg(t, d, path):
    mpl_style()
    m = d["masks"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([t["s1"], t["neutral"]])  # False = attend, True = masked
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=100,
                                 gridspec_kw={"width_ratios": [2.3, 1]})
    fig.patch.set_facecolor(t["surface"])
    for ax in (a1, a2):
        style_axes(ax, t)
        ax.spines["bottom"].set_visible(False)
    pad = m["pad"]
    a1.imshow(pad, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
    a1.set_yticks(range(len(m["names"])))
    a1.set_yticklabels([f"{n}  ({L:,} → {s_})" for n, L, s_ in zip(m["names"], m["lens"], m["sub_lens"])],
                       fontsize=9.5, color=t["ink2"])
    a1.set_xlabel("encoder position after 4× subsampling", color=t["ink2"], fontsize=10)
    a1.set_title("Encoder pad mask: one row per utterance in a batch", loc="left", color=t["ink"],
                 fontsize=11, fontweight="bold")
    for i in range(1, len(m["names"])):
        a1.axhline(i - 0.5, color=t["surface"], lw=3)
    caus = m["causal"]
    a2.imshow(caus, cmap=cmap, vmin=0, vmax=1, interpolation="none")
    n = caus.shape[0]
    a2.set_xticks(range(n))
    a2.set_yticks(range(n))
    a2.set_xticklabels(m["tokens"][:n], rotation=90, fontsize=8.5, color=t["ink2"],
                       fontfamily="monospace")
    a2.set_yticklabels(m["tokens"][:n], fontsize=8.5, color=t["ink2"], fontfamily="monospace")
    for k in range(n + 1):
        a2.axhline(k - 0.5, color=t["surface"], lw=1.5)
        a2.axvline(k - 0.5, color=t["surface"], lw=1.5)
    a2.set_title("Decoder causal mask", loc="left", color=t["ink"], fontsize=11, fontweight="bold")
    a2.set_xlabel("key (token attended to)", color=t["ink2"], fontsize=9.5)
    a2.set_ylabel("query", color=t["ink2"], fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["s1"]), plt.Rectangle((0, 0), 1, 1, color=t["neutral"])]
    leg = fig.legend(handles, ["can be attended to", "masked (True)"], loc="lower left", ncol=2,
                     frameon=False, fontsize=10, bbox_to_anchor=(0.01, -0.01))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Training curves (W&B run export)
# --------------------------------------------------------------------------- #
def training_svg(t, hist, path):
    mpl_style()
    ep = np.arange(1, len(hist) + 1)
    ce = np.array([h["train/ce_loss"] for h in hist])
    ctc = np.array([h["train/ctc_loss"] for h in hist])
    wer = np.array([h["val/wer"] for h in hist])
    cer = np.array([h["val/cer"] for h in hist])
    lr = np.array([h["learning_rate"] for h in hist])

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.2), dpi=100, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25, 0.6]})
    fig.patch.set_facecolor(t["surface"])
    a1, a2, a3 = axes
    for ax in axes:
        style_axes(ax, t, grid_y=True)

    a1.plot(ep, ce, color=t["s1"], lw=2, label="cross-entropy (label smoothing 0.1)")
    a1.plot(ep, ctc, color=t["s2"], lw=2, label="CTC")
    a1.set_ylim(0, 4)
    a1.annotate(f"{ce[-1]:.2f}", (ep[-1], ce[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=t["ink"], fontsize=10, fontweight="bold")
    a1.annotate(f"{ctc[-1]:.2f}", (ep[-1], ctc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=t["ink"], fontsize=10, fontweight="bold")
    a1.text(14, 3.1, f"off scale before epoch 4: CE {ce[0]:.1f} and CTC {ctc[0]:.1f} at epoch 1",
            color=t["muted"], fontsize=9.5)
    a1.set_title("Training loss per epoch (mean over batches)", loc="left", color=t["ink"],
                 fontsize=12, fontweight="bold", pad=8)
    leg = a1.legend(loc="upper right", frameon=False, fontsize=10)
    [x.set_color(t["ink2"]) for x in leg.get_texts()]

    a2.plot(ep, wer, color=t["s1"], lw=2, label="WER")
    a2.plot(ep, cer, color=t["s2"], lw=2, label="CER")
    a2.set_ylim(0, 40)
    for arr, col, off in [(wer, t["s1"], (-10, 12)), (cer, t["s2"], (-10, -20))]:
        b = int(np.argmin(arr))
        a2.plot(ep[b], arr[b], "o", ms=8, color=col, mec=t["surface"], mew=2, zorder=5)
        a2.annotate(f"{arr[b]:.2f} % (epoch {ep[b]})", (ep[b], arr[b]), xytext=off, ha="right",
                    textcoords="offset points", color=t["ink"], fontsize=10, fontweight="bold")
    a2.text(14, 31, f"off scale for epochs 1–3: WER {wer[0]:.0f}–{wer[2]:.0f} % before cross-attention aligns",
            color=t["muted"], fontsize=9.5)
    a2.set_ylabel("%", color=t["ink2"], fontsize=10)
    a2.set_title("Dev-clean error rate per epoch (first 5 batches = 160 utterances, greedy decoding)",
                 loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    leg = a2.legend(loc="upper right", frameon=False, fontsize=10)
    [x.set_color(t["ink2"]) for x in leg.get_texts()]

    a3.plot(ep, lr * 1e3, color=t["s1"], lw=2)
    a3.set_ylabel("× 10⁻³", color=t["ink2"], fontsize=10)
    a3.set_title("Learning rate at the end of each epoch: 4-epoch linear warmup, cosine decay to 1e-6",
                 loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    a3.set_xlabel("epoch", color=t["ink2"], fontsize=10)
    a3.set_xlim(0, len(hist) + 4)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Kaggle public-leaderboard history
# --------------------------------------------------------------------------- #
def kaggle_svg(t, subs, path):
    mpl_style()
    # one point per distinct (score, label) result; identical re-uploads collapse into the first one
    seen, pts = set(), []
    for sub in subs:
        key = (round(sub["public_score"], 5), sub["description"].split("|")[0])
        if round(sub["public_score"], 5) in {p[1] for p in pts} and sub["submitted_at"] >= "2026-11-06":
            continue  # re-uploads of the final predictions with the W&B tag
        if key in seen:
            continue
        seen.add(key)
        when = dt.datetime.fromisoformat(sub["submitted_at"])
        pts.append((when, round(sub["public_score"], 5), sub["description"]))

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style_axes(ax, t, grid_y=True)
    top = 7.2
    best_so_far, run_x, run_y = None, [], []
    for when, score, _ in pts:
        if score <= top:
            best_so_far = score if best_so_far is None else min(best_so_far, score)
        if best_so_far is not None:
            run_x.append(when)
            run_y.append(best_so_far)
    ax.step(run_x, run_y, where="post", color=t["s1"], lw=2, alpha=0.9, label="best so far")
    best = min(p[1] for p in pts)
    off = [p for p in pts if p[1] > top]
    for k, (when, score, label) in enumerate(off):
        ax.plot(when, top - 0.03, "^", ms=9, color=t["s2"], mec=t["surface"], mew=1.5, zorder=5)
        ax.annotate(f"{score:.1f}", (when, top - 0.03), xytext=(0, -16 - 13 * (k % 2)),
                    textcoords="offset points", ha="center", color=t["ink"], fontsize=9.5, fontweight="bold")
    if off:
        short = {"Baseline_1k_tokenizer_splitter": "1k-token run"}
        ax.text(0.99, 0.80, "off scale (labels as submitted): " + "; ".join(
            f"{w:%b %d} {s_:.1f} {short.get(lab, lab)}" for w, s_, lab in off),
            transform=ax.transAxes, ha="right", color=t["muted"], fontsize=9)
    for when, score, label in pts:
        if score > top:
            continue
        if score == best:
            ax.plot(when, score, "D", ms=9, color=t["s3"], mec=t["surface"], mew=2, zorder=6)
            ax.annotate(f"{score:.3f} %  best (W&B run yr0gfal7)", (when, score), xytext=(-10, -18),
                        textcoords="offset points", ha="right", color=t["ink"], fontsize=10.5,
                        fontweight="bold")
        else:
            ax.plot(when, score, "o", ms=8, color=t["s1"], mec=t["surface"], mew=2, zorder=5)
            ax.annotate(f"{score:.2f}", (when, score), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=t["ink"], fontsize=9.5)
    ax.set_ylim(6.0, top)
    ax.set_ylabel("public WER (%) · lower is better", color=t["ink2"], fontsize=10)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_title("Kaggle public leaderboard: every distinct result (test-clean, public split)",
                 loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=10)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    [x.set_color(t["ink2"]) for x in leg.get_texts()]
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    data = load_inputs()
    _, hist = load_history()
    subs = load_kaggle()
    for mode, t in THEMES.items():
        architecture_svg(t, os.path.join(ASSETS, f"architecture-{mode}.svg"))
        features_svg(t, data, os.path.join(ASSETS, f"features-{mode}.svg"))
        subsampling_svg(t, data, os.path.join(ASSETS, f"subsampling-{mode}.svg"))
        tokens_svg(t, data, os.path.join(ASSETS, f"tokens-{mode}.svg"))
        masks_svg(t, data, os.path.join(ASSETS, f"masks-{mode}.svg"))
        training_svg(t, hist, os.path.join(ASSETS, f"training-{mode}.svg"))
        kaggle_svg(t, subs, os.path.join(ASSETS, f"kaggle-{mode}.svg"))
    print(f"Figures written to {ASSETS} (SpecAugment seed {data['seed']}, "
          f"masks F{data['f_runs']} T{data['t_runs']})")


if __name__ == "__main__":
    main()
