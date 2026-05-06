"""
Compose: ParaView raw image + matplotlib legend panel + overlap callouts.

讀取 outline_walls.py 輸出的:
  - output/outline_walls_raw.png  (乾淨 3D 圖)
  - output/extrema_projected.json (極值 2D 投影座標)
搭配 zplus_summary 用 matplotlib 繪製:
  - 右側圖例面板
  - 重疊極值摺線標註 (斜線 → 橫線 → 堆疊變數)

用法:
  python compose_outline.py
"""

import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    'text.usetex':           True,
    'text.latex.preamble':   r'\usepackage{amsmath,amssymb}',
    'font.family':           'serif',
})

# ============================================================
# 設定區
# ============================================================
BG_COLOR = [1, 1, 1]
IMG_SIZE = (26, 10)
IMG_DPI  = 200

EXTREMA_COLORS = {
    'u_tau_local_max':  [0.80, 0.00, 0.00],
    'u_tau_local_min':  [0.00, 0.50, 0.00],
    'delta_y_plus_max': [0.80, 0.68, 0.00],
    'delta_y_plus_min': [0.50, 0.00, 0.65],
    'delta_z_plus_max': [0.85, 0.35, 0.00],
    'delta_z_plus_min': [0.00, 0.00, 0.65],
}

EXTREMA_LATEX = {
    'u_tau_local_max':  r'$u_{\tau,\,\mathrm{local},\,\max}$',
    'u_tau_local_min':  r'$u_{\tau,\,\mathrm{local},\,\min}$',
    'delta_y_plus_max': r'$\Delta y^{+}_{\max}$',
    'delta_y_plus_min': r'$\Delta y^{+}_{\min}$',
    'delta_z_plus_max': r'$\Delta z^{+}_{\max}$',
    'delta_z_plus_min': r'$\Delta z^{+}_{\min}$',
}

VAR_ORDER = [
    'u_tau_local_max', 'u_tau_local_min',
    'delta_y_plus_max', 'delta_y_plus_min',
    'delta_z_plus_max', 'delta_z_plus_min',
]

# ---- 摺線標註參數 ----
CALLOUT_DIAG_DX   = 180    # 斜線水平分量 (px)
CALLOUT_DIAG_DY   = 115    # 斜線垂直分量 (px), 正=向下
CALLOUT_HORIZ_DX  = 140    # 橫線長度 (px)
CALLOUT_LINE_W    = 1.0
CALLOUT_LINE_C    = [0.30, 0.30, 0.30]
CALLOUT_DOT_SIZE  = 5
CALLOUT_FONT_SIZE = 7.5
CALLOUT_TEXT_GAP  = 28     # 每行文字間距 (px)


# ============================================================
# 解析 zplus_summary (僅取 k 值用於標題)
# ============================================================
def parse_wall_k_indices(filepath):
    k_bottom, k_top = 0, None
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if 'Bottom Wall' in s:
                m = re.search(r'k\s*=\s*(\d+)', s)
                if m:
                    k_bottom = int(m.group(1))
            elif 'Top Wall' in s:
                m = re.search(r'k\s*=\s*(\d+)', s)
                if m:
                    k_top = int(m.group(1))
    return k_bottom, k_top


# ============================================================
# 格式化數值 (LaTeX)
# ============================================================
def format_value_latex(v):
    if v == 0:
        return "0"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    exp = int(np.floor(np.log10(abs(v))))
    m   = v / (10 ** exp)
    return f"{m:.4f}" + r"\times 10^{" + str(exp) + "}"


# ============================================================
# 建構圖例面板
# ============================================================
def build_legend_panel(ax_leg, projected, k_bottom, k_top):
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    x_marker = 0.06
    x_name   = 0.14
    x_eq     = 0.54
    x_val    = 0.59

    sections = [
        ('bottom', r'\textbf{Bottom Wall}\ $(k = %d)$' % k_bottom),
        ('top',    r'\textbf{Top Wall}\ $(k = %s)$' % (k_top if k_top is not None else '?')),
    ]

    row_h   = 0.062
    sec_gap = 0.070
    fs      = 13

    total_rows = 0
    for sec_key, _ in sections:
        n_ext = sum(1 for v in VAR_ORDER
                    if any(e['name'] == v for e in projected[sec_key]))
        total_rows += 1 + n_ext
    total_h = total_rows * row_h + (len(sections) - 1) * sec_gap
    y = 0.50 + total_h / 2.0 + 0.02

    for s_idx, (sec_key, sec_title) in enumerate(sections):
        if s_idx > 0:
            sep_y = y + sec_gap * 0.45
            ax_leg.plot([0.02, 0.98], [sep_y, sep_y],
                        color=[0.45, 0.45, 0.45], linewidth=0.8)

        ax_leg.text(0.50, y, sec_title,
                    fontsize=fs + 1, ha='center', va='center')
        ax_leg.plot([0.02, 0.12], [y, y],
                    color=[0.5, 0.5, 0.5], linewidth=0.6)
        ax_leg.plot([0.88, 0.98], [y, y],
                    color=[0.5, 0.5, 0.5], linewidth=0.6)
        y -= row_h

        by_name = {e['name']: e for e in projected[sec_key]}
        for vname in VAR_ORDER:
            if vname not in by_name:
                continue
            ext   = by_name[vname]
            color = EXTREMA_COLORS[vname]

            ax_leg.scatter(x_marker, y, s=60, c=[color],
                           marker='o', edgecolors='black',
                           linewidths=0.6, zorder=5, clip_on=False)
            ax_leg.text(x_name, y, EXTREMA_LATEX[vname],
                        fontsize=fs, ha='left', va='center')
            ax_leg.text(x_eq, y, r'$=$',
                        fontsize=fs, ha='center', va='center')
            val_str = format_value_latex(ext['value'])
            ax_leg.text(x_val, y, r'$%s$' % val_str,
                        fontsize=fs, ha='left', va='center')
            y -= row_h

        y -= sec_gap


# ============================================================
# 找出重疊群組 (同一 px, py 的極值點)
# ============================================================
def find_overlap_groups(projected):
    groups = []
    for section in ['bottom', 'top']:
        seen = {}
        for ext in projected[section]:
            key = (round(ext['px'], 3), round(ext['py'], 3))
            seen.setdefault(key, []).append(ext)
        for entries in seen.values():
            if len(entries) > 1:
                groups.append({
                    'section': section,
                    'px':      entries[0]['px'],
                    'py':      entries[0]['py'],
                    'entries': entries,
                })
    return groups


# ============================================================
# 繪製重疊摺線標註 (斜線 → 橫線 → 堆疊變數)
# ============================================================
def draw_overlap_callouts(ax_img, projected, img_shape):
    img_h, img_w = img_shape[:2]
    groups = find_overlap_groups(projected)

    if not groups:
        print("  No overlap groups found.")
        return

    groups.sort(key=lambda g: g['py'], reverse=True)

    for i, grp in enumerate(groups):
        anchor_x = grp['px'] * img_w
        anchor_y = (1.0 - grp['py']) * img_h

        sign_y = -1 if grp['py'] > 0.5 else +1

        elbow_x = anchor_x + CALLOUT_DIAG_DX
        elbow_y = anchor_y + sign_y * CALLOUT_DIAG_DY

        end_x = elbow_x + CALLOUT_HORIZ_DX
        end_y = elbow_y

        ax_img.plot([anchor_x, elbow_x], [anchor_y, elbow_y],
                    color=CALLOUT_LINE_C, linewidth=CALLOUT_LINE_W,
                    solid_capstyle='round', zorder=20)
        ax_img.plot([elbow_x, end_x], [elbow_y, end_y],
                    color=CALLOUT_LINE_C, linewidth=CALLOUT_LINE_W,
                    solid_capstyle='round', zorder=20)

        ax_img.plot(anchor_x, anchor_y, 'o',
                    color='black', markersize=CALLOUT_DOT_SIZE,
                    zorder=25)

        entries = grp['entries']
        entries_sorted = sorted(entries,
                                key=lambda e: VAR_ORDER.index(e['name'])
                                if e['name'] in VAR_ORDER else 99)
        n = len(entries_sorted)

        text_x = end_x + 8
        for j, ext in enumerate(entries_sorted):
            color = EXTREMA_COLORS[ext['name']]
            label = EXTREMA_LATEX[ext['name']]
            val   = format_value_latex(ext['value'])

            text_y = end_y + (j - (n - 1) / 2.0) * CALLOUT_TEXT_GAP

            ax_img.text(text_x, text_y,
                        r'%s $= %s$' % (label, val),
                        fontsize=CALLOUT_FONT_SIZE,
                        color=color, va='center', ha='left',
                        zorder=30)

        sec_label = 'Bot' if grp['section'] == 'bottom' else 'Top'
        names = ', '.join(e['name'] for e in entries_sorted)
        print(f"  [{sec_label}] {n} vars @ ({grp['px']:.3f}, {grp['py']:.3f}): {names}")


# ============================================================
# 主程式
# ============================================================
def compose(raw_img_path, proj_path, summary_path, output_path):
    img = plt.imread(str(raw_img_path))
    with open(str(proj_path), 'r') as f:
        projected = json.load(f)

    k_bottom, k_top = parse_wall_k_indices(str(summary_path))

    print(f"Image: {raw_img_path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Bottom: {len(projected['bottom'])} pts,  Top: {len(projected['top'])} pts")

    fig = plt.figure(figsize=IMG_SIZE)
    fig.patch.set_facecolor(BG_COLOR)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1], wspace=0.02)

    ax_img = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])

    ax_img.imshow(img)
    ax_img.axis('off')

    build_legend_panel(ax_leg, projected, k_bottom, k_top)

    print("Overlap callouts:")
    draw_overlap_callouts(ax_img, projected, img.shape)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.02)
    plt.savefig(str(output_path), dpi=IMG_DPI,
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"Output: {output_path}")


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent

    raw_img  = sys.argv[1] if len(sys.argv) > 1 else str(SCRIPT_DIR / "output" / "outline_walls_raw.png")
    proj     = sys.argv[2] if len(sys.argv) > 2 else str(SCRIPT_DIR / "output" / "extrema_projected.json")
    summary  = sys.argv[3] if len(sys.argv) > 3 else str(SCRIPT_DIR / "input"  / "14.Re1400_zplus_summary.txt")
    out_file = sys.argv[4] if len(sys.argv) > 4 else str(SCRIPT_DIR / "output" / "outline_walls.png")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    compose(raw_img, proj, summary, out_file)
