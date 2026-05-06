"""
Plot the y-z plane grid at a given i-index from a VTK STRUCTURED_GRID file.
With wall/center boundary lines and extrema markers from zplus_summary.

繪圖邏輯對齊 outline_walls.py:
  - 逐段建構 line segments (水平 j 方向 + 垂直 k 方向)
  - 節點以圓形 marker 標示
  - 可調參數集中於設定區
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from pathlib import Path

plt.rcParams.update({
    'text.usetex':           True,
    'text.latex.preamble':   r'\usepackage{amsmath,amssymb}',
    'font.family':           'serif',
})

# ============================================================
# 設定區  (對齊 outline_walls.py 參數命名)
# ============================================================
BG_COLOR     = [1, 1, 1]
IMG_SIZE     = (22, 10)
IMG_DPI      = 200

NODE_RADIUS  = 1.5
NODE_COLOR   = [0.0, 0.0, 0.0]
NODE_OPACITY = 1.0

GRID_COLOR   = [0.5, 0.5, 0.5]
GRID_OPACITY = 0.6
GRID_WIDTH   = 0.3

WALL_LINE_COLOR   = [0.0, 0.0, 1.0]
WALL_LINE_OPACITY = 0.7
WALL_LINE_WIDTH   = 2.5

CENTER_LINE_COLOR   = [1.0, 0.0, 0.0]
CENTER_LINE_OPACITY = 0.7
CENTER_LINE_WIDTH   = 2.5

EXTREMA_MARKER_SIZE = 120
EXTREMA_EDGE_WIDTH  = 0.8

EXTREMA_COLORS = {
    'delta_y_max': [0.80, 0.68, 0.00],
    'delta_y_min': [0.50, 0.00, 0.65],
    'delta_z_max': [0.85, 0.35, 0.00],
    'delta_z_min': [0.00, 0.00, 0.65],
}

SECTION_MARKERS = {
    'bottom': 'v',
    'center': 'o',
    'top':    '^',
}

EXTREMA_LATEX = {
    'delta_y_max': r'$\Delta y_{\max}$',
    'delta_y_min': r'$\Delta y_{\min}$',
    'delta_z_max': r'$\Delta z_{\max}$',
    'delta_z_min': r'$\Delta z_{\min}$',
}

VAR_ORDER = ['delta_y_max', 'delta_y_min',
             'delta_z_max', 'delta_z_min']


# ============================================================
# 共用函數: 讀取 VTK STRUCTURED_GRID
# ============================================================
def read_vtk_structured_grid(filepath):
    """Parse an ASCII VTK STRUCTURED_GRID file, return dims and 3D coords."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    dims = None
    n_points = None
    point_start = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DIMENSIONS"):
            parts = stripped.split()
            dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        elif stripped.startswith("POINTS"):
            parts = stripped.split()
            n_points = int(parts[1])
            point_start = idx + 1
            break

    if dims is None or n_points is None:
        raise ValueError("Could not parse DIMENSIONS or POINTS from VTK file.")

    coords = []
    for i in range(point_start, point_start + n_points):
        vals = lines[i].strip().split()
        coords.append([float(vals[0]), float(vals[1]), float(vals[2])])

    coords = np.array(coords)
    nx, ny, nz = dims
    coords_3d = coords.reshape((nz, ny, nx, 3))
    return dims, coords_3d


# ============================================================
# 共用函數: 解析 zplus_summary
# ============================================================
def parse_zplus_summary(filepath):
    """Parse zplus summary, return extrema dict and k_center."""
    extrema = {'bottom': [], 'center': [], 'top': []}
    k_center = None
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()

            if s.startswith('k_center'):
                k_center = int(s.split('=')[1].split('#')[0].strip())

            if 'Bottom Wall' in s:
                current_section = 'bottom'
                continue
            elif 'Top Wall' in s:
                current_section = 'top'
                continue
            elif 'Center Node' in s:
                current_section = 'center'
                continue
            elif 'Full-grid' in s:
                current_section = None
                continue

            if current_section is None:
                continue

            # Wall format: (i=, j=, k=)  x=  y=  z=
            m = re.match(
                r'(delta_[yz]_(?:max|min))\s*=\s*([\d.eE+-]+)\s+at\s+'
                r'\(i=(\d+),\s*j=(\d+),\s*k=(\d+)\)\s+'
                r'x=([+\-\d.eE]+)\s+y=([+\-\d.eE]+)\s+z=([+\-\d.eE]+)',
                s)
            if m:
                extrema[current_section].append({
                    'name':  m.group(1),
                    'value': float(m.group(2)),
                    'i': int(m.group(3)),
                    'j': int(m.group(4)),
                    'k': int(m.group(5)),
                    'y': float(m.group(7)),
                    'z': float(m.group(8)),
                })
                continue

            # Center format: (j=, k=)  y=  z=
            m2 = re.match(
                r'(delta_[yz]_(?:max|min))\s*=\s*([\d.eE+-]+)\s+at\s+'
                r'\(j=(\d+),\s*k=(\d+)\)\s+'
                r'y=([+\-\d.eE]+)\s+z=([+\-\d.eE]+)',
                s)
            if m2:
                extrema[current_section].append({
                    'name':  m2.group(1),
                    'value': float(m2.group(2)),
                    'j': int(m2.group(3)),
                    'k': int(m2.group(4)),
                    'y': float(m2.group(5)),
                    'z': float(m2.group(6)),
                })

    return extrema, k_center


# ============================================================
# 共用函數: 建構 y-z 平面格點 + 網格線段
# ============================================================
def build_plane_grid(coords_3d, dims, i_index):
    """
    從 3D 結構化網格取出 i=i_index 的 y-z 平面,
    建構所有節點座標與線段 (對齊 outline_walls.py 的 build_wall_grid).
    """
    nx, ny, nz = dims
    if i_index < 0 or i_index >= nx:
        raise ValueError(f"i_index={i_index} out of range [0, {nx - 1}]")

    plane  = coords_3d[:, :, i_index, :]
    y_vals = plane[:, :, 1]
    z_vals = plane[:, :, 2]

    h_segments = []
    for k in range(nz):
        for j in range(ny - 1):
            h_segments.append([(y_vals[k, j],   z_vals[k, j]),
                               (y_vals[k, j+1], z_vals[k, j+1])])

    v_segments = []
    for j in range(ny):
        for k in range(nz - 1):
            v_segments.append([(y_vals[k, j],   z_vals[k, j]),
                               (y_vals[k+1, j], z_vals[k+1, j])])

    print(f"  Built: {nz*ny} pts, "
          f"{len(h_segments)} h-lines, {len(v_segments)} v-lines")
    return y_vals, z_vals, h_segments, v_segments


# ============================================================
# 共用函數: 繪製網格 overlay (對齊 add_grid_overlay)
# ============================================================
def add_grid_overlay(ax, y_vals, z_vals, h_segments, v_segments):
    all_segments = h_segments + v_segments
    lc = LineCollection(all_segments,
                        colors=[GRID_COLOR],
                        linewidths=GRID_WIDTH,
                        alpha=GRID_OPACITY)
    ax.add_collection(lc)

    ax.scatter(y_vals.ravel(), z_vals.ravel(),
               s=NODE_RADIUS,
               c=[NODE_COLOR],
               alpha=NODE_OPACITY,
               edgecolors="none",
               zorder=5)


# ============================================================
# 共用函數: 壁面 / 中心線
# ============================================================
def add_boundary_lines(ax, coords_3d, dims, i_index, k_center):
    nx, ny, nz = dims

    bottom = coords_3d[0, :, i_index, :]
    ax.plot(bottom[:, 1], bottom[:, 2],
            color=WALL_LINE_COLOR, linewidth=WALL_LINE_WIDTH,
            alpha=WALL_LINE_OPACITY, zorder=10)

    center = coords_3d[k_center, :, i_index, :]
    ax.plot(center[:, 1], center[:, 2],
            color=CENTER_LINE_COLOR, linewidth=CENTER_LINE_WIDTH,
            alpha=CENTER_LINE_OPACITY, zorder=10)

    top = coords_3d[nz - 1, :, i_index, :]
    ax.plot(top[:, 1], top[:, 2],
            color=WALL_LINE_COLOR, linewidth=WALL_LINE_WIDTH,
            alpha=WALL_LINE_OPACITY, zorder=10)

    print(f"  Boundary lines: k=0 (bottom), k={k_center} (center), k={nz-1} (top)")


# ============================================================
# 共用函數: 將右側極值鏡像至左側對稱點
# ============================================================
def mirror_extrema_to_left(extrema, coords_3d, dims, i_index):
    """
    網格左右對稱 (dy[j] ≈ dy[ny-2-j]), 但 summary 只報告
    第一個全域最小值, 可能落在右側. 此函式將 y > y_mid 的
    極值搬到左側對稱格點, 座標從實際網格讀取.
    """
    nx, ny, nz = dims
    j_mid = (ny - 1) // 2

    for section, entries in extrema.items():
        for ext in entries:
            j = ext.get('j')
            if j is None or j <= j_mid:
                continue
            j_sym = (ny - 1) - j
            k = ext.get('k', j_mid)
            old_y = ext['y']
            ext['j'] = j_sym
            ext['y'] = float(coords_3d[k, j_sym, i_index, 1])
            ext['z'] = float(coords_3d[k, j_sym, i_index, 2])
            print(f"  Mirror: {ext['name']} j={j}(y={old_y:.4f}) -> "
                  f"j={j_sym}(y={ext['y']:.4f})")


# ============================================================
# 共用函數: 極值 scatter
# ============================================================
def add_extrema_markers(ax, extrema):
    for section, entries in extrema.items():
        marker = SECTION_MARKERS[section]
        for ext in entries:
            color = EXTREMA_COLORS[ext['name']]
            ax.scatter(ext['y'], ext['z'],
                       s=EXTREMA_MARKER_SIZE,
                       c=[color],
                       marker=marker,
                       edgecolors='black',
                       linewidths=EXTREMA_EDGE_WIDTH,
                       zorder=15)


# ============================================================
# 共用函數: 格式化數值 (LaTeX)
# ============================================================
def format_value_latex(v):
    """Format a float into LaTeX string (inside math mode)."""
    if v == 0:
        return "0"
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    exp = int(np.floor(np.log10(abs(v))))
    m   = v / (10 ** exp)
    return f"{m:.4f}" + r"\times 10^{" + str(exp) + "}"


# ============================================================
# 共用函數: 建構圖例面板
# ============================================================
def build_legend_panel(ax_leg, extrema, k_center, nz):
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    # ---- 欄位 x 座標 (變數與 = 號靠近) ----
    x_marker = 0.12
    x_name   = 0.21
    x_eq     = 0.50
    x_val    = 0.54
    x_unit   = 0.88

    # ---- 各區段定義 ----
    sections = [
        ('bottom', r'\textbf{Bottom Wall}\ \ $(k = 0)$'),
        ('center', r'\textbf{Center}\ \ $(k = %d)$' % k_center),
        ('top',    r'\textbf{Top Wall}\ \ $(k = %d)$' % (nz - 1)),
    ]
    line_info = {
        'bottom': (r'wall boundary', WALL_LINE_COLOR,   WALL_LINE_OPACITY),
        'center': (r'center line',   CENTER_LINE_COLOR, CENTER_LINE_OPACITY),
        'top':    (r'wall boundary', WALL_LINE_COLOR,   WALL_LINE_OPACITY),
    }

    # ---- 排版參數 (統一字體, 緊湊行距, 區段間距加大) ----
    row_h   = 0.052
    sec_gap = 0.055
    fs      = 14

    # ---- 計算起始 y: 動態計算實際行數後垂直置中 ----
    n_sections = len(sections)
    total_rows = 0
    for sec_key, _ in sections:
        n_ext = sum(1 for v in VAR_ORDER
                    if v in {e['name'] for e in extrema[sec_key]})
        total_rows += 2 + n_ext           # 1 title + 1 line-sample + extrema
    total_h = total_rows * row_h + (n_sections - 1) * sec_gap
    y = 0.50 + total_h / 2.0 + 0.02

    for s_idx, (sec_key, sec_title) in enumerate(sections):

        # ---- 區段分隔線 (與標題保持距離) ----
        if s_idx > 0:
            sep_y = y + sec_gap * 0.50
            ax_leg.plot([0.10, 0.94], [sep_y, sep_y],
                        color=[0.45, 0.45, 0.45], linewidth=0.8)

        # ---- 標題 (置中) + 左右裝飾線 ----
        ax_leg.text(0.52, y, sec_title,
                    fontsize=fs, ha='center', va='center')
        ax_leg.plot([0.08, 0.19], [y, y],
                    color=[0.5, 0.5, 0.5], linewidth=0.6)
        ax_leg.plot([0.85, 0.96], [y, y],
                    color=[0.5, 0.5, 0.5], linewidth=0.6)
        y -= row_h

        # ---- 線段圖例 (加長) ----
        lbl, lc, la = line_info[sec_key]
        ax_leg.plot([x_marker - 0.02, x_marker + 0.10],
                    [y, y],
                    color=lc, linewidth=3.5, alpha=la,
                    solid_capstyle='round')
        ax_leg.text(x_name + 0.02, y, lbl,
                    fontsize=fs, ha='left', va='center',
                    color=[0.30, 0.30, 0.30])
        y -= row_h

        # ---- 極值條目 ----
        marker  = SECTION_MARKERS[sec_key]
        by_name = {e['name']: e for e in extrema[sec_key]}

        for vname in VAR_ORDER:
            if vname not in by_name:
                continue
            ext   = by_name[vname]
            color = EXTREMA_COLORS[vname]

            ax_leg.scatter(x_marker, y, s=70, c=[color],
                           marker=marker, edgecolors='black',
                           linewidths=0.6, zorder=5, clip_on=False)

            ax_leg.text(x_name, y, EXTREMA_LATEX[vname],
                        fontsize=fs, ha='left', va='center')

            ax_leg.text(x_eq, y, r'$=$',
                        fontsize=fs, ha='center', va='center')

            val_str = format_value_latex(ext['value'])
            ax_leg.text(x_val, y, r'$%s$' % val_str,
                        fontsize=fs, ha='left', va='center')

            ax_leg.text(x_unit, y, r'$[\text{--}]$',
                        fontsize=fs, ha='left', va='center')

            y -= row_h

        y -= sec_gap


# ============================================================
# 主程式
# ============================================================
def plot_yz_plane(vtk_file, summary_file, i_index, output_path):
    dims, coords_3d = read_vtk_structured_grid(vtk_file)
    nx, ny, nz = dims
    print(f"VTK dims: nx={nx}, ny={ny}, nz={nz}")

    extrema, k_center = parse_zplus_summary(summary_file)
    if k_center is None:
        k_center = (nz - 1) // 2
    print(f"k_center = {k_center}")
    for sec in ['bottom', 'center', 'top']:
        print(f"  {sec}: {len(extrema[sec])} extrema")

    mirror_extrema_to_left(extrema, coords_3d, dims, i_index)

    print(f"\n--- Y-Z plane at i={i_index} ---")
    y_vals, z_vals, h_segs, v_segs = build_plane_grid(coords_3d, dims, i_index)

    fig = plt.figure(figsize=IMG_SIZE)
    fig.patch.set_facecolor(BG_COLOR)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_leg  = fig.add_subplot(gs[1])
    ax_main.set_facecolor(BG_COLOR)

    add_grid_overlay(ax_main, y_vals, z_vals, h_segs, v_segs)
    add_boundary_lines(ax_main, coords_3d, dims, i_index, k_center)
    add_extrema_markers(ax_main, extrema)

    ax_main.set_xlabel(r'$Y$', fontsize=14)
    ax_main.set_ylabel(r'$Z$', fontsize=14)
    ax_main.set_title(
        r'$Y$-$Z\ \mathrm{Plane\ Grid\ at}\ i = '
        + str(i_index)
        + r'\quad (n_x='  + str(nx)
        + r',\ n_y='      + str(ny)
        + r',\ n_z='      + str(nz) + r')$',
        fontsize=13)
    ax_main.set_aspect('equal')
    ax_main.autoscale_view()
    ax_main.tick_params(labelsize=10)

    build_legend_panel(ax_leg, extrema, k_center, nz)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.08, wspace=0.08)
    plt.savefig(output_path, dpi=IMG_DPI,
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n  Output: {output_path}")
    print(f"  Plane: {ny}(j) x {nz}(k) = {ny * nz} nodes")


if __name__ == "__main__":
    vtk_file     = sys.argv[1] if len(sys.argv) > 1 else r"input\1.Re5600_129x257x129_v2.vtk"
    summary_file = sys.argv[2] if len(sys.argv) > 2 else r"input\14.Re5600_zplus_summary.txt"
    i_idx        = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out_file     = sys.argv[4] if len(sys.argv) > 4 else r"output\yz_plane_i0.png"

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plot_yz_plane(vtk_file, summary_file, i_idx, out_file)
