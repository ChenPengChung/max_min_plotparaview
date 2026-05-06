"""
ParaView Python Shell: Outline + Walls + Grid + Extrema Spheres
================================================================
純 3D 渲染輸出, 不含文字標註 / 圖例 / 牽引線.
後續由 compose_outline.py 結合 matplotlib 圖例面板.

用法:
  pvpython outline_walls.py
"""
from paraview.simple import *
from paraview import servermanager
import vtk as vtk_mod
import os, glob, re, json

# ============================================================
# 設定區
# ============================================================
WALL_COLOR   = [0.7, 0.7, 0.7]
WALL_OPACITY = 0.35
BG_COLOR     = [1, 1, 1]
IMG_SIZE     = [1920, 1080]

NX_NODES     = 32
NY_NODES     = 64
NODE_RADIUS  = 0.04
NODE_COLOR   = [0.0, 0.0, 0.0]
NODE_OPACITY = 0.15
GRID_COLOR   = [0.0, 0.0, 0.0]
GRID_OPACITY = 0.15
GRID_WIDTH   = 2.0

BALL_RADIUS_EX = 0.06

EXTREMA_COLORS = {
    'u_tau_local_max':  [0.80, 0.00, 0.00],
    'u_tau_local_min':  [0.00, 0.50, 0.00],
    'delta_y_plus_max': [0.80, 0.68, 0.00],
    'delta_y_plus_min': [0.50, 0.00, 0.65],
    'delta_z_plus_max': [0.85, 0.35, 0.00],
    'delta_z_plus_min': [0.00, 0.00, 0.65],
}
EXTREMA_NAMES = {
    'u_tau_local_max':  'red',
    'u_tau_local_min':  'green',
    'delta_y_plus_max': 'yellow',
    'delta_y_plus_min': 'purple',
    'delta_z_plus_max': 'orange',
    'delta_z_plus_min': 'deep-blue',
}


# ============================================================
# 共用函數: 在壁面上建構格點 + 網格線
# ============================================================
def build_wall_grid(wall_source, nx, ny, z_offset, out_vtk_path):
    wall_data = servermanager.Fetch(wall_source)
    dims = wall_data.GetDimensions()
    ni_w, nj_w = dims[0], dims[1]

    i_idx = [int(round((ni_w - 1) * float(s) / max(nx - 1, 1))) for s in range(nx)]
    j_idx = [int(round((nj_w - 1) * float(s) / max(ny - 1, 1))) for s in range(ny)]

    pts   = vtk_mod.vtkPoints()
    verts = vtk_mod.vtkCellArray()
    lines = vtk_mod.vtkCellArray()

    for jj in range(ny):
        for ii in range(nx):
            flat = i_idx[ii] + j_idx[jj] * ni_w
            x, y, z = wall_data.GetPoint(flat)
            pts.InsertNextPoint(x, y, z + z_offset)

    for k in range(nx * ny):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)

    for j in range(ny):
        for i in range(nx - 1):
            seg = vtk_mod.vtkLine()
            seg.GetPointIds().SetId(0, j * nx + i)
            seg.GetPointIds().SetId(1, j * nx + i + 1)
            lines.InsertNextCell(seg)

    for i in range(nx):
        for j in range(ny - 1):
            seg = vtk_mod.vtkLine()
            seg.GetPointIds().SetId(0, j * nx + i)
            seg.GetPointIds().SetId(1, (j + 1) * nx + i)
            lines.InsertNextCell(seg)

    polydata = vtk_mod.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetVerts(verts)
    polydata.SetLines(lines)

    writer = vtk_mod.vtkPolyDataWriter()
    writer.SetFileName(out_vtk_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    writer.Write()

    return pts.GetNumberOfPoints(), lines.GetNumberOfCells()


def add_grid_overlay(vtk_path, renderView, label):
    reader = LegacyVTKReader(FileNames=[vtk_path])
    reader.UpdatePipeline()
    n = reader.GetDataInformation().GetNumberOfPoints()
    print("  [%s] Loaded: %d points" % (label, n))

    disp = Show(reader, renderView)
    disp.Representation   = 'Wireframe'
    disp.AmbientColor     = GRID_COLOR
    disp.DiffuseColor     = GRID_COLOR
    disp.LineWidth         = GRID_WIDTH
    disp.Opacity           = GRID_OPACITY
    disp.ColorArrayName    = ['POINTS', '']

    glyph = Glyph(Input=reader, GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray       = ['POINTS', 'No scale array']
    glyph.ScaleFactor      = NODE_RADIUS
    glyph.GlyphMode        = 'All Points'
    glyph.GlyphType.ThetaResolution = 16
    glyph.GlyphType.PhiResolution   = 16
    glyph.UpdatePipeline()

    gDisp = Show(glyph, renderView)
    gDisp.Representation  = 'Surface'
    gDisp.DiffuseColor    = NODE_COLOR
    gDisp.AmbientColor    = NODE_COLOR
    gDisp.Opacity          = NODE_OPACITY
    gDisp.Specular         = 0.3
    gDisp.ColorArrayName   = ['POINTS', '']

    nc = glyph.GetDataInformation().GetNumberOfCells()
    print("  [%s] Glyph spheres: %d cells" % (label, nc))


def parse_zplus_summary(filepath):
    extrema = {'bottom': [], 'top': []}
    current_section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Bottom Wall' in line:
                current_section = 'bottom'
                continue
            elif 'Top Wall' in line:
                current_section = 'top'
                continue
            elif 'Center Node' in line or 'Full-grid' in line:
                current_section = None
                continue

            if current_section is None:
                continue

            m = re.match(
                r'(u_tau_local_(?:max|min)|delta_[yz]_plus_(?:max|min))\s*=\s*'
                r'([\d.eE+-]+)\s+at\s+'
                r'\(i=(\d+),\s*j=(\d+),\s*k=(\d+)\)\s+'
                r'x=([+\-\d.eE]+)\s+y=([+\-\d.eE]+)\s+z=([+\-\d.eE]+)',
                line
            )
            if m:
                extrema[current_section].append({
                    'name':  m.group(1),
                    'value': float(m.group(2)),
                    'i': int(m.group(3)),
                    'j': int(m.group(4)),
                    'k': int(m.group(5)),
                    'x': float(m.group(6)),
                    'y': float(m.group(7)),
                    'z': float(m.group(8)),
                })

    return extrema


# ============================================================
# VTK 檔案路徑
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
vtk_path = os.path.join(script_dir, "input", "1.Re1400_129x257x129_v2.vtk")
if not os.path.isfile(vtk_path):
    raise FileNotFoundError("找不到 %s" % vtk_path)
print("Loading: %s" % vtk_path)

summary_files = glob.glob(os.path.join(script_dir, "input", "*Re*_zplus_summary.txt"))
if summary_files:
    summary_path = summary_files[0]
    print("Summary: %s" % summary_path)
    extrema = parse_zplus_summary(summary_path)
    print("  Bottom wall extrema: %d,  Top wall extrema: %d" % (
        len(extrema['bottom']), len(extrema['top'])))
else:
    extrema = {'bottom': [], 'top': []}
    print("WARNING: No *Re*_zplus_summary.txt found in input/")

# ============================================================
# Step 1: 載入 VTK
# ============================================================
reader = LegacyVTKReader(FileNames=[vtk_path])
reader.UpdatePipeline()

info   = reader.GetDataInformation()
extent = info.GetExtent()
ni = extent[1] - extent[0] + 1
nj = extent[3] - extent[2] + 1
nk = extent[5] - extent[4] + 1
print("Dimensions: ni=%d, nj=%d, nk=%d" % (ni, nj, nk))

# ============================================================
# Step 2: 渲染視圖
# ============================================================
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize   = IMG_SIZE
renderView.Background = BG_COLOR

# ============================================================
# Step 3: Outline (黑色線框)
# ============================================================
outline = Outline(Input=reader)
outline.UpdatePipeline()
outlineDisp = Show(outline, renderView)
outlineDisp.Representation = 'Wireframe'
outlineDisp.AmbientColor   = [0, 0, 0]
outlineDisp.DiffuseColor   = [0, 0, 0]
outlineDisp.LineWidth       = 2.0

# ============================================================
# Step 4: 底壁面 (k=0, Hill) - 灰色半透明
# ============================================================
bottomWall = ExtractSubset(Input=reader)
bottomWall.VOI = [extent[0], extent[1],
                  extent[2], extent[3],
                  extent[4], extent[4]]
bottomWall.UpdatePipeline()

bottomDisp = Show(bottomWall, renderView)
bottomDisp.Representation = 'Surface'
bottomDisp.DiffuseColor   = WALL_COLOR
bottomDisp.AmbientColor   = WALL_COLOR
bottomDisp.Opacity         = WALL_OPACITY
bottomDisp.ColorArrayName  = ['POINTS', '']
print("Bottom wall (k=%d): %d pts" % (extent[4], bottomWall.GetDataInformation().GetNumberOfPoints()))

# ============================================================
# Step 5: 頂壁面 (k=nk-1, Flat) - 灰色半透明
# ============================================================
topWall = ExtractSubset(Input=reader)
topWall.VOI = [extent[0], extent[1],
               extent[2], extent[3],
               extent[5], extent[5]]
topWall.UpdatePipeline()

topDisp = Show(topWall, renderView)
topDisp.Representation = 'Surface'
topDisp.DiffuseColor   = WALL_COLOR
topDisp.AmbientColor   = WALL_COLOR
topDisp.Opacity         = WALL_OPACITY
topDisp.ColorArrayName  = ['POINTS', '']
print("Top wall    (k=%d): %d pts" % (extent[5], topWall.GetDataInformation().GetNumberOfPoints()))

# ============================================================
# Step 6: 頂壁面格點標示
# ============================================================
print("\n--- Top wall grid ---")
top_vtk = os.path.join(script_dir, "_top_wall_grid.vtk")
npts, nlines = build_wall_grid(topWall, NX_NODES, NY_NODES, +0.01, top_vtk)
print("  Built: %d pts, %d lines" % (npts, nlines))
add_grid_overlay(top_vtk, renderView, "Top")

# ============================================================
# Step 7: 底壁面格點標示 (貼合 Hill 曲面)
# ============================================================
print("\n--- Bottom wall grid (Hill surface) ---")
bot_vtk = os.path.join(script_dir, "_bot_wall_grid.vtk")
npts, nlines = build_wall_grid(bottomWall, NX_NODES, NY_NODES, +0.01, bot_vtk)
print("  Built: %d pts, %d lines" % (npts, nlines))
add_grid_overlay(bot_vtk, renderView, "Bottom")

# ============================================================
# Step 8: 極值彩色球體
# ============================================================
print("\n--- Extremum markers (3D spheres) ---")

for section in ['bottom', 'top']:
    for ext in extrema[section]:
        color = EXTREMA_COLORS[ext['name']]
        cname = EXTREMA_NAMES[ext['name']]

        sphere = Sphere()
        sphere.Center          = [ext['x'], ext['y'], ext['z']]
        sphere.Radius          = BALL_RADIUS_EX
        sphere.ThetaResolution = 24
        sphere.PhiResolution   = 24
        sphere.UpdatePipeline()

        sd = Show(sphere, renderView)
        sd.Representation  = 'Surface'
        sd.DiffuseColor    = color
        sd.AmbientColor    = color
        sd.Opacity         = 1.0
        sd.Specular        = 0.0
        sd.ColorArrayName  = ['POINTS', '']

        print("  [%s] %-20s = %12.6e  (%s)" % (
            section, ext['name'], ext['value'], cname))

# ============================================================
# Step 9: 相機
# ============================================================
camera = renderView.GetActiveCamera()
camera.SetFocalPoint(2.25, 4.5, 1.5)
camera.SetPosition(18, -8, 8)
camera.SetViewUp(0, 0, 1)
renderView.ResetCamera()
camera.Dolly(2.0)

# ============================================================
# Step 10: 座標軸
# ============================================================
renderView.AxesGrid.Visibility = 1
renderView.AxesGrid.XTitle = 'X (spanwise)'
renderView.AxesGrid.YTitle = 'Y (streamwise)'
renderView.AxesGrid.ZTitle = 'Z (wall-normal)'
for fam in ('XTitleFontFamily', 'YTitleFontFamily', 'ZTitleFontFamily',
            'XLabelFontFamily', 'YLabelFontFamily', 'ZLabelFontFamily'):
    setattr(renderView.AxesGrid, fam, 'Times')

# ============================================================
# Step 11: 渲染 + 截圖
# ============================================================
Render()

output_dir = os.path.join(script_dir, "output")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

output_png = os.path.join(output_dir, "outline_walls_raw.png")
SaveScreenshot(output_png, renderView,
               ImageResolution=IMG_SIZE,
               TransparentBackground=0)

# ============================================================
# Step 12: 投影極值 3D 座標 -> 2D 像素, 輸出 JSON 供 compose 拉線
# ============================================================
pv_view  = renderView.SMProxy.GetClientSideObject()
ren      = pv_view.GetRenderer()

projected = {'bottom': [], 'top': []}
for section in ['bottom', 'top']:
    for ext in extrema[section]:
        ren.SetWorldPoint(ext['x'], ext['y'], ext['z'], 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        projected[section].append({
            'name':  ext['name'],
            'value': ext['value'],
            'px': dx / IMG_SIZE[0],
            'py': dy / IMG_SIZE[1],
        })

proj_path = os.path.join(output_dir, "extrema_projected.json")
with open(proj_path, 'w') as f:
    json.dump(projected, f, indent=2)
print("Projected extrema -> %s" % proj_path)

print("""
======================================================
  Done!
  Bottom wall (Hill): k=%d   Top wall (Flat): k=%d
  Grid per wall: %d(x) x %d(y) = %d nodes (r=%.2f)
  Output: %s
======================================================
""" % (extent[4], extent[5],
       NX_NODES, NY_NODES, NX_NODES * NY_NODES,
       NODE_RADIUS, output_png))
