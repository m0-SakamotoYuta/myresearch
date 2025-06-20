#----------------[前提パッケージの導入]---------------------------------
import numpy as np
import open3d as o3d
from tkinter import Tk, filedialog
import xml.etree.ElementTree as ET
import pandas as pd
import time
import copy
import open3d.visualization.gui as gui # type: ignore
import open3d.visualization.rendering as rendering # type: ignore
import threading
#---------------------------------------------------------------------


#----------------[大腿骨ファイル選択]-----------------------------------
root = Tk()
root.withdraw()
fe_asc_file = filedialog.askopenfilename(
    title="大腿骨ASCファイルを選択してください",
    filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
)
if not fe_asc_file:
    print("大腿骨ASCファイルの選択がキャンセルされました。")
    exit()

fe_pp_file = filedialog.askopenfilename(
    title="大腿骨PPファイルを選択してください",
    filetypes=[("PP files", "*.pp"), ("All files", "*.*")]
)
if not fe_pp_file:
    print("大腿骨PPファイルの選択がキャンセルされました。")
    exit()
#----------------------------------------------------------------------


#----------------[大腿骨モデルの読み込み]--------------------------------
fe_model = o3d.io.read_point_cloud(fe_asc_file ,format='xyz')
fe_model.paint_uniform_color([0.5, 0.5, 0.9]) 
if not fe_model:
    print("大腿骨ASCファイルの読み込みに失敗しました。")
    exit()
#----------------------------------------------------------------------


#----------------[大腿骨PPファイルの読み込みと座標抽出]-------------------
# feA = 内側顆，feB = 外側顆，feC1，feC2 = 大腿骨頭の任意2点，feC = 大腿骨頭の中点，feD = mid FE
fe_tree = ET.parse(fe_pp_file)
fe_root = fe_tree.getroot()

fe_point_dict = {}
for point in fe_root.findall(".//point"):
    name = point.get("name")
    x = float(point.get("x"))
    y = float(point.get("y"))
    z = float(point.get("z"))
    fe_point_dict[name] = np.array([x, y, z])

# feAとfeBの中点feD、feC1とfeC2の中点feCを生成
feD = None
feC = None
if "feA" in fe_point_dict and "feB" in fe_point_dict:
    feD = (fe_point_dict["feA"] + fe_point_dict["feB"]) / 2
if "feC1" in fe_point_dict and "feC2" in fe_point_dict:
    feC = (fe_point_dict["feC1"] + fe_point_dict["feC2"]) / 2
#---------------------------------------------------------------------


#----------------[座標系Cfの構築]--------------------------------------
if feC is not None and feD is not None and "feA" in fe_point_dict and "feB" in fe_point_dict:
    # Y軸
    y_axis = feC - feD
    y_axis /= np.linalg.norm(y_axis)
    # Z軸（feA - feB方向をY軸に直交化）
    ba_vec = fe_point_dict["feA"] - fe_point_dict["feB"]
    ba_proj = ba_vec - np.dot(ba_vec, y_axis) * y_axis
    z_axis = ba_proj / np.linalg.norm(ba_proj)
    # X軸
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # 直交化再調整
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    # Cf座標系の変換行列
    cf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    cf_transform = np.eye(4)
    cf_transform[:3, 0] = x_axis
    cf_transform[:3, 1] = y_axis
    cf_transform[:3, 2] = z_axis
    cf_transform[:3, 3] = feC
    cf_frame.transform(cf_transform)
else:
    print("大腿骨の特徴点が不足しています。")
    cf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
#---------------------------------------------------------------------

#----------------[大腿骨モデルとCfの可視化]------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(fe_model)
vis.add_geometry(cf_frame)
vis.run()
vis.destroy_window() 
#---------------------------------------------------------------------


#----------------[骨盤ファイル選択]--------------------------------
pv_asc_file = filedialog.askopenfilename(
    title="骨盤ASCファイルを選択してください",
    filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
)
if not pv_asc_file:
    print("骨盤ASCファイルの選択がキャンセルされました。")
    exit()

pv_pp_file = filedialog.askopenfilename(
    title="骨盤PPファイルを選択してください",
    filetypes=[("PP files", "*.pp"), ("All files", "*.*")]
)
if not pv_pp_file:
    print("骨盤PPファイルの選択がキャンセルされました。")
    exit()
#----------------------------------------------------------------------


#----------------[骨盤モデルの読み込み]--------------------------------
pv_model = o3d.io.read_point_cloud(pv_asc_file ,format='xyz')
pv_model.paint_uniform_color([0.9, 0.5, 0.5]) 
if not pv_model:
    print("骨盤ASCファイルの読み込みに失敗しました。")
    exit()
#---------------------------------------------------------------------

#----------------[骨盤PPファイルの読み込みと座標抽出]-------------------
#pvA,pvB = ASIS, pvC,pvD = PSIS, pvE1,pvE2 = 寛骨臼の任意2点，pvE = 回転中心(pvE1とE2の中点)，pvF = PSISの中点，
pv_tree = ET.parse(pv_pp_file)
pv_root = pv_tree.getroot()

pv_point_dict = {}
for point in pv_root.findall(".//point"):
    name = point.get("name")
    x = float(point.get("x"))
    y = float(point.get("y"))
    z = float(point.get("z"))
    pv_point_dict[name] = np.array([x, y, z])

# pvE1とpvE2の中点pvE、pvCとpvDの中点pvFを生成
pvE = None
pvF = None
if "pvE1" in pv_point_dict and "pvE2" in pv_point_dict:
    pvE = (pv_point_dict["pvE1"] + pv_point_dict["pvE2"]) / 2
if "pvC" in pv_point_dict and "pvD" in pv_point_dict:
    pvF = (pv_point_dict["pvC"] + pv_point_dict["pvD"]) / 2
#----------------------------------------------------------------------


#----------------[座標系Cpの構築]--------------------------------------
if pvE is not None and "pvA" in pv_point_dict and "pvB" in pv_point_dict and pvF is not None:
    # Z軸
    z_axis = pv_point_dict["pvA"] - pv_point_dict["pvB"]
    z_axis /= np.linalg.norm(z_axis)

    # X軸（F→A方向をZ軸に直交化し、F→Aが正になるように）
    fa_vec = pv_point_dict["pvA"] - pvF
    fa_proj = fa_vec - np.dot(fa_vec, z_axis) * z_axis
    x_axis = fa_proj / np.linalg.norm(fa_proj)

    # Y軸
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 直交化再調整（数値誤差対策）
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Cp座標系の変換行列
    cp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    cp_transform = np.eye(4)
    cp_transform[:3, 0] = x_axis
    cp_transform[:3, 1] = y_axis
    cp_transform[:3, 2] = z_axis
    cp_transform[:3, 3] = pvE
    cp_frame.transform(cp_transform)
else:
    print("骨盤の特徴点が不足しています。")
    cp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
#---------------------------------------------------------------------

#----------------[骨盤モデルとCpの可視化]------------------------------
wd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pv_model)
vis.add_geometry(cp_frame)
vis.add_geometry(wd_frame)
vis.run()
vis.destroy_window() 
#---------------------------------------------------------------------

#----------------[Cp→Cpy]------------------------------
cp_inv = np.linalg.inv(cp_transform)
pv_model.transform(cp_inv)  # pv_modelをCp座標系に変換
cp_frame.transform(cp_inv)  # cp_frameをCp座標系に変換
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pv_model)
vis.add_geometry(cp_frame)
vis.add_geometry(wd_frame)
vis.run()
vis.destroy_window() 

#-----------------[Cf→Cpy]------------------------------
cf_inv = np.linalg.inv(cf_transform)
fe_model.transform(cf_inv)  # fe_modelをCf座標系に変換
cf_frame.transform(cf_inv)  # cf_frameをCf座標系に変換
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(fe_model)
vis.add_geometry(cf_frame)
vis.add_geometry(pv_model)
vis.add_geometry(cp_frame)
vis.run()
vis.destroy_window()
#-------------------------------------------------------

#----------------[同次変換パラメータのエクセルファイルの読み込み]-------------
# エクセルの列: 時間, Z軸回転(°), X回転(°), Y回転(°), Z変位, X変位, Y変位
CpCf_Trans_xl_path = filedialog.askopenfilename(
    title="力学試験の変位csvファイルを選択してください",
    filetypes=[("xlsx files", "*.xlsx"), ("All files", "*.*")]
)
if not CpCf_Trans_xl_path:
    print("エクセルファイルの選択がキャンセルされました。")
    exit()

df = pd.read_excel(CpCf_Trans_xl_path, header=None, skiprows=15)

transform_list = []
frame_list = []

for i in range(len(df)):
    row = df.iloc[i]
    frame_time = row[0]
    rz = np.deg2rad(row[27])  # FE
    tz = row[28]              # ML
    rx = np.deg2rad(row[29])  # VV
    tx = row[30]              # AP
    ry = np.deg2rad(row[31])  # IE
    ty = row[32]              # PD

    # 回転行列（Z→X→Yの順で回転）
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])
    Rx = np.array([
        [1, 0,           0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0,          1, 0],
        [-np.sin(ry),0, np.cos(ry)]
    ])
    # 合成回転行列（Z→X→Yの順）
    R = Rz @ Rx @ Ry

    # 同次変換行列
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = [tx, ty, tz]  # X, Y, Zの順で並進

    transform_list.append(mat)
    frame_list.append(frame_time)
#---------------------------------------------------------------------
print("同次変換行列のエクセルファイルの読み込みが完了しました")

#---
cp_traj_points = [mat[:3, 3] for mat in transform_list]
cp_traj_points_np = np.array(cp_traj_points)

cp_traj_pcd = o3d.geometry.PointCloud()
cp_traj_pcd.points = o3d.utility.Vector3dVector(cp_traj_points_np)
cp_traj_pcd.paint_uniform_color([1, 1, 0])  # 黄色
#---


print("シミュレーションのモードを選択してください(auto/manual):")
mode = input("mode = ").strip().lower()

if mode == "auto":
    #----------------------[動作シミュレーション]------------------------------
    print("「動作シミュレーション:自動再生」を開始します")
    # 動作シミュレーションのウィンドウと世界を設置
    gui.Application.instance.initialize()
    at_window1 = gui.Application.instance.create_window("Model Viewer", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(at_window1.renderer)
    at_window1.add_child(scene)

    # 時間表示用のウィンドウラベル
    time_window = gui.Application.instance.create_window("Time", 300, 60)
    time_label = gui.Label(f"Time: {frame_list[0]}")
    time_window.add_child(time_label)

    # cf_frameのオリジナルを保存
    cf_frame_orig = copy.deepcopy(cf_frame)
    fe_model_orig = copy.deepcopy(fe_model)

    # 最初のcf_frameを追加
    scene.scene.add_geometry("pv_model", pv_model, rendering.MaterialRecord())
    scene.scene.add_geometry("fe_model", fe_model, rendering.MaterialRecord())
    scene.scene.add_geometry("cp_traj_line", cp_traj_pcd, rendering.MaterialRecord())
    bounds = fe_model.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    current_frame = [0]

    def auto_play(at_frame):
        cf_frame_copy = copy.deepcopy(cf_frame_orig)
        fe_model_copy = copy.deepcopy(fe_model_orig)
        fe_model_copy.transform(transform_list[at_frame])
        aabb = cf_frame_copy.get_axis_aligned_bounding_box()
        scene.scene.remove_geometry("cf_frame")
        scene.scene.remove_geometry("fe_model")
        if not aabb.is_empty():
            scene.scene.add_geometry("fe_model", fe_model_copy, rendering.MaterialRecord())
        else:
            print("同次変換行列のエクセルファイルに異常があります")
        time_label.text = f"Time: {frame_list[at_frame]}"

    def animation_loop():
        at_frame = 0
        while True:
            gui.Application.instance.post_to_main_thread(
                at_window1, lambda f=at_frame: auto_play(f)
            )
            at_frame = (at_frame + 1) % len(transform_list)
            time.sleep(0.02)  # フレーム間隔

    threading.Thread(target=animation_loop, daemon=True).start()
    gui.Application.instance.run()
    #---------------------------------------------------------------------

elif mode == "manual":
    #---------------------------------------------------------------------
    print("「動作シミュレーション:再生バー付」を開始します")
    # 動作シミュレーションのウィンドウと世界を設置
    gui.Application.instance.initialize()
    window1 = gui.Application.instance.create_window("Model Viewer", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window1.renderer)
    window1.add_child(scene)

    # cf_frameのオリジナルを保存
    cf_frame_orig = copy.deepcopy(cf_frame)
    fe_model_orig = copy.deepcopy(fe_model)

    # 最初のcf_frameを追加
    scene.scene.add_geometry("cf_frame", cf_frame, rendering.MaterialRecord())
    scene.scene.add_geometry("pv_model", pv_model, rendering.MaterialRecord())
    scene.scene.add_geometry("cp_frame", cp_frame, rendering.MaterialRecord())
    scene.scene.add_geometry("fe_model", fe_model, rendering.MaterialRecord())
    scene.scene.add_geometry("cp_traj_line", cp_traj_line, rendering.MaterialRecord())
    bounds = fe_model.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    # スライダーだけのウィンドウ
    window2 = gui.Application.instance.create_window("Slider", 400, 60)
    slider = gui.Slider(gui.Slider.INT)
    slider.set_limits(0, max(0, len(transform_list) - 1))
    slider.int_value = 0
    window2.add_child(slider)

    # --- スライダーの値でcf_frameを動かす ---
    def on_slider_change(value):
        frame = int(value)
        cf_frame_copy = copy.deepcopy(cf_frame_orig)
        fe_model_copy = copy.deepcopy(fe_model_orig)
        cf_frame_copy.transform(transform_list[frame])
        fe_model_copy.transform(transform_list[frame])
        aabb = cf_frame_copy.get_axis_aligned_bounding_box()
        scene.scene.remove_geometry("cf_frame")
        scene.scene.remove_geometry("fe_model")
        if not aabb.is_empty():
            scene.scene.add_geometry("cf_frame", cf_frame_copy, rendering.MaterialRecord())
            scene.scene.add_geometry("fe_model", fe_model_copy, rendering.MaterialRecord())
        else:
            print("同次変換行列のエクセルファイルに異常があります")
    slider.set_on_value_changed(on_slider_change)
    gui.Application.instance.run()
    #---------------------------------------------------------------------
