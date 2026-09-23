from pxr import Usd, UsdGeom
import numpy as np

USD_PATH = "/home/jorge/Documents/Code/silver2_isaacsim/src/scenes/silver2_isaac_sim_locomotion.usd"
stage = Usd.Stage.Open(USD_PATH)

links_to_check = [
    "/World/SILVER2",
    "/World/SILVER2/Body",
    "/World/SILVER2/Coxa_0",
    "/World/SILVER2/Femur_0",
    "/World/SILVER2/Tibia_0",
    "/World/SILVER2/Coxa_1",
    "/World/SILVER2/Femur_1",
    "/World/SILVER2/Tibia_1",
    "/World/SILVER2/Coxa_2",
    "/World/SILVER2/Femur_2",
    "/World/SILVER2/Tibia_2",
    "/World/SILVER2/Coxa_3",
    "/World/SILVER2/Femur_3",
    "/World/SILVER2/Tibia_3",
    "/World/SILVER2/Coxa_4",
    "/World/SILVER2/Femur_4",
    "/World/SILVER2/Tibia_4",
    "/World/SILVER2/Coxa_5",
    "/World/SILVER2/Femur_5",
    "/World/SILVER2/Tibia_5",
]

print("\n" + "=" * 75)
print(f"{'Prim Path':<28} | {'Translation':<20} | {'Has NaN/Inf?'}")
print("=" * 75)

for path in links_to_check:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"{path:<28} | Prim not found")
        continue

    xformable = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()
    transform_matrix = xformable.ComputeLocalToWorldTransform(time)
    translation = transform_matrix.ExtractTranslation()
    
    vals = np.array([translation[0], translation[1], translation[2]])
    has_nan = np.isnan(vals).any() or np.isinf(vals).any()
    
    print(f"{path:<28} | {str(translation):<20} | {has_nan}")

print("=" * 75 + "\n")