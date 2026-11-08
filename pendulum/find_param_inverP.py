import gymnasium as gym
import numpy as np
import mujoco

print("=" * 60)
print("InvertedPendulum-v4 Environment Parameters")
print("=" * 60)

env = gym.make("InvertedPendulum-v4")
m = env.unwrapped.model  # MuJoCo model
d = env.unwrapped.data   # MuJoCo data

# Basic model information
print(f"\nModel Overview:")
print(f"  Bodies: {m.nbody}")
print(f"  Geoms: {m.ngeom}")
print(f"  Joints: {m.njnt}")
print(f"  Actuators: {m.nu}")

print(f"\nBody masses (kg): {m.body_mass}")
print(f"Geom names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]}")
print(f"Joint names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}")

# --- Detailed Body Masses ---
print(f"\nDetailed Body Masses:")
try:
    cart_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")
    pole_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pole")
    print(f"  Cart mass: {float(m.body_mass[cart_bid]):.3f} kg")
    print(f"  Pole mass: {float(m.body_mass[pole_bid]):.3f} kg")
    print(f"  Total mass: {float(m.body_mass[cart_bid] + m.body_mass[pole_bid]):.3f} kg")
except:
    print("  Could not find cart/pole bodies by name")

# --- Geometry Information ---
print(f"\nGeometry Information:")
geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
geom_type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}

for i, name in enumerate(geom_names):
    if name:
        type_name = geom_type_names.get(m.geom_type[i], f'unknown({m.geom_type[i]})')
        print(f"  {name}: {type_name}, size={m.geom_size[i]}")

# --- Pole Dimensions ---
print(f"\nPole Dimensions:")
def get_capsule_dimensions(geom_name):
    try:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if m.geom_type[gid] == 3:  # capsule
            radius, half_len = m.geom_size[gid, 0], m.geom_size[gid, 1]
            return float(2.0 * half_len), float(radius)
    except:
        pass
    return None, None

for gname in ["cpole", "pole", "pole_geom"]:
    if gname in geom_names:
        length, radius = get_capsule_dimensions(gname)
        if length is not None:
            print(f"  {gname}: length = {length:.3f} m, radius = {radius:.3f} m")

# --- Joint Limits & Physics ---
print(f"\nJoint Limits & Physics:")
try:
    slider_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slider")
    print(f"  Slider range: {m.jnt_range[slider_jid]} m")
except:
    print("  Could not find slider joint")
print(f"  Gravity: {m.opt.gravity} m/s²")

# --- Control System ---
print(f"\nControl System:")
if m.nu > 0:  # Check if there are actuators
    act_id = 0  # First actuator (cart slider motor)
    gear = m.actuator_gear[act_id, 0]  # First component of gear vector
    ctrl_range = m.actuator_ctrlrange[act_id]
    print(f"  Actuator gear: {gear}")
    print(f"  Control range: {ctrl_range}")
    print(f"  Max force: {gear * ctrl_range[1]:.1f} N")
    print(f"  Min force: {gear * ctrl_range[0]:.1f} N")
else:
    print("  No actuators found")

print(f"\n" + "=" * 60)
print("Summary for CBF Implementation:")
print("=" * 60)
try:
    cart_mass = float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")])
    pole_mass = float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pole")])
    pole_length, _ = get_capsule_dimensions("cpole")
    slider_range = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "slider")]
    
    print(f"Cart mass (M): {cart_mass:.3f} kg")
    print(f"Pole mass (m): {pole_mass:.3f} kg") 
    print(f"Pole length (L): {pole_length:.3f} m")
    print(f"Half-pole length (l): {pole_length/2:.3f} m")
    print(f"Position limits: ±{slider_range[1]:.1f} m")
    print(f"Force limits: ±{gear * ctrl_range[1]:.0f} N")
    print(f"Gravity: {abs(m.opt.gravity[2]):.2f} m/s²")
except:
    print("Could not extract all parameters")

env.close()
