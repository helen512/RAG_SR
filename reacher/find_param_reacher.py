import gymnasium as gym
import numpy as np
import mujoco

print("=" * 60)
print("Reacher-v5 Environment Parameters")
print("=" * 60)

env = gym.make("Reacher-v5")
m = env.unwrapped.model  # MuJoCo model
d = env.unwrapped.data   # MuJoCo data

# Basic model information
print(f"\nModel Overview:")
print(f"  Bodies: {m.nbody}")
print(f"  Geoms: {m.ngeom}")
print(f"  Joints: {m.njnt}")
print(f"  Actuators: {m.nu}")
print(f"  Sites: {m.nsite}")

# Body and joint names
print(f"\nBody names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]}")
print(f"Geom names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]}")
print(f"Joint names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}")
print(f"Site names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]}")

# --- Detailed Body Masses ---
print(f"\nDetailed Body Masses:")
print(f"Body masses (kg): {m.body_mass}")

try:
    # Try to find link bodies by name
    link_names = ["body0", "body1", "link0", "link1", "upper_arm", "forearm"]
    found_links = {}
    
    for link_name in link_names:
        try:
            link_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link_name)
            mass = float(m.body_mass[link_bid])
            found_links[link_name] = mass
            print(f"  {link_name} mass: {mass:.6f} kg")
        except:
            continue
            
    if not found_links:
        # If specific names not found, list all non-world bodies
        print("  All body masses:")
        for i in range(m.nbody):
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name != "world":
                print(f"    {body_name}: {float(m.body_mass[i]):.6f} kg")
                
except Exception as e:
    print(f"  Error accessing body masses: {e}")

# --- Body Inertias ---
print(f"\nBody Inertias (kg⋅m²):")
try:
    for i in range(m.nbody):
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and body_name != "world":
            # Inertia matrix diagonal elements (Ixx, Iyy, Izz)
            inertia = m.body_inertia[i]
            print(f"  {body_name}: Ixx={inertia[0]:.6f}, Iyy={inertia[1]:.6f}, Izz={inertia[2]:.6f}")
except Exception as e:
    print(f"  Error accessing inertias: {e}")

# --- Geometry Information ---
print(f"\nGeometry Information:")
geom_type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 
                   4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}

for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name:
        type_name = geom_type_names.get(m.geom_type[i], f'unknown({m.geom_type[i]})')
        size = m.geom_size[i]
        print(f"  {name}: {type_name}, size={size}")

# --- Link Dimensions ---
print(f"\nLink Dimensions:")
def get_capsule_dimensions(geom_name):
    """Get length and radius for capsule geometry"""
    try:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if m.geom_type[gid] == 3:  # capsule
            radius, half_len = m.geom_size[gid, 0], m.geom_size[gid, 1]
            return float(2.0 * half_len), float(radius)
    except:
        pass
    return None, None

def get_cylinder_dimensions(geom_name):
    """Get radius and height for cylinder geometry"""
    try:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if m.geom_type[gid] == 5:  # cylinder
            radius, half_height = m.geom_size[gid, 0], m.geom_size[gid, 1]
            return float(2.0 * half_height), float(radius)
    except:
        pass
    return None, None

# Try different possible names for reacher links
link_geom_names = ["link0", "link1", "body0", "body1", "upper_arm", "forearm", 
                   "arm1", "arm2", "shoulder", "elbow", "geom0", "geom1"]

for gname in link_geom_names:
    length, radius = get_capsule_dimensions(gname)
    if length is not None:
        print(f"  {gname} (capsule): length = {length:.6f} m, radius = {radius:.6f} m")
    else:
        length, radius = get_cylinder_dimensions(gname)
        if length is not None:
            print(f"  {gname} (cylinder): height = {length:.6f} m, radius = {radius:.6f} m")

# --- Joint Information ---
print(f"\nJoint Information:")
for i in range(m.njnt):
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name:
        joint_type = m.jnt_type[i]
        joint_type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        type_name = joint_type_names.get(joint_type, f'unknown({joint_type})')
        
        # Joint limits
        if m.jnt_limited[i]:
            range_min, range_max = m.jnt_range[i]
            print(f"  {joint_name}: {type_name}, range=[{range_min:.3f}, {range_max:.3f}] rad")
        else:
            print(f"  {joint_name}: {type_name}, unlimited")

# --- Actuator Information ---
print(f"\nActuator Information:")
for i in range(m.nu):
    act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
    gear = m.actuator_gear[i, 0]  # First component of gear vector
    ctrl_range = m.actuator_ctrlrange[i]
    
    print(f"  {act_name}:")
    print(f"    Gear: {gear:.3f}")
    print(f"    Control range: [{ctrl_range[0]:.3f}, {ctrl_range[1]:.3f}]")
    print(f"    Max torque: {abs(gear * ctrl_range[1]):.3f} N⋅m")

# --- Site Information (Target location) ---
print(f"\nSite Information:")
for i in range(m.nsite):
    site_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
    if site_name:
        # Site position in world coordinates
        site_pos = d.site_xpos[i]
        print(f"  {site_name}: position=({site_pos[0]:.3f}, {site_pos[1]:.3f}, {site_pos[2]:.3f})")

# --- Physics Settings ---
print(f"\nPhysics Settings:")
print(f"  Gravity: {m.opt.gravity} m/s²")
print(f"  Timestep: {m.opt.timestep:.6f} s")
if hasattr(m.opt, 'density'):
    print(f"  Air density: {m.opt.density:.6f} kg/m³")

# --- State Space Information ---
print(f"\nState Space:")
obs, _ = env.reset()
print(f"  Observation dimension: {len(obs)}")
print(f"  Action dimension: {env.action_space.shape[0]}")
print(f"  Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

# Get current joint positions and velocities
qpos = d.qpos[:m.nq]  # Joint positions
qvel = d.qvel[:m.nv]  # Joint velocities

print(f"\nCurrent State:")
print(f"  Joint positions (rad): {qpos}")
print(f"  Joint velocities (rad/s): {qvel}")

# Calculate end effector position
def compute_end_effector_pos(theta1, theta2, l1=0.1, l2=0.11):
    """Compute end effector position from joint angles"""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

if len(qpos) >= 2:
    ee_x, ee_y = compute_end_effector_pos(qpos[0], qpos[1])
    print(f"  End effector position: ({ee_x:.3f}, {ee_y:.3f})")

print(f"\n" + "=" * 60)
print("Summary for CBF Implementation:")
print("=" * 60)

try:
    # Extract key parameters for control and safety
    # Default link lengths (typical for Reacher)
    l1, l2 = 0.1, 0.11  # meters
    
    # Try to get actual link masses
    m1, m2 = 1.0, 1.0  # default values
    try:
        # Look for body masses (excluding world)
        masses = []
        for i in range(1, m.nbody):  # Skip world body (index 0)
            masses.append(float(m.body_mass[i]))
        if len(masses) >= 2:
            m1, m2 = masses[0], masses[1]
    except:
        pass
    
    # Get joint limits
    joint_limits = []
    for i in range(min(2, m.njnt)):  # First two joints
        if m.jnt_limited[i]:
            range_min, range_max = m.jnt_range[i]
            joint_limits.append((float(range_min), float(range_max)))
        else:
            joint_limits.append((-np.pi, np.pi))  # Default unlimited
    
    # Get actuator limits
    torque_limits = []
    for i in range(min(2, m.nu)):  # First two actuators
        gear = m.actuator_gear[i, 0]
        print(f"wGear: {gear}")
        ctrl_range = m.actuator_ctrlrange[i]
        print(f"wControl range: {ctrl_range}")
        max_torque = abs(gear * ctrl_range[1])
        torque_limits.append(float(max_torque))
    
    print(f"Link 1 mass (m1): {m1:.6f} kg")
    print(f"Link 2 mass (m2): {m2:.6f} kg")
    print(f"Link 1 length (l1): {l1:.3f} m")
    print(f"Link 2 length (l2): {l2:.3f} m")
    
    if joint_limits:
        for i, (min_angle, max_angle) in enumerate(joint_limits[:2]):
            print(f"Joint {i+1} limits: [{min_angle:.3f}, {max_angle:.3f}] rad")
    
    if torque_limits:
        for i, max_torque in enumerate(torque_limits[:2]):
            print(f"Joint {i+1} max torque: ±{max_torque:.3f} N⋅m")
    
    print(f"Gravity: {abs(m.opt.gravity[2]):.2f} m/s²")
    print(f"Control frequency: {1.0/m.opt.timestep:.1f} Hz")
    
    # Workspace information
    max_reach = l1 + l2
    min_reach = abs(l1 - l2)
    print(f"Maximum reach: {max_reach:.3f} m")
    print(f"Minimum reach: {min_reach:.3f} m")
    print(f"Workspace area: π × {max_reach:.3f}² = {np.pi * max_reach**2:.3f} m²")
    
except Exception as e:
    print(f"Could not extract all parameters: {e}")

env.close()
print(f"\n" + "=" * 60)
print("Parameter extraction completed successfully!")
