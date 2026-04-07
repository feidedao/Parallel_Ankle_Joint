import mujoco, mujoco_viewer
import numpy as np
# conda activate mujocoRl12
import os
import matplotlib.pyplot as plt 

# 模型为个人装配，可能和官方不一致，动力学参数完全错误  
current_dir = os.getcwd()
print("当前目录是:", current_dir)  
folder_path = current_dir + '/parallel_robots_ankle/5_tiangong/'
xml_path = 'urdf/TG10-00.xml'   #23urdf311
 
model = mujoco.MjModel.from_xml_path(folder_path + xml_path )
data = mujoco.MjData(model)


viewer = mujoco_viewer.MujocoViewer(model,data) 
viewer.cam.distance=3.0 
viewer.cam.azimuth = 90
viewer.cam.elevation=-45
viewer.cam.lookat[:]=np.array([0.0,-0.25,0.824])
 
dt = 0.001
T = 10000
model.opt.timestep = dt  # ← 这就是 dt！

q_vel = []
q_pos = []
count = 0
damping_all_joints =  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,  0.5  ]
kp = 100 
kv = 1
for i in range(T):
    count += 1
    sin_data = 0.4 * np.sin(2* np.pi *0.6 * dt * count)
    data.ctrl[0] = sin_data # left motor
    data.ctrl[1] = 0.5*sin_data # right motor 

    mujoco.mj_step(model, data)
    viewer.render()

    # generalized acceleration
    q_vel.append(data.qvel.copy())
    q_pos.append(data.qpos.copy())
    # 检查是否有 equality 约束
    if model.neq > 0:  
        print(f"--- Step {count} Equality Constraints Error ---") 
        
        num_eq_constraints = model.neq 
        dims_per_connect = 3 
        
        for i in range(num_eq_constraints):
            start_idx = i * dims_per_connect
            end_idx = start_idx + dims_per_connect
            
            if end_idx <= len(data.efc_pos): 
                error_vec = data.efc_pos[start_idx:end_idx] 
                distance_error = np.linalg.norm(error_vec) 
                print(f"Constraint  Error Vector = [{error_vec[0]:.6f}, {error_vec[1]:.6f}, {error_vec[2]:.6f}], "
                    f"Total Distance Gap = {distance_error:.6e} m")
            else:
                print(f"Warning: Index out of range for constraint {i}")
    else:
        print("No equality constraints found in the model.")

qpos=np.array(q_pos)     
plt.figure()
plt.plot(qpos[:,-1]    ,'m')
plt.plot(qpos[:,-2],'k')
plt.show()
np.savez(folder_path + 'mujoco_tg102.npz',  
          qvel=np.array(q_vel),  
          qpos=np.array(q_pos)  )
print(1)


