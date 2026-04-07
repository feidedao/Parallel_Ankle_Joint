import numpy as np 
import matplotlib.pyplot as plt 
import torch 
# import joblib,os 
import os 
current_dir = os.getcwd()
print("当前目录是:", current_dir) 

path_folder = current_dir+ '/parallel_robots_ankle/2_g1_ankle/'
data = np.load(path_folder+ 'mujoco_g12.npz')
qvel = np.array(data['qpos'])
qpos = np.array(data['qpos'])
  
# for i in range(3):
#     plt.figure() 
#     plt.plot(qddot_mujoco[:,i],'k')
#     plt.show()
  
gdata1 = np.load(path_folder+ 'paral_g13.npz')
save_data = np.array(gdata1['qpos']) 
   
# plt.figure()
# plt.plot(save_data[:,0]   ,'m')
# plt.plot( save_data[:,1],'k')
# plt.plot( save_data[:,2],'c')
# plt.plot( save_data[:,3],'r')
# plt.show()



plt.figure()
plt.plot(save_data[:,0]   ,'m') 
plt.plot(qpos[:,0],'k')  
plt.title('pitch')
plt.show()

plt.figure()
plt.plot(save_data[:,1]  ,'m') 
plt.plot(qpos[:,1],'k')  
plt.title('roll')
plt.show()

plt.figure()
plt.plot(save_data[:,2]   ,'m')
plt.plot(qpos[:,-2],'k')
plt.title('left_motor')
plt.show()

plt.figure()
plt.plot(save_data[:,3]  ,'m')
plt.plot(qpos[:,-1],'k') 
plt.title('right_motor')
plt.show()
print(21)