## fdd
## wechat   feidedaoRobot
## article ---
# On the Comprehensive Kinematics Analysis
# of a Humanoid Parallel Ankle Mechanism

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 模型为个人装配，可能和官方不一致，动力学参数完全错误  
current_dir = os.getcwd()
print("当前目录是:", current_dir) 

class parallelAnkle:
    def __init__(self ):
 
        self.lspace = 0.04325*2
   
        self.ra01 = np.array([0, self.lspace * 0.5, 0.215])
        self.ra02 = np.array([0, -self.lspace * 0.5, 0.140])
  
        self.rb01 = np.array([-0.05582, self.lspace * 0.5, 0.237])
        self.rb02 = np.array([-0.05582, -self.lspace * 0.5, 0.162])
        cz = 0.022
        self.rc01 = np.array([-0.05582, self.lspace * 0.5, cz])
        self.rc02 = np.array([-0.05582, -self.lspace * 0.5, cz])

        self.re01 = np.array([0.1, self.lspace * 0.5, cz])
        self.re02 = np.array([0.1, -self.lspace * 0.5, cz])
        
        self.lbar1 = np.linalg.norm(self.ra01 - self.rb01)
        self.lbar2 = np.linalg.norm(self.ra02 - self.rb02)
        self.lrod1 = np.linalg.norm(self.rc01 - self.rb01)
        self.lrod2 = np.linalg.norm(self.rc02 - self.rb02)

        self.rab01 = np.array([-self.lbar1, 0, 0])
        self.rab02 = np.array([-self.lbar2, 0, 0])
 
        self.m01 = 0.37543059278589647
        self.m02 = 0.3754305927858966

        self.pitch_range = [ -1.22 ,   0.5236 ]
        self.roll_range =  [ -0.4363 ,  0.4363 ] 

        print("parallelAnkle init ok")

    def RyPlot(self,pitch):
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        return Ry
    
    def RxPlot(self, roll):
        Rx = np.array([[1,0,0],
                       [0,np.cos(roll),-np.sin(roll)],
                       [0,np.sin(roll),np.cos(roll)]])
        return Rx
    
    def xrot(self,pitch,roll):
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                       [0,1,0],
                       [-np.sin(pitch),0,np.cos(pitch)]])

        Rx = np.array([[1,0,0],
                       [0,np.cos(roll),-np.sin(roll)],
                       [0,np.sin(roll),np.cos(roll)]])

        # Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
        #                [np.sin(yaw),np.cos(yaw),0],
        #                [0,  0,1]])

        return Ry @ Rx
 
    # Inverse Kinematics Solver
    # Inputs:
    #   pitch, roll:  joint angles   
    # Output:
    #   theta1, theta2: Actuator joint angles (motor positions). 
    def ik(self, pitch, roll ):  
        error_state = 0  

        rc1 = self.xrot(pitch, roll) @ self.rc01
        ra1 = self.ra01
        rac1 = (rc1 - ra1)
        a1 = rac1[0]
        b1 = rac1[2]
        c1 = -(self.lrod1**2 - self.lbar1**2 - rac1@rac1)/(2*self.lbar1)

        jjj = b1*b1*c1*c1 - (a1**2 + b1**2)*(c1**2 - a1**2)
        if jjj < .0:
            # 超出上限，可能是浮点误差或物理不可达 
            print("1 Warning: Argument <0 ,jjj:  ",jjj)
            jjj = .0001
            error_state = 1

        ss = (b1*c1 + np.sqrt(jjj))/(a1**2 + b1**2)
        if ss < -1:
            # 超出上限，可能是浮点误差或物理不可达 
            print("1 Warning:  ,ss:  ",ss)
            ss = -1
            error_state = 2
        elif   ss > 1:
            # 超出上限，可能是浮点误差或物理不可达 
            print("1 Warning:  ,ss:  ",ss)
            ss =  1
            error_state = 3
        theta1 = np.arcsin(ss)- self.m01     
        # theta1 = np.arcsin((b1*c1 + np.sqrt(b1*b1*c1*c1 - (a1**2 + b1**2)*(c1**2 - a1**2)))/(a1**2 + b1**2))

        rc2 = self.xrot(pitch, roll) @ self.rc02
        ra2 = self.ra02
        rac2 = (rc2 - ra2)
        a2 = rac2[0]
        b2 = rac2[2]
        c2 = -(self.lrod2**2 - self.lbar2**2 - rac2@rac2)/(2*self.lbar2)
        jjj = b2**2*c2**2 - (a2**2 + b2**2)*(c2**2 - a2**2)
        if jjj < .0:
            # 超出上限，可能是浮点误差或物理不可达
            jjj = .0
            print("2Warning: Argument <0 ,jjj:  ",jjj) 
            error_state = 4

        ss = (b2*c2 + np.sqrt( jjj))/(a2**2 + b2**2)
        if ss < -1:
            # 超出上限，可能是浮点误差或物理不可达 
            print("1 Warning:  ,ss:  ",ss)
            ss = -1
            error_state = 5
        elif ss > 1:
            # 超出上限，可能是浮点误差或物理不可达 
            print("1 Warning:  ,ss:  ",ss)
            ss =  1
            error_state = 6
        # theta2 = -(np.arccos(jjj) + np.arctan2(b2,a2 ) - self.m02)
     
        theta2 = np.arcsin(ss) - self.m02

        # theta2 = np.arcsin((b2*c2 + np.sqrt(b2**2*c2**2 - (a2**2 + b2**2)*(c2**2 - a2**2)))/(a2**2 + b2**2))
        return theta1, theta2, error_state

    # Jacobi
    # Inputs:
    #   pitchRef, rollRef: Reference orientation angles for the platform.
    #   theta1, theta2: Actuator joint angles (motor positions).
    # Output:
    #   Computed jacobi
    def Jac(self,pitch,roll,theta1,theta2):

        Jx = np.zeros((2,6))
        Jtheta = np.zeros((2, 2))
        G = np.array([[0,0,0, np.cos(pitch),0,-np.sin(pitch)],
                      [0,0,0, 0,1,0]]).transpose()

        s11 = np.array([0,1,0])
        s21 = s11
        rb1 = self.ra01 + self.RyPlot(theta1 + self.m01) @ self.rab01
        rc1 = self.xrot(pitch, roll) @ self.rc01
        ra1 = self.ra01

        rbar1 = rb1 - ra1
        rrod1 = rc1 - rb1

        rb2 = self.ra02 + self.RyPlot(theta2 + self.m02) @ self.rab02
        rc2 = self.xrot(pitch, roll) @ self.rc02
        ra2 = self.ra02

        rbar2 = rb2 - ra2
        rrod2 = rc2 - rb2

        Jtheta = np.array([[s11@np.cross(rbar1,rrod1),0],
                           [0,s21@np.cross(rbar2 ,rrod2)]])

        Jx[0,0:3] = rrod1.transpose()
        Jx[0, 3:6] = (np.cross(rc1,rrod1)).transpose()

        Jx[1, 0:3] = rrod2.transpose()
        Jx[1, 3:6] = (np.cross(rc2,rrod2)).transpose()

        Jc = np.linalg.inv(Jtheta) @ (Jx @ G)
        ## del theta = Jc * del rollpitch
        rank = np.linalg.matrix_rank(Jc)
        error_state = 0
        if rank != 2:
            print("jac Warning:  , rank:  Jc:  ", rank, Jc)
            error_state = 1
        return Jc, error_state

    # Forward Kinematics Solver
    # Inputs:
    #   pitchRef, rollRef: Reference orientation angles for the platform.
    #   theta1, theta2: Actuator joint angles (motor positions).
    # Output:
    #   Computed actual pitch and roll angles of the end-effector.
    def fw(self , pitchRef, rollRef, theta1 , theta2 ): 
    # 使用迭代方法求解逆运动学问题
        epsilon = 1e-4  # 收敛阈值
        max_iterations = 100  # 最大迭代次数
        dt = 1.0# 0.51 # 时间步长 , is important
        pitch = pitchRef
        roll = rollRef
        error_state = 0 # no error

        for i in range(max_iterations):

            if pitch > self.pitch_range[1]:
                pitch = self.pitch_range[1]
            elif pitch < self.pitch_range[0]:
                pitch = self.pitch_range[0]

            if roll > self.roll_range[1]:
                roll = self.roll_range[1]
            elif roll < self.roll_range[0]:
                roll = self.roll_range[0]
             

            # 计算当前末端执行器的位置
            m1, m2, _  = self.ik(pitch,roll)
            if(np.isnan(m1) or np.isnan(m2)):
                error_state = 1 # error
                return 0, 0, error_state
               
            # 计算误差
            error = np.array([m1 - theta1, m2 - theta2])

            # 如果当前位置与目标位置足够接近，则停止
            if np.linalg.norm(error ) < epsilon:
                print("逆运动学收敛！")
                print(i)
                return pitch, roll, error_state

            # 计算雅可比矩阵
            Jc, _  = self.Jac(pitch,roll, m1,m2) # good than theta1,theta2

            try:
                JcInv = np.linalg.inv(Jc)
            except np.linalg.LinAlgError:
                print(" jc inv is error ")
                error_state = 1 # error
                return 0, 0, error_state

            # 计算速度
            v = JcInv @ error  #  droll dpitch
            pitch -= v[1]*dt
            roll -= v[0]*dt
        else:
            print("逆运动学未在最大迭代次数内收敛。")
            # return pitchRef, rollRef

        error_state = 1 # error
        return 0, 0, error_state

    ## test ik,fk,jac
    ## plot data
    def plotAnkle(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        m1 = []
        m2 = [] 

        pitch_range = self.pitch_range  
        roll_range = self.roll_range   

        num_roll_points = int((roll_range[1] - roll_range[0]) / np.deg2rad(2))
        num_pitch_points = int((pitch_range[1] - pitch_range[0]) / np.deg2rad(2))

        # 生成均匀分布的弧度序列
        roll_values = np.linspace(roll_range[0], roll_range[1], num_roll_points)
        pitch_values = np.linspace(pitch_range[0], pitch_range[1], num_pitch_points)

        for roll in roll_values:
            for pitch in pitch_values: 
                plt.cla()

                theta1, theta2, _ = self.ik(pitch, roll) 
                rb1 = self.ra01 + self.RyPlot(theta1 + self.m01) @ self.rab01
                rc1 = self.xrot(pitch, roll) @ self.rc01
                re1 = self.xrot(pitch, roll) @ self.re01

                rb2 = self.ra02 + self.RyPlot(theta2 + self.m02) @ self.rab02
                rc2 = self.xrot(pitch, roll) @ self.rc02
                re2 = self.xrot(pitch, roll) @ self.re02

                # # 验证逆向运动学是否正确
                if abs(np.linalg.norm((rb1-rc1)) - self.lrod1) > 0.0001:
                    print(" ik 11 error ")
                if abs(np.linalg.norm((rb2 - rc2)) - self.lrod2) > 0.0001:
                    print(" ik 21 error ")
                if abs(np.linalg.norm((rb1 - self.ra01)) - self.lbar1) > 0.0001:
                    print(" ik 12 error ")
                if abs(np.linalg.norm((rb2 - self.ra02)) - self.lbar2) > 0.0001:
                    print(" ik 22 error ")

                i = 0
                xx = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i] ])
                i = 1
                yy = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i]])
                i = 2
                zz = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i]])


                ax.plot(xx, yy, zz, c='r' ,marker='*' )

                ax.set_xlabel('x label')
                ax.set_ylabel('y label')
                ax.set_zlabel('z label')

                ax.set_xlim([-0.1, 0.1])
                ax.set_ylim([-0.1, 0.1])
                ax.set_zlim([-0.05, 0.25])
                plt.show()

                ## 验证 Jca 的正确性
                Jc, _  = self.Jac(pitch,roll,theta1,theta2)
                dpr = 0.01 * np.array([[1],[1]]) #np.random.rand(2,1)
                newtheta1, newtheta2, _  = self.ik(pitch + dpr[1][0], roll + dpr[0][0])
                dth = Jc @ dpr  #  th1,th2 = Jc @ [roll,pitch]
                dthReal = [newtheta1-theta1, newtheta2-theta2]

                if abs(dthReal[0] - dth[0]) > 0.0002 or abs(dthReal[1] - dth[1]) > 0.0002:
                    print(" jac is wrong ",abs(dthReal[0] - dth[0]),abs(dthReal[1] - dth[1]))
                pitchRef=0
                rollRef=0
                pitchTest,rollTest, _ = self.fw(pitchRef, rollRef, theta1, theta2)

                if abs(pitchTest-pitch) > 0.0005 or abs(rollTest - roll) > 0.0005:
                    print("fw is error ", abs(pitchTest-pitch) )
                    print("fw is error ",abs(rollTest - roll) )


        
    def plotAnklePoint(self,theta1,theta2):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') 
            pitch,roll, _ = self.fw(0, 0, theta1, theta2)
            # theta1, theta2 = self.ik(pitch, roll)
            rb1 = self.ra01 + self.RyPlot(theta1 + self.m01) @ self.rab01
            rc1 = self.xrot(pitch, roll) @ self.rc01
            re1 = self.xrot(pitch, roll) @ self.re01

            rb2 = self.ra02 + self.RyPlot(theta2 + self.m02) @ self.rab02
            rc2 = self.xrot(pitch, roll) @ self.rc02
            re2 = self.xrot(pitch, roll) @ self.re02

            # 验证逆向运动学是否正确
            if abs(np.linalg.norm((rb1-rc1)) - self.lrod1) > 0.001:
                print(" ik 1 error ")
            if abs(np.linalg.norm((rb2 - rc2)) - self.lrod2) > 0.001:
                print(" ik 2 error ")

            i = 0
            xx = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i] ])
            i = 1
            yy = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i]])
            i = 2
            zz = np.array([0, rc1[i], rb1[i], self.ra01[i], re1[i], rc2[i], rb2[i], self.ra02[i], re2[i], rc2[i],rc1[i],re1[i],re2[i]])


            ax.plot(xx, yy, zz, c='r' ,marker='*' )

            ax.set_xlabel('x label')
            ax.set_ylabel('y label')
            ax.set_zlabel('z label')

            ax.set_xlim([-0.1, 0.1])
            ax.set_ylim([-0.1, 0.1])
            ax.set_zlim([-0.05, 0.15])
            plt.show() 

    ## 运用速度来验证雅克比
    ## 设定固定的sin关节轨迹，通过ik可以求解电机的曲线，差分可以得到速度
    ## 通过对sin关节轨迹的速度，进行jac运算，求解速度
    # 对比雅克比计算得到的速度可以确认雅克比的正确性

    def testJac(self):

        motorSave1 = []
        motorSave2 = []
        velSave1 = []
        velSave2 = []
        dt = 0.002
        for t in range(1000):
            time = t * dt
            pitch = 0.6 * np.sin(time * 2 * np.pi)
            roll = 0.3 * np.cos(time*2*4*np.pi)
            velPitch = 0.6*2*np.pi * np.cos(time * 2 * np.pi)
            velRoll = -0.3*2*np.pi*4* np.sin(time * 2 * 4 * np.pi)

            theta1, theta2, _ = self.ik(pitch, roll)
            motorSave1.append(theta1)
            motorSave2.append(theta2)

            Jc, _ = self.Jac(pitch,roll,theta1,theta2)

            v = Jc @ np.array([velRoll,velPitch])
            velSave1.append(v[0])
            velSave2.append(v[1])

        motorSave1 = np.array(motorSave1)
        motorSave2 = np.array(motorSave2)

        mv1 = (motorSave1[1:] - motorSave1[0:-1]) / dt
        mv2 = (motorSave2[1:] - motorSave2[0:-1]) / dt

        plt.figure()
        plt.plot(mv1,'r',label=' diffence vel ')
        plt.plot(velSave1, 'k', label=' jac vel ')
        plt.title(" motor vel 1")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(mv2, 'r', label=' diffence vel ')
        plt.plot(velSave2, 'k', label=' jac vel ')
        plt.title(" motor vel 2")
        plt.legend()
        plt.show()

    ## 力的解算     moror force  --> Jac.T --> joint force
    ## 速度计算    joint vle --> Jac --> motor vel
    def forceVelJac(self,pitch,roll,theta1,theta2):
        Jc, _ = self.Jac(pitch,roll,theta1,theta2)

        jointVel = np.array([0.1,0.4])# joint vel  roll，pitch vel
        motorVel = Jc @ jointVel

        motorForce = np.array([0.1,0.2]) # motor force
        jointForce = Jc.transpose() @ motorForce
        return  1

# 以左腿为例，左侧连杆电机为1，右侧电机为2，两电机上转为正，此时脚尖向下运动，
# 因为x指向前（roll方向），y是右腿指向左腿（pitch方向）
# 右腿和左腿的情况为镜像关系，右侧电机为1，左侧电机为2，两电机上转为正，此时脚尖向下，
# roll方向也是向前，通过左腿这个程序计算出来的roll取反，达到镜像的效果
# ab线连接保持水平，就是电机零点

import os
if __name__=="__main__": 

    parTest = parallelAnkle( )
 
    left_m1, right_m2, _ = parTest.ik(0, 0)
    print("left_m1, right_m2: ", left_m1, right_m2)
    parTest.plotAnklePoint(0, 0)
    pitch = 0.468
    roll = 0
    left_m1, right_m2, _  = parTest.ik(pitch, roll )
    print("left_m1, right_m2: ", left_m1, right_m2)
    parTest.plotAnklePoint(pitch, roll)
    pitch, roll, state = parTest.fw(0,  0, left_m1, right_m2)
    pitch1, roll1, state1 = parTest.fw(0,  0, -0.446, 0.446)
    parTest.plotAnkle()

    dt = 0.001
    T = 10000   
    q_pos = [] 
    count = 0
    for i in range(T): 
        sin_data = 0.4 * np.sin(2* np.pi *0.6 * dt * count)
        # data.ctrl[0] = sin_data #  right_motor_joint 
        # data.ctrl[1] = 0 #0.5*sin_data  # left_motor_joint 
        # pitch =  sin_data
        # roll = 0.5*sin_data #roll
        # left_motor, right_motor, state = parTest.ik(pitch, roll)
        
        left_motor = sin_data
        right_motor = 0.5*sin_data

        pitch, roll, state = parTest.fw(0,  0, left_motor, right_motor)
        q_pos.append([pitch , roll , left_motor , right_motor , state ])
        count += 1 

    folder_path = current_dir + '/parallel_robots_ankle/5_tiangong/'    
    np.savez(folder_path + 'paral_tg10.npz',     
              qpos=np.array(q_pos)  )    
    print(1)
