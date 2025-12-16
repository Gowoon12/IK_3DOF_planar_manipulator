# IK_3DOF_planar_manipulator

+ This project was developed at the Medical Robotics and Intelligent Control Laboratory(MeRIC-LAB), Department of Mechanical Engineering, Chonnam National University, as part of the hands-on exercises for the course “AI-based Mechanical Systems”. 

+ The project is designed to provide practical experience with learning-based inverse kinematics for a 3-DOF planar robot, allowing students to explore and compare learning-based approaches with analytical inverse kinematics methods. 

+ This package has been tested in the following environments:
  
   (1) Python 3.9.23, PyTorch 2.8.0, CPU-only
  
   (2) Python 3.12, PyTorch 2.6.0, NVIDIA H100E GPU, CUDA 12.8
  

## Requirements 

+ All required Python libraries are listed in the requirements.txt file and can be installed using:

      pip install -r requirements.txt
      

## Program Description

+ The Jupyter Notebook (.ipynb) and Python script (.py) cover the same content.

### workspace_3dof.ipynb 

+ These files define a 3-DOF planar robot and visualize its workspace based on analytical forward kinematics.

+ They also provide visualization of the robot’s links and joints, allowing intuitive understanding of the robot structure and reachable workspace.

### IK_3dof_1.ipynb 

+ IK_3dof_1.ipynb implements a learning-based inverse kinematics solver for a 3-DOF planar robot using an MLP trained on FK-generated data without considering multiple IK solutions.

+ The learned IK results are evaluated against analytical inverse kinematics through joint-level error metrics and forward-kinematics-based visualization.

### IK_3dof_2.ipynb 

+ This script trains a learning-based inverse kinematics model for a 3-DOF planar robot using an elbow-down constrained FK-generated dataset.

+ The trained MLP is quantitatively evaluated on a test set and qualitatively compared with analytical inverse kinematics through forward-kinematics-based visualization.

+ IK_3dof_1.ipynb and IK_3dof_2.ipynb use the same network architecture, training procedure, and hyperparameters.

+ The only difference lies in the dataset generation strategy, where IK_3dof_2.ipynb applies a joint constraint to remove multiple inverse kinematics solutions.
