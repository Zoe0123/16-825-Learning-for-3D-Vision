# 16-825 Assignment 1: Rendering Basics with PyTorch3D (Instructions to Run)

## 0. Setup

You will need to install Pytorch3d. See the directions for your platform
[here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
You will also need to install Pytorch. If you do not have a GPU, you can directly pip
install it (`pip install torch`). Otherwise, follow the installation directions
[here](https://pytorch.org/get-started/locally/).

Other miscellaneous packages that you will need can be installed using the 
`requirements.txt` file (`pip install -r requirements.txt`).

If you have access to a GPU, the rendering code may run faster, but everything should
be able to run locally on a CPU.

## 1. Practicing with Cameras

### 1.1. 360-degree Renders (5 points)
Run `python -m starter.main --question 1.1`

### 1.2 Re-creating the Dolly Zoom (10 points)
Run `python -m starter.dolly_zoom`

## 2. Practicing with Meshes   

### 2.1 Constructing a Tetrahedron (5 points)
Run `python -m starter.main --question 2.1`

### 2.2 Constructing a Cube (5 points)
Run `python -m starter.main --question 2.2`

## 3. Re-texturing a mesh (10 points)
Run `python -m starter.main --question 3`

## 4. Camera Transformations (10 points)
Run `python -m starter.camera_transforms`

## 5. Rendering Generic 3D Representations
### 5.1 Rendering Point Clouds from RGB-D Images (10 points)
Run `python -m starter.render_generic --question 5.1`

### 5.2 Parametric Functions (10 points)
Run `python -m starter.render_generic --question 5.2`

### 5.3 Implicit Surfaces (15 points)
Run `python -m starter.render_generic --question 5.3`

## 6. Do Something Fun (10 points)
Run `python -m starter.main --question 5`