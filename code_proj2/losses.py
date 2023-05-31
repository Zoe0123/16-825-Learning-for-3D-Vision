import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	sigmoid = torch.nn.Sigmoid()

	func = torch.nn.BCELoss() 
	loss = func(sigmoid(voxel_src),voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	point_cloud_src_cpy, point_cloud_tgt_cpy = point_cloud_src, point_cloud_tgt
	n_src = torch.full((point_cloud_src_cpy.shape[0],), point_cloud_src_cpy.shape[1], dtype=torch.int64, device=point_cloud_src.device)
	n_tgt = torch.full((point_cloud_tgt_cpy.shape[0],), point_cloud_tgt_cpy.shape[1], dtype=torch.int64, device=point_cloud_tgt.device)
	
	src_nn = knn_points(point_cloud_src, point_cloud_tgt, lengths1=n_src, lengths2=n_tgt, norm=2, K=1)
	tgt_nn = knn_points(point_cloud_tgt, point_cloud_src, lengths1=n_tgt, lengths2=n_src, norm=2, K=1)

	cham_x = src_nn.dists[..., 0].sum(1)
	cham_y = tgt_nn.dists[..., 0].sum(1)

	loss_chamfer = torch.mean(cham_x + cham_y)
	
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src) # uniform?
	# implement laplacian smoothening loss
	return loss_laplacian