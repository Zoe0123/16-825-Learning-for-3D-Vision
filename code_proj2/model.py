from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

# # get the idea from the resource: https://arxiv.org/pdf/1901.11153.pdf
class VoxDecoder(nn.Module):
    def __init__(self):
        super(VoxDecoder, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(4, 1, kernel_size=1),
            nn.BatchNorm3d(1)
        )

    def forward(self, feats):
        vox = feats.view((-1, 64, 2, 2, 2))
        vox = self.layer1(vox)
        vox = self.layer2(vox)
        vox = self.layer3(vox)
        vox = self.layer4(vox)
        vox = self.layer5(vox)

        return vox

# class VoxDecoder(nn.Module):
#     def __init__(self, vox_size):
#         super(VoxDecoder, self).__init__()
#         self.vox_size = vox_size
#         self.layer = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 5096),
#             nn.BatchNorm1d(5096),
#             nn.ReLU(),
#             nn.Linear(5096, 10192),
#             nn.BatchNorm1d(10192),
#             nn.ReLU(),
#             # nn.Linear(10192, 20384),
#             # nn.BatchNorm1d(20384),
#             # nn.ReLU(),
#             nn.Linear(10192, self.vox_size**3),
#             # nn.Tanh()
#         )

#     def forward(self, feats):
#         voxes = self.layer(feats)
#         voxes = voxes.view(-1, 1, self.vox_size, self.vox_size, self.vox_size)

#         return voxes

# get idea from the resource: PointNetFCAE
class PointDecoder(nn.Module):
    def __init__(self, point_size):
        super(PointDecoder, self).__init__()
        self.point_size = point_size
        self.layer = nn.Sequential(
            nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            nn.Linear(1024, self.point_size*3),
            # nn.Tanh()
        )

    def forward(self, feats):
        points = self.layer(feats)
        points = points.reshape((-1, self.point_size, 3))

        return points

class MeshDecoder(nn.Module):
    def __init__(self, vert_size):
        super(MeshDecoder, self).__init__()
        self.vert_size = vert_size
        self.layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.vert_size * 3)
        )

    def forward(self, feats):
        meshes = self.layer(feats)
        meshes = meshes.view(-1, self.vert_size, 3)

        return meshes



class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            # pass
            # TODO:
            # self.decoder = VoxDecoder(32)  
            self.decoder = VoxDecoder()      
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = PointDecoder(self.n_point)         
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  # 2562
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = MeshDecoder(mesh_pred.verts_packed().shape[0])            

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)           
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)           
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)       
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

