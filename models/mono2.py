import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from components import get_smooth_loss
from .disp_net import DispNet
from .layers import SSIM, Backproject, Project
from .utils import *
from .registry import MODELS


@MODELS.register_module(name='mono2')
class Mono2Model(LightningModule):
    """
    The training process
    """
    def __init__(self, opt):
        super(Mono2Model, self).__init__()

        self.opt = opt.model

        # components
        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project_3d = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)

        # networks
        self.net = DispNet(self.opt)

    def forward(self, inputs):
        return self.net(inputs)

    def training_step(self, batch_data, batch_idx):
        # outputs
        outputs = self.net(batch_data)

        # loss for ego-motion
        disp_loss_dict = self.compute_disp_losses(batch_data, outputs)
        disp_loss = sum(disp_loss_dict.values())

        # log
        self.log('train/loss', disp_loss, logger=True, on_epoch=True, on_step=False)

        # return
        return disp_loss

    def configure_optimizers(self):
        optim = Adam(self.net.parameters(), lr=self.opt.learning_rate)
        sch = MultiStepLR(optim, milestones=[15], gamma=0.5)

        return [optim], [sch]

    def get_color_input(self, inputs, frame_id, scale):
        return inputs[("color_equ", frame_id, scale)] if self.opt.use_equ else inputs[("color", frame_id, scale)]

    def generate_images_pred(self, inputs, outputs, scale):
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs["inv_K", 0])
            pix_coords = self.project_3d(cam_points, inputs["K", 0], T)  # [b,h,w,2]
            src_img = self.get_color_input(inputs, frame_id, 0)
            outputs[("color", frame_id, scale)] = F.grid_sample(src_img.double(), pix_coords, padding_mode="border",
                                                                align_corners=False)

        return outputs

    def undistort_point(self, K, pt, depth):
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        assert all(K[:, 0, 1] == 0) and all(K[:, 1, 0] == 0)
        assert all(K[:, 2, 2] == 1) and all(K[:, 2, 0] == 0)
        assert all(K[:, 2, 1] == 0) 
        x = pt[:, :, 3]
        y = pt[:, :, 4]
        batch_size = len(x)
        pts3d = torch.zeros((batch_size, len(x[0]), 3))
        # not very efficient
        # but this is sparse anyway
        for b in range(batch_size):
            for i, (xi, yi) in enumerate(zip(x[b], y[b])):
                # not sure how to use off the shelf 
                # with preserving gradient
                # hence we will do this the odd fashion way
                zi = depth[b, 0, yi, xi]
                pts3d[b, i, 0] = zi*(xi - cx[b]) / fx[b]
                pts3d[b, i, 1] = zi*(yi - cy[b]) / fy[b]
                pts3d[b, i, 2] = zi
        return pts3d

    def get_batch_cross_product_matrix(self, t):
        assert(t.shape[1] == 3)
        batch_size = t.shape[0]
        result = torch.zeros((batch_size, 3, 3)).to(t.get_device())
        for i in range(batch_size):
            x = t[i, 0]
            y = t[i, 1]
            z = t[i, 2]
            result[i, 1, 0] = -z
            result[i, 2, 0] = y
            result[i, 0, 1] = z
            result[i, 2, 1] = -x
            result[i, 0, 2] = -y
            result[i, 1, 2] = x
            """
            C = [[0, -z, y],
                [z, 0, -x],
                [-y, x, 0]]
            """
        return result

    def compute_epipolar_loss(self, inputs, outputs, scale):
        """
        compute epipolar loss
        """
        orig_frame_id = self.opt.frame_ids[0]
        disp = outputs[("disp", orig_frame_id, scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        # get and undistort the points
        # note that we want to use the original resolution regardless
        # of the scale to yield a more accurate result
        pt0 = inputs[("point_correspondence", orig_frame_id, 0)]
        # build index dict  
        #for pt in pt0:
        #    print(pt[:10])
        #exit(0)
        pt0_idx = [{x[0].cpu().item():i for i, x in enumerate(pt)} for pt in pt0]
        K = inputs["K", 0]
        # undistort the points
        # not sure how to do this with cv2 and preserve the gradient
        # we will do it the old fashioned way
        pts3d_0 = self.undistort_point(K, pt0, depth)
        
        #print(pts3d_0.requires_grad)
        epipolar_losses = []
        item_counts = 0

        
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            # get the corresponding depth map
            disp = outputs[("disp", frame_id, scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth1 = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # get and undistort the points
            pt1 = inputs[("point_correspondence", frame_id, 0)]
            pts3d_1 = self.undistort_point(K, pt1, depth1)
            # preallocate and compute count
            batch_size = len(pt1)
            max_track_size = 0
            for b in range(batch_size):
                counter = 0
                for i1 in pt1[b]:
                    track_id = i1[0].cpu().item()
                    if track_id != 0 and track_id in pt0_idx[b]:
                        counter += 1
                max_track_size = max(max_track_size, counter)
                if counter != 0:
                    item_counts += counter
            
            # filter the points
            pts3d_0_filtered = torch.zeros((batch_size, max_track_size, 3)).to(pt0.get_device())
            pts3d_1_filtered = torch.zeros((batch_size, max_track_size, 3)).to(pt0.get_device())
            for b in range(batch_size):
                counter = 0 
                for i1, pt3d_1 in zip(pt1[b], pts3d_1[b]):
                    track_id = i1[0].cpu().item()
                    if track_id != 0 and track_id in pt0_idx[b]:
                        n0 = pt0_idx[b][track_id] # list index of point
                        pts3d_0_filtered[b, counter] = pts3d_0[b, n0]
                        pts3d_1_filtered[b, counter] = pt3d_1
                        counter += 1
            #pts3d_0_filtered = torch.stack(pts3d_0_filtered)
            #pts3d_1_filtered = torch.stack(pts3d_1_filtered)
            assert pts3d_0_filtered.shape == pts3d_1_filtered.shape
            #exit(0)
            if pts3d_0_filtered.shape[1] == 0:
                #epipolar_losses.append(0)
                print("did not find correspondence")
                #epipolar_losses.append(0.0)
                pass
            else:
                #print("found some correspondence")
                # compute the epipolar matrix
                T = outputs[("cam_T_cam", 0, frame_id)]
                R = T[:, :3, :3]
                t = T[:, :3, 3]
                t = t.unsqueeze(2)
                R_T = torch.transpose(R, 1, 2)
                # not sure why but in experimentation this rotation is necessary
                t = torch.matmul(R_T, t)
                pts3d_1_pred = torch.matmul(R, torch.transpose(pts3d_0_filtered, 1, 2)) 
                t = torch.transpose(t, 1, 2)
                pts3d_1_pred = torch.transpose(pts3d_1_pred, 1, 2) + t
                epipolar_losses.append(robust_l1(pts3d_1_pred, pts3d_1_filtered))
                """
                t_cross = self.get_batch_cross_product_matrix(t)
                E = torch.matmul(R, t_cross)
                # compute the epipolar loss
                loss = torch.matmul(pts3d_1_filtered, E)
                print(loss.shape)
                pts3d_0_filtered = torch.transpose(pts3d_0_filtered, 1, 2)
                loss = torch.matmul(E, pts3d_0_filtered)
                print(loss.shape)
                print(pts3d_1_filtered.shape)
                exit(0)
                epipolar_losses.append(loss.view(-1))
                """
        result_loss = torch.zeros(1).to(pt0.get_device())
        #assert len(epipolar_losses) == len(item_counts), "len epi {}, len item{}".format(len(epipolar_losses), len(item_counts))
        for l in epipolar_losses:
            result_loss += l.sum()
        #epipolar_losses = torch.as_tensor(epipolar_losses).to(pt0.get_device())
        #print(result_loss.requires_grad)
        #print(result_loss)
        return (1e-9)*result_loss/item_counts
            
            
        """
            # compute predicted 3d points
            if self.training:
                # self.opt.frame_ids[0] is the frame id of the first frame
                pt_orig = inputs[("point_correspondence", self.opt.frame_ids[0], scale)]
                # new points
                pt_fake_true = inputs[("point_correspondence", frame_id, scale)]
                # filter out the correspondences
                filtered_pt_0 = []
                filtered_pt_1 = []
                for i_orig, i_ft in zip(pt_orig, pt_fake_true):
                    if i_orig[0] == i_ft[0]:
                        # get the depth
                        filtered_pt_0.append((i_orig))
                        filtered_pt_1.append(i_ft)
                print(pt.shape) 
                R = T[:, :3, :3]
                t = T[:, 3, :3]
                pt_pred = torch.matmul(R, pt_orig.T).T + t
                print(R.shape, t.shape)
                print(R.requires_grad, t.requires_grad)
                print(frame_id)
                print(self.opt.frame_ids)
                #outputs[("pred_points", frame_id, scale)] = self.project_3d(cam_points, inputs["K", 0], T, pt)
                print(inputs.keys())
        """

    def compute_reprojection_loss(self, pred, target):
        #print(pred.size(), target.size())

        photometric_loss = robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_disp_losses(self, inputs, outputs):
        loss_dict = {}
        for scale in self.opt.scales:
            """
            initialization
            """
            disp = outputs[("disp", 0, scale)]
            target = self.get_color_input(inputs, 0, 0)
            reprojection_losses = []

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            automask
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = self.get_color_input(inputs, frame_id, 0)
                color_diff = self.compute_reprojection_loss(pred, target)
                identity_reprojection_loss = color_diff + torch.randn(color_diff.shape).type_as(color_diff) * 1e-5
                reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, _ = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            epipolar loss
            """
            if True:
                loss_dict[("epipolar_loss", scale)] = self.compute_epipolar_loss(inputs, outputs, scale)

            """
            disp mean normalization
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            smooth loss
            """
            smooth_loss = get_smooth_loss(disp, self.get_color_input(inputs, 0, scale))
            loss_dict[('smooth_loss', scale)] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale) / len(
                self.opt.scales)
        #for k, v in loss_dict.items():
        #    print("k is {}, v is {}".format(k, v))
        return loss_dict
