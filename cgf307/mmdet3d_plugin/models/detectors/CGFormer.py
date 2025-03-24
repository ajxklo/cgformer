import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder

from mmdet3d_plugin.models.self_add.flow.NeighborhoodCrossAttention import NeighborhoodCrossAttention
from mmdet3d_plugin.models.self_add.flow.flow import  OpticalFlowAlignmentWithAttention
from mmdet3d_plugin.models.self_add.vipocc.DepthRefinement import DepthRefinement
from mmdet3d_plugin.models.self_add.vipocc.RID import RID
from mmdet3d_plugin.models.self_add.vipocc.model.time_loss import time_loss
import torch.nn.functional as F


def compute_cosine_similarity(feat1, feat2):
    """计算余弦相似度"""
    num = feat2.dim()
    if num == 5:
        b, n, c, h, w = feat2.shape
        feat2 = feat2.view(b * n, c, h, w)
    cos_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-8)
    return cos_sim.unsqueeze(1)  # 添加通道维度
@DETECTORS.register_module()
class CGFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        '''
        vipocc
        '''
        self.depth_refine = DepthRefinement()
        self.rev_inv_depth = RID(resnet_layers=18)
        self.time_loss = time_loss()
        self.depth_loss = True
        self.flow = OpticalFlowAlignmentWithAttention()
        self.nca = NeighborhoodCrossAttention(kernel_size=3)
#===================================================================================


        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        # devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(img_inputs) == 9:
            img_enc_feats_last = self.image_encoder(img_inputs[-1]).detach()
            img_enc_feats = self.image_encoder(img_inputs[0])
            warped_feats,mask = self.flow(img_enc_feats_last, img_metas["flow"])
            sim = compute_cosine_similarity(warped_feats,img_enc_feats)
            flow_loss = 1 - sim
            flow_loss = flow_loss.mean()
            img_enc_feats = self.nca(img_enc_feats,warped_feats,mask.unsqueeze(0)).unsqueeze(0)
            # img_enc_feats = img_enc_feats.unsqueeze(0)
        else:
            img_enc_feats = self.image_encoder(img_inputs[0])
            # img_inputs[0] = temporal_feature_aggregation(img_inputs[0],img_inputs[-1],img_metas["flow"])
            # img_inputs1 = temporal_feature_aggregation(img_inputs[0],img_inputs[-1],img_metas["flow"])
            # img_metas['stereo_depth'][0] = temporal_feature_aggregation(img_metas['stereo_depth'][0],img_metas['stereo_depth'][1],img_metas["flow"])

        # if len(img_inputs) == 9:
        #     data = {}
        #     data["predicted_depth"] = img_metas['stereo_depth']
        #     data["imgs1"] = img_inputs[0]
        #     data["imgs2"] = img_inputs[-1]
        #     data["poses"] = img_metas['pose']
        #     data["k"] = img_metas['k']
        #     data['k1'] = img_metas['k1']
        #     loss_time = self.time_loss(data).to(devices)
        # else:
        #     loss_time = torch.tensor(1e-6).float().to(devices)

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])

        stereo_depth = img_metas['stereo_depth'][0]
        B,N,C,H,W = img_inputs[0].shape
        x = img_inputs[0].view(B*N,C,H,W)
        rev_inv_depth = self.rev_inv_depth(x)
        stereo_depth = self.depth_refine(stereo_depth,rev_inv_depth)
        # img_metas['stereo_depth'] = stereo_depth

        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas,stereo_depth)
        
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=img_inputs[1:7],
            lss_volume=coarse_queries,
            img_metas=img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)]
        )

        return x, depth,flow_loss
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth,flow_loss = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()
        losses['flow_loss'] = flow_loss
        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, _ = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)