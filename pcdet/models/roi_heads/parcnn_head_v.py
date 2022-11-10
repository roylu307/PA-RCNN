import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils, box_utils
from .roi_head_template import RoIHeadTemplate

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils


class VoxelRCNNHead_v1(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0

        self.raw_point_pool_layer, num_c_out_raw = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=1, config=self.model_cfg.ROI_GRID_POOL.RAW_POINT_LAYER
        )
        c_out += num_c_out_raw

        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)

        #######
        raw_points = batch_dict['points']
        _, complement_pos_features = self.get_complement_pos_features(
            rois=rois, points=raw_points, new_xyz=roi_grid_xyz
        )
        # print(complement_pos_features)


        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        ms_pooled_features = torch.cat([complement_pos_features, ms_pooled_features], dim=-1)
        
        return ms_pooled_features

    def get_complement_pos_features(self, rois, points, new_xyz):
        """
        Args:
            self
            rois: (B, num_rois, 7 + C)
            points: (num_rawpoints, 4) [bs_idx, x, y, z]
            new_xyz: (BxN, 6x6x6, 3)
        Returns:

        """

        batch_size = rois.shape[0]
        num_rois = rois.shape[1]

        batch_idx = points[:, 0]
        raw_xyz = points[:, 1:]

        sampled_points_list = []
        xyz_roi_cnt = []

        for bs_idx in range(batch_size):
            
            bs_mask = batch_idx == bs_idx

            sampled_points = raw_xyz[bs_mask][None, :, :] # (1, N, 3+C)
            sampled_rois = rois[bs_idx][None, :, :] # (1, num_rois(M), 7+C)
            
            import numpy as np
            # sample_extra_width = np.array([0.4])
            sample_extra_width = self.model_cfg.ROI_GRID_POOL.RAW_POINT_LAYER.ROI_EXTRA_WIDTH
            sample_extra_width = sample_extra_width if len(sample_extra_width) == 3 else sample_extra_width[0] * np.array([1, 1, 1])

            enlarged_rois = box_utils.enlarge_box3d(sampled_rois.squeeze(dim=0),\
                                                    extra_width=sample_extra_width).unsqueeze(dim=0)
            
            point_assignment = roipoint_pool3d_utils.points_in_boxes_gpu(
                sampled_points[:, :, :3], enlarged_rois[:, :, 0:7]) # (1, N, M)
            point_mask = point_assignment.squeeze(0).permute(1,0)==1 # (M, N)
            num_point_in_rois_stack = point_mask.sum(dim=1)
            
            empty_flag = num_point_in_rois_stack == 0
            
            #### add point transformation here
            sampled_points, num_point_in_rois_stack = self.get_tranformed_points(
                sampled_points, enlarged_rois, point_mask, empty_flag, num_point_in_rois_stack)
            
            # sampled_points = torch.cat([(sampled_points.new_zeros(0, sampled_points.shape[-1]) if empty_flag[i] else sampled_points.squeeze(0)[point_mask[i,:],:]) \
            #                             for i in range(num_rois)], dim=0) # (R1 + R2 ..., 3)
            sampled_points_list.append(sampled_points)
            xyz_roi_cnt.append(num_point_in_rois_stack.int())

        batch_sampled_points = torch.cat(sampled_points_list, dim=0)
        batch_xyz_roi_cnt = torch.cat(xyz_roi_cnt, dim=0)
        new_xyz_roi_cnt = (new_xyz.new_ones(new_xyz.shape[0]) * new_xyz.shape[1]).int()

        pooled_points, pooled_features = self.raw_point_pool_layer(
            xyz=batch_sampled_points[:, :3].contiguous(),
            xyz_batch_cnt=batch_xyz_roi_cnt,
            new_xyz=new_xyz.view(-1, 3),
            new_xyz_batch_cnt=new_xyz_roi_cnt,
            features=batch_sampled_points[:, 3:].contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)

        return pooled_points, pooled_features

    def get_tranformed_points(self, points, rois, point_mask, empty_flag, num_point_in_rois_stack):
        """
        Args:
            self
            points: (1, N, 4)
            rois: (1, num_rois, 7 + C)
            point_mask: (num_rois, N) 
            empty_flag: (num_rois)
            num_point_in_rois_stack: (num_rois) point count in rois
        Returns:

        """

        # points = points.view(-1, points.shape[-1])
        # rois = rois.view(-1, rois.shape[-1])

        transformed_points_list = []

        # if complement method
        num_rois = rois.shape[1]

        for i in range(num_rois):
            if empty_flag[i]:
                # transformed_points = points.new_zeros(1, points.shape[-1])
                transformed_points = points[:, 0, :]
                num_point_in_rois_stack[i] = 1
            else:
                transformed_points = points[:, point_mask[i,:], 0:3]
                transformed_features = points[:, point_mask[i,:], 3:]
                
                roi_center = rois[:, i, 0:3].clone()
                transformed_points -= roi_center
                
                complement_points = common_utils.rotate_points_along_z(
                    transformed_points.clone(), -rois[:, i, 6])
                
                pt_transform_type = self.model_cfg.ROI_GRID_POOL.RAW_POINT_LAYER.get('POINT_TRANSFORM_TYPE', None)
                if pt_transform_type == 'flip':
                    complement_points[:, :, 1] = -complement_points[:, :, 1]
                elif pt_transform_type == 'rotate':
                    complement_points[:, :, :1] = -complement_points[:, :, :1]
                elif pt_transform_type == 'raw':
                    complement_points = complement_points
                else:
                    raise NotImplementedError

                complement_points += roi_center
                complement_points = common_utils.rotate_points_along_z(
                        complement_points.clone(), rois[:, i, 6])

                feat_transofrm_type = self.model_cfg.ROI_GRID_POOL.RAW_POINT_LAYER.get('FEATURE_TRANSFORM_TYPE', 'equal')
                if feat_transofrm_type == 'inverted':
                    complement_features = -transformed_features
                elif feat_transofrm_type == 'equal':
                    complement_features = transformed_features
                elif feat_transofrm_type == 'half':
                    complement_features = 0.5 * transformed_features
                else:
                    raise NotImplementedError
                
                complement_points = torch.cat([complement_points, complement_features], dim=-1)
                transformed_points = torch.cat([points[:, point_mask[i,:], :], complement_points], dim=1).squeeze(0)
                num_point_in_rois_stack[i] = num_point_in_rois_stack[i]*2 # double point counts in rois
                
            transformed_points_list.append(transformed_points)

        transformed_points = torch.cat(transformed_points_list, dim=0)
        

        return transformed_points, num_point_in_rois_stack

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
