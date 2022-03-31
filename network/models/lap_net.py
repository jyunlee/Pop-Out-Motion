import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DFAUSTDataset
from models.base import BaseClassification
from point_transformer_ops.point_transformer_modules import PointTransformerBlock, TransitionDown, TransitionUp


class LapNet(BaseClassification):

    def _build_model(self, dim=[3,64,64,128,256,512], output_dim=1, pos_mlp_hidden=64, attn_mlp_hidden=4, k=16, sampling_ratio=0.25):

        self.Encoder = nn.ModuleList()

        for i in range(len(dim)-1):

            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], k, sampling_ratio, fast=True))

            self.Encoder.append(PointTransformerBlock(dim[i+1], k))

        self.Decoder = nn.ModuleList()

        for i in range(5,0,-1):
            if i == 5:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(dim[i+1], dim[i]))

            self.Decoder.append(PointTransformerBlock(dim[i], k))


        self.l_layer = nn.Sequential(
            nn.Conv1d(dim[1]+3, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=1, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, output_dim, kernel_size=1),
        )

        self.w_layer = nn.Sequential(
            nn.Conv1d(dim[1]+3, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=1, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.minv_layer = nn.Sequential(
            nn.Conv1d(dim[1]+3, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, output_dim, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, pointcloud, ref_pt, knn, in_features=None, feature_only=False):

        features = xyz = pointcloud

        if in_features is None: 

            l_xyz, l_features = [xyz], [features]

            for i in range(int(len(self.Encoder)/2)):
                if i == 0:
                    li_features = self.Encoder[2*i](l_features[i])
                    li_xyz = l_xyz[i]
                else:
                    li_features, li_xyz = self.Encoder[2*i](l_features[i], l_xyz[i].contiguous())
                li_features = self.Encoder[2*i+1](li_features, li_xyz)

                l_features.append(li_features)
                l_xyz.append(li_xyz)
                del li_features, li_xyz            

            D_n = int(len(self.Decoder)/2)

            for i in range(D_n):
                if i == 0:
                    l_features[D_n-i] = self.Decoder[2*i](l_features[D_n-i])
                    l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
                else:
                    l_features[D_n-i], l_xyz[D_n-i] = self.Decoder[2*i](l_features[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                    l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])

            del l_features[0], l_features[1:], l_xyz

            pt_features = l_features[0].transpose(1,2)

        else:
            pt_features = in_features

        if feature_only:
            return pt_features
 
        pt_xyz = xyz.transpose(1,2)

        ref_pt_xyz = torch.zeros(pt_xyz.shape).cuda()
        ref_pt_features = torch.zeros(pt_features.shape).cuda()
        
        for i in range(pt_features.shape[0]):
            ref_pt_coord = pt_xyz[i, :, ref_pt[i]]
            ref_pt_coord = ref_pt_coord.reshape(1, pt_xyz.shape[1], 1) 
            ref_pt_xyz[i] = ref_pt_coord.repeat(1, 1, pt_xyz.shape[2])

            ref_pt_feature = pt_features[i, :, ref_pt[i]]
            ref_pt_feature = ref_pt_feature.reshape(1, pt_features.shape[1], 1)
            ref_pt_features[i] = ref_pt_feature.repeat(1, 1, pt_features.shape[2])

        concat_input = torch.cat((torch.abs(pt_xyz - ref_pt_xyz), torch.mul(pt_features, ref_pt_features)), dim=1)

        knn_concat_input = torch.zeros([concat_input.shape[0], concat_input.shape[1], knn.shape[1]]).cuda()

        for i in range(concat_input.shape[0]):
            knn_concat_input[i] = concat_input[i, :, knn[i]]

        l = self.l_layer(knn_concat_input)
        w = self.w_layer(knn_concat_input)

        minv_input = torch.cat((pt_xyz, pt_features), dim=1)

        sampled_minv_input = torch.zeros([minv_input.shape[0], minv_input.shape[1]]).cuda()

        for i in range(minv_input.shape[0]):
            sampled_minv_input[i, :] = minv_input[i, :, ref_pt[i]]

        minv = self.minv_layer(sampled_minv_input.unsqueeze(2))

        l = l * w
        
        return l.squeeze(), minv.squeeze(), w.squeeze(), pt_features


    def prepare_data(self):
        self.train_dset = DFAUSTDataset(split='train', data_root='/workspace/Pop-Out-Motion/data/')
        self.val_dset = DFAUSTDataset(split='val', data_root='/workspace/Pop-Out-Motion/data/')

 
    def training_step(self, batch, batch_idx):

        pc, ref_pt_idx, l_label, minv_label, knn_list = batch
        pc, ref_pt_idx, l_label, minv_label, knn_list = pc.float(), ref_pt_idx.long(), l_label.float(), minv_label.float(), knn_list.long()

        l_pred, minv_pred, w_pred, _ = self.forward(pc, ref_pt_idx, knn_list)

        knn_l_label = torch.zeros(knn_list.shape).cuda()

        for i in range(knn_l_label.shape[0]):
          knn_l_label[i] = l_label[i, knn_list[i]]

        w_label = (knn_l_label != 0).float()

        l_loss = F.mse_loss(l_pred.squeeze(), knn_l_label.squeeze())
        w_loss = F.mse_loss(w_pred.squeeze(), w_label.squeeze())
        m_loss = F.mse_loss(minv_pred.squeeze().float(), minv_label.squeeze().float())

        loss = l_loss + 100 * w_loss + m_loss

        ''' temp '''
        if batch_idx % 10 == 0:
            print('\n', l_pred.squeeze()[0])
            print(knn_l_label.squeeze()[0])
            print('\n', minv_pred.squeeze())
            print(minv_label.squeeze())

        log = dict(train_loss=loss, train_miou=loss)

        return dict(loss=loss, log=log, progress_bar=dict(train_miou=loss))


    def validation_step(self, batch, batch_idx):

        pc, ref_pt_idx, l_label, minv_label, knn_list = batch
        pc, ref_pt_idx, l_label, minv_label, knn_list = pc.float(), ref_pt_idx.long(), l_label.float(), minv_label.float(), knn_list.long()

        l_pred, minv_pred, w_pred, _ = self.forward(pc, ref_pt_idx, knn_list)

        knn_l_label = torch.zeros(knn_list.shape).cuda()

        for i in range(knn_l_label.shape[0]):
          knn_l_label[i] = l_label[i, knn_list[i]]

        w_label = (knn_l_label != 0).float()

        l_loss = F.mse_loss(l_pred.squeeze(), knn_l_label.squeeze())
        w_loss = F.mse_loss(w_pred.squeeze(), w_label.squeeze())
        m_loss = F.mse_loss(minv_pred.squeeze().float(), minv_label.squeeze().float())

        loss = l_loss +  m_loss

        return dict(val_loss=loss, val_acc=-1*loss)


