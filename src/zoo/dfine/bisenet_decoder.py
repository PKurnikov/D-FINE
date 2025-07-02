import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv
import torch

from ...core import register

__all__ = ['SpatialPath', 'BiSeNetDecoder']

@register()
class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d): #nn.BatchNorm2d
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False, has_coords=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False, has_coords=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False, has_coords=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False, has_coords=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output

class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False, has_coords=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm

class BatchRenormalization2D(nn.Module):

    def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step = 0.0001):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor( (momentum), requires_grad = False)

        self.weight = torch.nn.Parameter(torch.ones((num_features)), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((num_features)), requires_grad=True)

        self.running_mean = torch.ones((num_features), requires_grad=False)
        self.running_var = torch.zeros((num_features), requires_grad=False)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor( (1.0), requires_grad = False)
        self.d_max = torch.tensor( (0.0), requires_grad = False)

    def forward(self, x):

        device = self.weight.device

        batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        self.running_var = self.running_var.to(device)
        self.running_mean = self.running_mean.to(device)
        self.momentum = self.momentum.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)


        if self.training:

            r = torch.clamp(batch_ch_std / self.running_var.reshape((1, self.running_var.shape[0], 1, 1)), 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_mean.reshape((1, self.running_mean.shape[0], 1, 1))) / self.running_var.reshape((1, self.running_var.shape[0], 1, 1)), -self.d_max, self.d_max).to(device).data.to(device)

            x = ((x - batch_ch_mean) * r )/ batch_ch_std + d
            x = self.weight.reshape((1, self.weight.shape[0], 1, 1)) * x + self.bias.reshape((1, self.bias.shape[0], 1, 1))

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]
        else:
            x = (x - self.running_mean.reshape((1, self.running_mean.shape[0], 1, 1))) / self.running_var.reshape((1, self.running_var.shape[0], 1, 1))
            x = self.weight.reshape((1, self.weight.shape[0], 1, 1)) * x + self.bias.reshape((1, self.bias.shape[0], 1, 1))

        self.running_mean = self.running_mean.reshape((1, self.running_mean.shape[0], 1, 1)) + self.momentum * (batch_ch_mean.data.to(device) - self.running_mean.reshape((1, self.running_mean.shape[0], 1, 1)))
        self.running_var = self.running_var.reshape((1, self.running_var.shape[0], 1, 1)) + self.momentum * (batch_ch_std.data.to(device) - self.running_var.reshape((1, self.running_var.shape[0], 1, 1)))

        return x

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True): #use_cuda=True
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda#use_cuda False

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.float32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.float32)

            xx_range = torch.arange(dim_y, dtype=torch.float32)
            yy_range = torch.arange(dim_x, dtype=torch.float32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        return out

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=BatchRenormalization2D, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False, has_coords=False): #norm_layer=nn.BatchNorm2d
        super(ConvBnRelu, self).__init__()
        if (has_coords):
            self.conv = CoordConv2d(in_planes, out_planes, kernel_size=ksize,
                                  stride=stride, padding=pad,
                                  dilation=dilation, groups=groups, bias=has_bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                                  stride=stride, padding=pad,
                                  dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d): #nn.BatchNorm2d
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            #todo output = F.upsample_bilinear(output, )
            # size = context_blocks[0].size()[2:]
            output = F.interpolate(output, size=torch.Size([576, 1024]), #640, 640
                                   mode='bilinear',
                                   align_corners=True)
        else:
            #todo output = F.upsample_bilinear(output, )
            # size = context_blocks[0].size()[2:]
            output = F.interpolate(output, size=torch.Size([576, 1024]),
                                   mode='bilinear',
                                   align_corners=True)
        return output
    
class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)

        # TODO UNIVERSAL SHAPE WxH
        fm_se = F.interpolate(fm_se, size=(fm.size()[2:]),
                        mode='bilinear', align_corners=True)
                    
        # fm_se = F.interpolate(fm_se, size=torch.Size([80, 80]),
        #               mode='bilinear',
        #               align_corners=True)

        output = fm + fm * fm_se
        return output

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(ASPP, self).__init__()
        self.blocks = nn.ModuleList([
            ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                       has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer),
            ConvBnRelu(in_channels, out_channels, 3, 1, 1,
                       has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer, dilation=1),
            ConvBnRelu(in_channels, out_channels, 3, 1, 6,
                       has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer, dilation=6),
            ConvBnRelu(in_channels, out_channels, 3, 1, 12,
                       has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer, dilation=12),
            nn.Sequential(  # image pooling branch
                nn.AdaptiveAvgPool2d(1),
                ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                           has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
            )
        ])
        self.project = ConvBnRelu(out_channels * 5, out_channels, 1, 1, 0,
                                  has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)

    def forward(self, x):
        size = x.shape[2:]
        feats = [blk(x) if i < 4 else F.interpolate(blk(x), size=size, mode='bilinear', align_corners=True)
                 for i, blk in enumerate(self.blocks)]
        x = torch.cat(feats, dim=1)
        return self.project(x)

@register()
class BiSeNetDecoder(nn.Module):
    def __init__(self, out_planes, in_planes,
                 norm_layer=nn.BatchNorm2d, head_configs=None):
        super(BiSeNetDecoder, self).__init__()

        conv_channels = [256, 256] # 128 for small_dfine  [p // 4 for p in in_planes]  # например, [384, 192]
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_planes[1], conv_channels[0], 1, 1, 0, # 1024 -> in_planes
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )
        # self.global_context = ASPP(in_planes[1], conv_channels[1], norm_layer)

        arms = [AttentionRefinement(in_planes[1], conv_channels[0], norm_layer), # 1024 -> in_planes
                AttentionRefinement(in_planes[0], conv_channels[0], norm_layer)] # 512 -> in_planes
        refines = [ConvBnRelu(conv_channels[1], conv_channels[0], 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channels[0], conv_channels[0], 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        # heads = [BiSeNetHead(conv_channels[0], out_planes, 16,
        #                      True, norm_layer),
        #          BiSeNetHead(conv_channels[0], out_planes, 8,
        #                      True, norm_layer),
        #          BiSeNetHead(conv_channels[0] * 2, out_planes, 8,
        #                      False, norm_layer)]

        self.ffm = FeatureFusion(conv_channels[0] * 2, conv_channels[0] * 2,
                                 4, norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        # self.heads = nn.ModuleList(heads)

        # Механизм инициализации сегментационных голов: aux1, aux2, main
        self.heads = nn.ModuleDict()

        for cfg in head_configs:
            output_name = cfg['name']
            out_planes = cfg['out_planes']
            self.heads[output_name] = nn.ModuleDict()
            self.heads[output_name]['aux1'] = BiSeNetHead(
                    in_planes=conv_channels[0],
                    out_planes=out_planes,
                    scale=16,
                    is_aux=True,
                    norm_layer=norm_layer)
            self.heads[output_name]['aux2'] = BiSeNetHead(
                    in_planes=conv_channels[0],
                    out_planes=out_planes,
                    scale=8,
                    is_aux=True,
                    norm_layer=norm_layer)
            self.heads[output_name]['main'] = BiSeNetHead(
                    in_planes=conv_channels[0] * 2,
                    out_planes=out_planes,
                    scale=8,
                    is_aux=False,
                    norm_layer=norm_layer)

            # for head_name, cfg in output_heads.items():
            #     self.heads[output_name][head_name] = BiSeNetHead(
            #         in_planes=cfg.get('in_planes', conv_channels[0]),
            #         out_planes=cfg['out_planes'],
            #         scale=cfg.get('scale', 8),
            #         is_aux=cfg.get('aux', False),
            #         norm_layer=norm_layer)
            
        # self.reduce_layers = nn.ModuleList([
        #     ConvBnRelu(in_planes[1], conv_channels[1], 1, 1, 0,
        #             has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer),
        #     ConvBnRelu(in_planes[0], conv_channels[0], 1, 1, 0,
        #             has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        # ])

    def forward(self, spatial_out, context_blocks, targets=None):
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        
        # global_context = F.interpolate(global_context,
        #                             size=context_blocks[0].size()[2:],
        #                             mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                self.refines)):
            # fm = self.reduce_layers[i](fm)  # добавлено: адаптация
            fm = arm(fm)
            fm += last_fm

            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)

        pred_out.append(concate_fm)

        outputs = {}
        for out_name, heads_dict in self.heads.items():
            for i, (head_name, head) in enumerate(heads_dict.items()):
                if targets is not None:
                    # получаем индекс тензора под необходимую голову
                    # feature_map_idx = next((i for i, target in enumerate(targets) if target.get('source') == out_name), -1)
                    # feature_map = pred_out[i][feature_map_idx].unsqueeze(0)  # добавляем batch dim

                    feature_map_indices = [j for j, target in enumerate(targets) if target.get('source') == out_name]

                    if feature_map_indices:
                        # собрать feature maps по найденным индексам
                        feature_maps = [pred_out[i][idx].unsqueeze(0) for idx in feature_map_indices]  # добавляем batch dim
                        feature_map = torch.cat(feature_maps, dim=0)  # объединяем в батч

                else:
                    # Если targets нет, используем весь выход (eval режим)
                    feature_map = pred_out[i] 

                key = f"pred_sgm_head_{out_name}_{head_name}"
                outputs[key] = head(feature_map)

        return outputs


        # h_0 = self.heads[0](pred_out[0])
        # h_1 = self.heads[1](pred_out[1])
        # h_2 = self.heads[-1](pred_out[2])
        
        # return {"pred_sgm_head_0":h_0, "pred_sgm_head_1":h_1, "pred_sgm_head_2":h_2}