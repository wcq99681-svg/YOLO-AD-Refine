# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
import torch.nn.functional as F
from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
import traceback
__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    """Head for integrating YOLOv8 detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
###############################################

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """为给定的kernel自动计算padding以保持'same'形状。k, p, d是简写。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际kernel大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动padding
    return p
class Conv_GN(nn.Module):
    """标准卷积，参数包括(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)。"""
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化Conv层，包括激活函数。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2) # 根据你的定义，固定16组
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """对输入张量应用卷积、批归一化和激活。"""
        return self.act(self.gn(self.conv(x)))

    def forward_fuse(self, x):
        """执行2D数据的转置卷积。(注：此函数名可能来自YOLO特定用法，此处保持原样)"""
        return self.act(self.conv(x))

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()
        
    def init_weights(self):
        # self.la_conv1.weight.normal_(std=0.001)
        # self.la_conv2.weight.normal_(std=0.001)
        # self.la_conv2.bias.data.zero_()
        # self.reduction_conv.conv.weight.normal_(std=0.01)
        
        torch.nn.init.normal_(self.la_conv1.weight.data, mean=0, std=0.001)
        torch.nn.init.normal_(self.la_conv2.weight.data, mean=0, std=0.001)
        torch.nn.init.zeros_(self.la_conv2.bias.data)
        torch.nn.init.normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs.
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat

class CoordAtt(nn.Module):
    """Coordinate Attention Block."""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() # Or nn.SiLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class EfficientRepBlock(nn.Module):
    """Simplified Residual Block inspired by RepVGG."""
    def __init__(self, c1, c2, k=3, s=1, p=1, act=nn.ReLU):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, p=p, act=act()) # Use standard Conv block
        self.conv2 = Conv(c2, c2, k, s, p=p, act=act()) # Use standard Conv block
        self.shortcut = nn.Identity() if c1 == c2 and s == 1 else Conv(c1, c2, 1, s, act=False) # Projection shortcut

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + res
class CrossTaskInteraction(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.cls_to_reg = nn.Conv2d(channels, channels, 1)
        self.reg_to_cls = nn.Conv2d(channels, channels, 1)
        self.cls_gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
        self.reg_gate = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, cls_feat, reg_feat):
        cls_to_reg = self.cls_to_reg(cls_feat)
        reg_to_cls = self.reg_to_cls(reg_feat)
        
        cls_gate = self.cls_gate(torch.cat([cls_feat, reg_to_cls], dim=1))
        reg_gate = self.reg_gate(torch.cat([reg_feat, cls_to_reg], dim=1))
        
        cls_enhanced = cls_feat + reg_to_cls * cls_gate
        reg_enhanced = reg_feat + cls_to_reg * reg_gate
        
        return cls_enhanced, reg_enhanced
    #Dynamic Gated Task-Head DGTH
from mmcv.ops import ModulatedDeformConv2d  # 移除依赖
from mmcv.cnn import build_norm_layer
class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class AYHead_org(nn.Module):
    """
    Advanced YOLO Head (AYHead) with Coordinate Attention and EfficientRepBlock.
    Based on the structure of Detect_DGTH but with innovations.
    """
    # Class variables (same as Detect_DGTH for compatibility)
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    format = None  # export format

    def __init__(self, nc=80, ch=()):
        """
        Initializes the AYHead.
        Args:
            nc (int): Number of classes.
            hidc (int): Hidden channels for the shared convolution backbone part.
                       The task-specific paths will often use hidc // 2.
            ch (list): List or tuple of input channels for each feature level.
        """
        super().__init__()
        self.nc = nc # number of classes
        self.nl = len(ch) # number of detection layers
        self.reg_max = 16 # DFL channels (predictor channels)
        # number of outputs per anchor for regression (4 coords * reg_max) + classification (nc)
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl) # strides computed during build
        self.ch = ch
        hidc = max(ch) if ch else 512
        # Intermediate channels for task-specific paths
        task_ch = hidc // 2

        self.stems = nn.ModuleList()  # 只添加这个列表，其他模块共享
        for i in range(self.nl):
            self.stems.append(Conv_GN(self.ch[i], hidc, 1))  # 1x1 conv 适配通道（假设 Conv_GN 支持 k=1）
        # Shared convolutions (can be adapted, using Conv_GN like original)
        self.share_conv = nn.Sequential(
            Conv_GN(hidc, task_ch, 3),
            Conv_GN(task_ch, task_ch, 3)
            # Note: Original Detect_DGTH concatenated features from these,
            #       adjust if that behaviour is strictly needed. Here using sequential.
        )

        # Task Decomposition (assuming same module interface)
        # The number of experts and expert channels might need tuning.
        self.cls_decomp = TaskDecomposition(feat_channels=task_ch, stacked_convs=1, la_down_rate=16)
        self.reg_decomp = TaskDecomposition(feat_channels=task_ch, stacked_convs=1, la_down_rate=16)

        # --- Innovations ---
        # Replace ResidualGhostModule with EfficientRepBlock for classification path
        self.rep_block_cls = EfficientRepBlock(task_ch, task_ch)
        self.coord_attention_reg = CoordAtt(task_ch, task_ch)


        self.cross_task = CrossTaskInteraction(task_ch)

        # Alignment Module (Keeping the original Dynamic Deformable Conv setup)
        # Convolution to predict offsets and masks for DCNv2
        # Input channels = hidc (from concatenated share_conv outputs in original)
        # Adjust input channels if share_conv structure changed. Assuming it's now task_ch *after* share_conv
        self.spatial_conv_offset = nn.Conv2d(task_ch, 3 * 3 * 3, 3, padding=1) # Input is task_ch now
        self.offset_dim = 2 * 3 * 3 # 2 coords * kernel points
        self.DyDCNV2 = DyDCNv2(task_ch, task_ch) # Input channels task_ch

        # Classification Confidence / Foreground Probability (Keeping the original)
        # Input channels = task_ch (output of share_conv)
        self.cls_prob_conv = nn.Sequential(
            nn.Conv2d(task_ch, task_ch // 2, 1), # Reduced channels
            nn.ReLU(),
            nn.Conv2d(task_ch // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Output convolutions
        self.cv2 = nn.Conv2d(task_ch, 4 * self.reg_max, 1) # Regression output
        self.cv3 = nn.Conv2d(task_ch, self.nc, 1)          # Classification output

        # Learnable scaling factors for regression outputs
        self.scale = nn.ModuleList([Scale(1.0) for _ in range(self.nl)])

        # Distribution Focal Loss Decode Layer
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # Initialize biases
        self.initialize_biases() # Automatically call bias init
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the AYHead.
        Args:
            x (list): List of feature maps from the neck at different scales.
                      Each element is a tensor of shape (B, C_in, H, W).
        Returns:
            During training: List of tensors [(B, C_out, H, W), ...] for each scale.
            During inference: Single tensor (B, N, 4+nc) or tuple (outputs, features).
                              N = total number of anchors across all scales.
        """
        outputs = []
        for i in range(self.nl):
            # Feature extraction using shared convolutions
            adapted_x = self.stems[i](x[i])  # 将 x[i] (ch[i]) 转换为 hidc 通道
            feat = self.share_conv(adapted_x) 

            # Task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat) # (B, task_ch, H, W)
            reg_feat = self.reg_decomp(feat, avg_feat) # (B, task_ch, H, W)

            # Cross task interaction
            cls_feat, reg_feat = self.cross_task(cls_feat, reg_feat)

            # Enhanced feature processing (Innovations applied here)
            cls_feat = self.rep_block_cls(cls_feat)       # Use EfficientRepBlock
            reg_feat = self.coord_attention_reg(reg_feat) # Use CoordAtt

            # Alignment (using Dynamic Deformable Conv on Regression Features)
            offset_and_mask = self.spatial_conv_offset(feat) # Predict from shared features
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask) # Apply DCNv2

            # Classification confidence/foreground prediction
            cls_prob = self.cls_prob_conv(feat) # Predict from shared features

            # Generate final outputs for this level
            # Apply learnable scale to regression predictions before concatenation
            reg_output = self.scale[i](self.cv2(reg_feat)) # (B, 4*reg_max, H, W)
            # Modulate classification output with confidence probability
            cls_output = self.cv3(cls_feat * cls_prob)     # (B, nc, H, W)

            # Concatenate regression and classification outputs
            level_output = torch.cat((reg_output, cls_output), 1) # (B, no, H, W)
            outputs.append(level_output)

        # --- Inference Path (Identical to original Detect_DGTH) ---
        if self.training:
            return outputs # Return list of features per level

        # Inference post-processing
        shape = outputs[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2) # (B, no, N)

        # Recompute anchors/strides if dynamic shape or first inference
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(outputs, self.stride, 0.5))
            self.shape = shape
            self.anchors = self.anchors.to(x_cat.device)
            self.strides = self.strides.to(x_cat.device)


        # Split into box and class predictions
        # Compatibility for TF exports (avoiding FlexSplitV)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # Decode bounding boxes
        dbox = self.decode_bboxes(box) # Decodes to xywh format

        # TFLite specific normalization (optional, based on original)
        if self.export and self.format in ('tflite', 'edgetpu'):
            # Precompute normalization factor for numerical stability
            img_h, img_w = shape[2], shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            # Ensure self.stride[0] is not zero and strides are valid
            norm = self.strides / (self.stride[0] * img_size) if self.stride[0] > 0 else self.strides / img_size
            # Re-decode using normalized values if needed for specific export formats
            # Note: The decode_bboxes already multiplies by stride, check if double normalization needed
            # dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
            # The standard decode_bboxes might be sufficient if strides are handled correctly.


        # Apply sigmoid to classification scores and concatenate
        y = torch.cat((dbox, cls.sigmoid()), 1) # (B, 4+nc, N)

        # Return format depends on export flag
        return y if self.export else (y, outputs)

    def initialize_biases(self):
        """Initialize biases for the final convolution layers."""
        # **添加默认 stride，如果 self.stride 全0（修复跳过问题）**  # NEW
        if (self.stride == 0).all():  # 如果 stride 未设置
            default_strides = [8, 16, 32]  # 标准 YOLO P3/P4/P5 strides（根据您的 nl=3 调整）
            self.stride = torch.tensor(default_strides[:self.nl], dtype=torch.float32)  # 设置默认
            print("Warning: Setting default strides in AYHead: ", self.stride)  # 调试打印

        for i, s in enumerate(self.stride):
            if s == 0: 
                continue  # 保持，但现在不会跳过

            # Bias initialization for regression layers (cv2)
            # **改进：使用 YOLO 标准公式初始化（替换您的 fill_(1.0)）**  # MODIFIED
            b = self.cv2.bias.data  # (4*reg_max,)
            b.fill_(math.log(5 / self.nc / (640 / s) ** 2))  # 标准 prior（假设 img=640）

            # Bias initialization for classification layers (cv3)
            # **改进：使用完整 YOLOv8 公式（替换您的 simplified prior）**  # MODIFIED
            prior = math.log(5 / self.nc / (640 / s) ** 2)  # 对象密度 prior
            self.cv3.bias.data.fill_(prior)

    # Keep bias_init method for compatibility if called explicitly elsewhere
    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        self.initialize_biases() # Call the internal initialization


    def decode_bboxes(self, bboxes):
        """Decode bounding boxes from Distance Format (ltrb) to xywh."""
        # bboxes shape: (B, 4 * reg_max, N)
        # anchors shape: (N, 2)
        # strides shape: (N, 1)
        # self.dfl processes bboxes from (B, 4*reg_max, N) -> (B, 4, N)
        decoded_dist = self.dfl(bboxes) # Process distributions
        # Use dist2bbox to convert distances to boxes relative to anchors
        # The result needs to be scaled by the stride
        # self.anchors and self.strides need to be on the correct device
        self.anchors = self.anchors.to(bboxes.device)
        self.strides = self.strides.to(bboxes.device)
        # Pass N dimension implicitly, operate on dimension 1 (channel dim)
        return dist2bbox(decoded_dist, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides.unsqueeze(0)


class ResidualBlockGN(nn.Module):
    """
    使用 Conv_GN 的标准残差块。
    (原 EfficientRepBlock 依赖未定义的'Conv'且未使用重参数化)
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        # 保持与 Conv_GN 一致的激活函数 (SiLU)
        self.conv1 = Conv_GN(c1, c2, k, s, p=p, act=act)
        self.conv2 = Conv_GN(c2, c2, k, s, p=p, act=act)
        self.shortcut = nn.Identity() if c1 == c2 and s == 1 else Conv_GN(c1, c2, 1, s, act=False) # 投影快捷连接

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + res

class AYHead1(nn.Module):
    """
    优化的 AYHead (Advanced YOLO Head)
    """
    dynamic = False 
    export = False 
    shape = None
    anchors = torch.empty(0) 
    strides = torch.empty(0) 
    format = None 

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc # 类别数
        self.nl = len(ch) # 检测层数
        self.reg_max = 16 # DFL 通道数
        self.no = nc + self.reg_max * 4 # 每层总输出通道
        self.stride = torch.zeros(self.nl) # 步长 (将在 build 时填充)
        self.ch = ch
        hidc = max(ch) if ch else 512 # 共享卷积的隐藏通道
        task_ch = hidc // 2 # 任务特定分支的通道

        self.stems = nn.ModuleList() 
        for i in range(self.nl):
            self.stems.append(Conv_GN(self.ch[i], hidc, 1)) 

        self.share_conv = nn.Sequential(
            Conv_GN(hidc, task_ch, 3),
            Conv_GN(task_ch, task_ch, 3)
        )

        # 任务分解 (使用你简化的 stacked_convs=1, 作为动态1x1卷积)
        self.cls_decomp = TaskDecomposition(feat_channels=task_ch, stacked_convs=1, la_down_rate=16)
        self.reg_decomp = TaskDecomposition(feat_channels=task_ch, stacked_convs=1, la_down_rate=16)

       
        # 1. 使用 ResidualBlockGN 替换 'EfficientRepBlock'
        self.rep_block_cls = ResidualBlockGN(task_ch, task_ch)
        
        # 2. 回归分支的 CoordAttention 
        self.coord_attention_reg = CoordAtt(task_ch, task_ch)

        # 任务交互 
        self.cross_task = CrossTaskInteraction(task_ch)

        # 对齐模块 
        self.spatial_conv_offset = nn.Conv2d(task_ch, 3 * 3 * 3, 3, padding=1) # 预测 offset 和 mask
        self.offset_dim = 2 * 3 * 3 # 18 (x, y) * 9
        self.DyDCNV2 = DyDCNv2(task_ch, task_ch) 

        # 置信度/前景预测
        self.cls_prob_conv = nn.Sequential(
            nn.Conv2d(task_ch, task_ch // 2, 1), 
            nn.ReLU(),
            nn.Conv2d(task_ch // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 输出卷积 
        self.cv2 = nn.Conv2d(task_ch, 4 * self.reg_max, 1) # 回归
        self.cv3 = nn.Conv2d(task_ch, self.nc, 1)          # 分类

        # 可学习的缩放因子 
        self.scale = nn.ModuleList([Scale(1.0) for _ in range(self.nl)])

        # DFL 解码层 
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 初始化偏置
        self.initialize_biases() 

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # --- 优化的 FORWARD ---
    def forward(self, x):
        """
        AYHead 的前向传播 (已优化逻辑顺序)
        """
        outputs = []
        for i in range(self.nl):
            # 1. Stem: 适配通道
            adapted_x = self.stems[i](x[i])
            # 2. Share Conv: 提取共享特征
            feat = self.share_conv(adapted_x) # (B, task_ch, H, W)

            # 3. Task Decomposition: 初步任务解耦
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat) # (B, task_ch, H, W)
            reg_feat = self.reg_decomp(feat, avg_feat) # (B, task_ch, H, W)

            # 4. Cross Task Interaction: 任务间交互
            cls_feat, reg_feat = self.cross_task(cls_feat, reg_feat)

            # --- 优化的特征处理流程 ---
            
            # 5a. 分类分支增强 (ResidualBlockGN)
            cls_feat_enhanced = self.rep_block_cls(cls_feat)
            
            # 5b. 回归分支: 先对齐 (DyDCNV2)，后增强 (CoordAtt)
            
            # 预测 DCN 的 offset 和 mask (从共享特征 feat 中预测)
            offset_and_mask = self.spatial_conv_offset(feat) 
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            
            # (优化点) 先用 DCNv2 对齐回归特征
            reg_feat_aligned = self.DyDCNV2(reg_feat, offset, mask)
            
            # (优化点) 再用 CoordAtt 增强已对齐的特征
            reg_feat_enhanced = self.coord_attention_reg(reg_feat_aligned)

            # 6. 预测前景概率 (从共享特征 feat 中预测)
            cls_prob = self.cls_prob_conv(feat) 

            # 7. 生成最终输出
            # (使用增强和对齐后的回归特征)
            reg_output = self.scale[i](self.cv2(reg_feat_enhanced)) 
            # (使用增强后的分类特征 * 前景概率)
            cls_output = self.cv3(cls_feat_enhanced * cls_prob)     

            # 8. 拼接
            level_output = torch.cat((reg_output, cls_output), 1) # (B, no, H, W)
            outputs.append(level_output)

        # --- 推理路径 (与原版相同) ---
        if self.training:
            return outputs 

        shape = outputs[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2) # (B, no, N)

        if self.dynamic or self.shape != shape:
            # 确保在推理时调用 make_anchors
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(outputs, self.stride, 0.5))
            self.shape = shape
            self.anchors = self.anchors.to(x_cat.device)
            self.strides = self.strides.to(x_cat.device)

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 解码 BBox
        dbox = self.decode_bboxes(box) # 解码为 xywh 格式
        
        # ... (TFLite specific normalization, 保持不变)

        y = torch.cat((dbox, cls.sigmoid()), 1) # (B, 4+nc, N)

        return y if self.export else (y, outputs)

    def initialize_biases(self):
        """初始化偏置 (使用YOLOv8的推荐公式)"""
        # 检查并设置默认 stride (修复 stride 为0的问题)
        if (self.stride == 0).all(): 
            default_strides = [8, 16, 32, 64, 128] # P3/P4/P5/P6/P7
            self.stride = torch.tensor(default_strides[:self.nl], dtype=torch.float32) 
            print(f"Warning: AYHead 使用默认步长: {self.stride}")

        for i, s in enumerate(self.stride):
            if s == 0: 
                print(f"Warning: Stride at index {i} is 0, skipping bias init.")
                continue 

            # 回归层偏置 (cv2)
            b = self.cv2.bias.data.view(4, -1) # (4, reg_max)
            b.data.fill_(1.0) # 原始
            # 可选的YOLOv8 DFL初始化 (注释掉，因为你的原始实现是 fill 1.0)
            # b.data[:, :self.reg_max // 2] = 1.0
            # b.data[:, self.reg_max // 2:] = -1.0
            self.cv2.bias.data = b.view(-1)

            # 分类层偏置 (cv3) - 使用标准 prior
            prior_prob = 0.01
            b = -math.log((1 - prior_prob) / prior_prob)
            self.cv3.bias.data.fill_(b)

    def bias_init(self):
        """兼容性方法"""
        self.initialize_biases() 

    def decode_bboxes(self, bboxes):
            """解码 Bounding Boxes (已修正维度错误)"""
            # bboxes: (B, 4 * reg_max, N)
            # self.anchors: (2, N)
            # self.strides: (1, N)
            
            self.anchors = self.anchors.to(bboxes.device)
            self.strides = self.strides.to(bboxes.device)
            
            # bboxes (B, 4*reg_max, N) -> (B, 4, N)
            decoded_dist = self.dfl(bboxes) 
            
            # self.anchors.unsqueeze(0) 形状变为 (1, 2, N)
            # self.strides.unsqueeze(0) 形状变为 (1, 1, N)
            
            # 移除 .transpose(1, 2)
            return dist2bbox(decoded_dist, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides.unsqueeze(0)

######################################
from ultralytics.nn.modules.attention import EffectiveSEModule, LSKBlock, deformable_LKA
import warnings
from torch.nn.init import constant_, normal_
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv_GN(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        effective_groups = min(num_groups, c2) if c2 > 0 else num_groups
        if c2 > 0 and c2 % effective_groups != 0:
             factors = [i for i in range(1, effective_groups + 1) if c2 % i == 0]
             effective_groups = max(factors) if factors else 1
        self.gn = nn.GroupNorm(effective_groups, c2) if c2 > 0 else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

# --- Modified TaskDecomposition using EffectiveSEModule ---
class TaskDecomposition_ESE(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8): # la_down_rate might be unused now
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs

        # Use EffectiveSEModule for attention
        self.attention = EffectiveSEModule(self.in_channels)

        # Reduction convolution remains the same
        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()

    def init_weights(self):
        # Initialize only the reduction convolution
        normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)
        # EffectiveSEModule typically has its own initialization or uses default Conv init

    def forward(self, feat, avg_feat=None): # avg_feat is ignored
        b, c, h, w = feat.shape # c == self.in_channels

        # Apply attention directly to the input feature map
        attention_weights = self.attention(feat) # Gets weighted feature map (B, C_in, H, W)

        # Apply reduction convolution to the weighted feature map
        feat_reduced = self.reduction_conv(attention_weights) # (B, feat_channels, H, W)

        return feat_reduced

class CrossTaskInteraction(nn.Module): # Keep original simple version
    def __init__(self, channels):
        super().__init__()
        self.cls_to_reg = nn.Conv2d(channels, channels, 1)
        self.reg_to_cls = nn.Conv2d(channels, channels, 1)
        self.cls_gate = nn.Sequential(nn.Conv2d(channels*2, channels, 1), nn.Sigmoid())
        self.reg_gate = nn.Sequential(nn.Conv2d(channels*2, channels, 1), nn.Sigmoid())
    def forward(self, cls_feat, reg_feat):
        cls_to_reg = self.cls_to_reg(cls_feat)
        reg_to_cls = self.reg_to_cls(reg_feat)
        cls_gate = self.cls_gate(torch.cat([cls_feat, reg_to_cls], dim=1))
        reg_gate = self.reg_gate(torch.cat([reg_feat, cls_to_reg], dim=1))
        cls_enhanced = cls_feat + reg_to_cls * cls_gate
        reg_enhanced = reg_feat + cls_to_reg * reg_gate
        return cls_enhanced, reg_enhanced
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
# --- AYHead using block.py's DCNv2 and new attention ---
class AYHead_Attention_Optimized(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    format = None

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.ch = ch
        hidc = max(ch) if ch else 512
        task_ch = hidc // 2

        self.stems = nn.ModuleList(Conv_GN(c, hidc, 1) for c in self.ch)

        self.share_conv = nn.Sequential(
            Conv_GN(hidc, task_ch, 3),
            Conv_GN(task_ch, task_ch, 3)
        )

        # Use modified TaskDecomposition
        self.cls_decomp = TaskDecomposition_ESE(feat_channels=task_ch, stacked_convs=1)
        self.reg_decomp = TaskDecomposition_ESE(feat_channels=task_ch, stacked_convs=1)

        # --- Innovations ---
        self.cls_enhance = LSKBlock(task_ch) # Classification branch enhancement
        # self.cls_enhance = nn.Identity()
        # self.reg_enhance = deformable_LKA(task_ch) # Regression branch enhancement
        self.reg_enhance = nn.Identity()
        self.cross_task = CrossTaskInteraction(task_ch) # Keep simple interaction for now

        # Alignment Module (using block.py DCNv2 - assumes internal offset/mask)
        # Remove self.spatial_conv_offset as DCNv2 from block.py handles it
        # try:
        #     # kernel_size=3, padding=1 based on DyDCNv2
        #     self.DyDCN = DCNv2(task_ch, task_ch, kernel_size=3, stride=1, padding=1)
        #     # self.DyDCN = nn.Identity()
        #     print("Using DCNv2 from block.py for alignment.")
        # except NameError:
        #      print("Warning: DCNv2 not found in block.py. Alignment will be skipped.")
        self.DyDCN = nn.Identity() # Fallback


        # Classification Confidence / Foreground Probability (Keep original)
        self.cls_prob_conv = nn.Sequential(
            nn.Conv2d(task_ch, task_ch // 2, 1),
            nn.ReLU(),
            nn.Conv2d(task_ch // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Output convolutions
        self.cv2 = nn.Conv2d(task_ch, 4 * self.reg_max, 1) # Regression
        self.cv3 = nn.Conv2d(task_ch, self.nc, 1)          # Classification

        # Learnable scaling factors
        self.scale = nn.ModuleList(Scale(1.0) for _ in range(self.nl))

        # DFL Decode Layer
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # Initialize biases
        self.initialize_biases()

    # def forward(self, x):
    #     outputs = []
    #     for i in range(self.nl):
    #         # 1. Stem
    #         adapted_x = self.stems[i](x[i])
    #         # 2. Share Conv
    #         feat = self.share_conv(adapted_x) # Shared features

    #         # 3. Task Decomposition (using ESE)
    #         cls_feat = self.cls_decomp(feat)
    #         reg_feat = self.reg_decomp(feat)

    #         # 4. Cross Task Interaction
    #         cls_feat, reg_feat = self.cross_task(cls_feat, reg_feat)

    #         # --- Feature Processing ---
    #         # 5a. Classification Branch Enhancement (LSKBlock)
    #         cls_feat_enhanced = self.cls_enhance(cls_feat)

    #         # 5b. Regression Branch: Align (block.py DCNv2) -> Enhance (deformable_LKA)
    #         # Alignment using block.py's DCNv2 (assumes internal offset/mask handling)
    #         reg_feat_aligned = self.DyDCN(reg_feat)
    #         # Enhancement using deformable_LKA
    #         reg_feat_enhanced = self.reg_enhance(reg_feat_aligned)

    #         # 6. Predict Foreground Probability
    #         cls_prob = self.cls_prob_conv(feat) # Predict from shared features

    #         # 7. Generate Final Outputs
    #         reg_output = self.scale[i](self.cv2(reg_feat_enhanced))
    #         cls_output = self.cv3(cls_feat_enhanced * cls_prob)

    #         # 8. Concatenate
    #         level_output = torch.cat((reg_output, cls_output), 1)
    #         outputs.append(level_output)

    #     # --- Inference Path ---
    #     if self.training:
    #         return outputs

    #     shape = outputs[0].shape
    #     x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2)

    #     if self.dynamic or self.shape != shape:
    #          if (self.stride == 0).all():
    #              self.stride = self._get_default_strides(outputs)

    #          self.anchors, self.strides = (y.transpose(0, 1) for y in make_anchors(outputs, self.stride, 0.5))
    #          self.shape = shape
    #          self.anchors = self.anchors.to(x_cat.device)
    #          self.strides = self.strides.to(x_cat.device)


    #     if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
    #         box = x_cat[:, :self.reg_max * 4]
    #         cls = x_cat[:, self.reg_max * 4:]
    #     else:
    #         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

    #     dbox = self.decode_bboxes(box)
    #     y = torch.cat((dbox, cls.sigmoid()), 1)

    #     return y if self.export else (y, outputs)
    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            print(f"Processing level {i}...")

            # 1. Stem
            adapted_x = self.stems[i](x[i])
            print(f"Stem output shape: {adapted_x.shape}")

            # 2. Share Conv
            feat = self.share_conv(adapted_x)
            print(f"Share Conv output shape: {feat.shape}")

            # 3. Task Decomposition (using ESE)
            cls_feat = self.cls_decomp(feat)
            reg_feat = self.reg_decomp(feat)

            print(f"Task Decomposition: cls_feat shape {cls_feat.shape}, reg_feat shape {reg_feat.shape}")

            # 4. Cross Task Interaction
            cls_feat, reg_feat = self.cross_task(cls_feat, reg_feat)
            print(f"Cross Task Interaction: cls_feat shape {cls_feat.shape}, reg_feat shape {reg_feat.shape}")

            # --- Feature Processing ---
            # 5a. Classification Branch Enhancement (LSKBlock)
            cls_feat_enhanced = self.cls_enhance(cls_feat)
            print(f"Enhanced classification feature shape: {cls_feat_enhanced.shape}")

            # 5b. Regression Branch: Align (block.py DCNv2) -> Enhance (deformable_LKA)
            print(f"reg_feat shape before DCNv2: {reg_feat.shape}")
            reg_feat_aligned = self.DyDCN(reg_feat)
            print(f"Aligned regression feature shape: {reg_feat_aligned.shape}")  # 确认这一行是否有输出
            
            reg_feat_enhanced = self.reg_enhance(reg_feat_aligned)
            print(f"Enhanced regression feature shape: {reg_feat_enhanced.shape}")

            # 6. Predict Foreground Probability
            cls_prob = self.cls_prob_conv(feat)
            print(f"cls_prob shape: {cls_prob.shape}")

            # 7. Generate Final Outputs
            reg_output = self.scale[i](self.cv2(reg_feat_enhanced))
            cls_output = self.cv3(cls_feat_enhanced * cls_prob)
            print(f"reg_output shape: {reg_output.shape}, cls_output shape: {cls_output.shape}")

            # 8. Concatenate
            level_output = torch.cat((reg_output, cls_output), 1)
            outputs.append(level_output)

        # --- Inference Path ---
        if self.training:
            return outputs

        shape = outputs[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2)
        print(f"x_cat shape: {x_cat.shape}")

        if self.dynamic or self.shape != shape:
            if (self.stride == 0).all():
                self.stride = self._get_default_strides(outputs)

            self.anchors, self.strides = (y.transpose(0, 1) for y in make_anchors(outputs, self.stride, 0.5))
            self.shape = shape
            self.anchors = self.anchors.to(x_cat.device)
            self.strides = self.strides.to(x_cat.device)

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(box)
        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y if self.export else (y, outputs)

    # --- Other methods (initialize_biases, bias_init, decode_bboxes, _get_default_strides) remain the same as AYHead_DCNv3 ---
    def _get_default_strides(self, feature_maps):
        try:
             h, w = self.shape[2:]
             if h == 0 or w == 0: raise ValueError("Shape not properly initialized")
        except (AttributeError, TypeError, ValueError):
             warnings.warn("Shape not available for stride calculation, estimating from first feature map.")
             h, w = feature_maps[0].shape[2:]
        base_stride = 8.0
        calculated_strides = [base_stride * (2**i) for i in range(self.nl)]
        print(f"Warning: {self.__class__.__name__} using estimated strides: {calculated_strides}")
        return torch.tensor(calculated_strides, dtype=torch.float32)

    def initialize_biases(self):
        if hasattr(self, 'stride') and (self.stride == 0).all():
             warnings.warn("Strides not available during bias initialization. Will attempt later or use defaults.")
             temp_strides = [8.0 * (2**i) for i in range(self.nl)]
             current_strides = torch.tensor(temp_strides, dtype=torch.float32)
        elif not hasattr(self, 'stride'):
             warnings.warn("Stride attribute not found during bias initialization.")
             return
        else:
            current_strides = self.stride

        img_size_default = 640
        for i, s in enumerate(current_strides):
            if s == 0:
                print(f"Warning: Stride at index {i} is 0, skipping bias init.")
                continue
            b_reg = self.cv2.bias.data.view(4, -1)
            b_reg.data.fill_(1.0)
            self.cv2.bias.data = b_reg.view(-1)
            prior_prob = 0.01
            b_cls = -math.log((1 - prior_prob) / prior_prob)
            self.cv3.bias.data.fill_(b_cls)

    def bias_init(self):
        self.initialize_biases()

    def decode_bboxes(self, bboxes):
        if self.anchors.numel() == 0 or self.strides.numel() == 0:
            if not self.training and self.shape is not None:
                warnings.warn("Anchors/strides not initialized in decode_bboxes, attempting initialization.")
                print("Error: Cannot decode bboxes before anchors/strides are initialized.")
                B, _, N = bboxes.shape
                return torch.zeros(B, 4, N, device=bboxes.device, dtype=bboxes.dtype)
            elif self.training:
                 raise RuntimeError("decode_bboxes called during training without anchors/strides.")
            else:
                 raise RuntimeError("decode_bboxes called before anchors/strides initialization.")

        self.anchors = self.anchors.to(bboxes.device)
        self.strides = self.strides.to(bboxes.device)
        decoded_dist = self.dfl(bboxes)
        strides_for_bbox = self.strides.unsqueeze(0)
        return dist2bbox(decoded_dist, self.anchors.unsqueeze(0), xywh=True, dim=1) * strides_for_bbox
AYHead=AYHead1
# %%
