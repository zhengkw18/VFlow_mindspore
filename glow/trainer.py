import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from .config import JsonConfig
from .models import Glow, GlowLoss
from mindspore.common.tensor import Tensor
import mindspore.context as context
import mindspore.nn as nn
from .train_one_step_cell import TrainOneStepCell
from mindspore.common.initializer import initializer
from mindspore.nn.optim import Adamax, Momentum, Adam
from mindspore.communication.management import init, NCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, _InternalCallbackParam, RunContext, CheckpointConfig
from mindspore.common import dtype as mstype

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", enable_mem_reuse=True, enable_dynamic_memory=False)
# init('nccl')


class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, batch_size):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, y_onehot):
        z_final, nll, augmented_nll, y_logits = self._backbone(x, y_onehot)
        return self._loss_fn(nll, augmented_nll, y_logits, y_onehot)


class Trainer(object):
    def __init__(self, graph, graph_decoder, optim, lrschedule,
                 devices, dataset, dataset_root, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        # model relative
        self.graph = graph
        self.graph_decoder = graph_decoder
        self.optim = optim
        self.weight_y = hparams.Train.weight_y
        # number of training batches
        if hparams.Glow.reverse is True:
            self.batch_size = 1
        else:
            self.batch_size = hparams.Train.batch_size
        self.pt_data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.n_epoches = (hparams.Train.num_batches + len(self.pt_data_loader) - 1)
        self.n_epoches = self.n_epoches // len(self.pt_data_loader)
        self.hparams = hparams
        self.dataset_root = dataset_root
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule

    def set_checkpoint(self, train_net):
        cb_params = _InternalCallbackParam()
        run_context = RunContext(cb_params)
        config_ck = CheckpointConfig(save_checkpoint_steps=5000, keep_checkpoint_max=20)
        ckpt_dir = "./checkpoints/"
        if not os.path.exists(ckpt_dir + "/ms_ckpt"):
            os.makedirs(ckpt_dir + "/ms_ckpt")
        dir_name = ckpt_dir + "/ms_ckpt/"
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_glow", directory=dir_name, config=config_ck)
        cb_params.epoch_num = self.n_epoches  # required for ModelCheckpoint
        cb_params.cur_step_num = 0
        cb_params.batch_num = self.batch_size
        cb_params.train_network = train_net
        cb_params.step_num = cb_params.epoch_num * self.batch_size
        ckpoint_cb.begin(run_context)
        return run_context, cb_params, ckpoint_cb

    def set_distributed_train(self):
        context.set_auto_parallel_context(parallel_mode="data_parallel", mirror_mean=True, device_num=get_group_size())
        return True

    def save_ckpt(self, run_context, cb_params, ckpoint_cb):
        cb_params.cur_epoch_num = self.global_step + 1
        cb_params.cur_step_num = (self.global_step + 1) * cb_params.batch_num
        ckpoint_cb.step_end(run_context)
        return

    def get_dataloader(self, data_dir):
        if self.hparams.Train.enable_minddata is True:
            import mindspore.dataengine as de
            import mindspore.transforms.c_transforms as vision
            ds = de.CelebADataset(data_dir, decode=True, num_shards=1, shard_id=0)
            crop_size = (self.hparams.Data.center_crop, self.hparams.Data.center_crop)
            resize_size = (self.hparams.Data.resize, self.hparams.Data.resize)
            rescale = 1.0 / 255.0
            shift = 0.0
            ds = ds.repeat(self.n_epoches)
            hwc2chw_op = vision.HWC2CHW()
            center_crop = vision.CenterCrop(crop_size)
            resize_op = vision.Resize(resize_size, vision.Inter.LINEAR)  # Bilinear mode
            rescale_op = vision.Rescale(rescale, shift)
            ds = ds.map(input_column_names="image", operation=center_crop)
            ds = ds.map(input_column_names="image", operation=resize_op)
            ds = ds.map(input_column_names="image", operation=rescale_op)
            ds = ds.map(input_column_names="image", operation=hwc2chw_op)
            ds = ds.batch(self.batch_size)
            return ds.create_dict_iterator()
        else:
            return self.pt_data_loader

    def multistep(self, total_steps, dtype=mstype.float32):
        steps = []
        for number in range(1, total_steps + 1):
            steps.append(number)
        return Tensor(np.array(steps), dtype)

    def multisteplr(self, init_lr):
        lr = []
        warmup_steps = 4000
        minimum = 1e-4
        for step in range(self.n_epoches * len(self.pt_data_loader) + 1):
            lr_ = init_lr * warmup_steps ** 0.5 * np.minimum((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
            if step > warmup_steps:
                if lr_ < minimum:
                    lr_ = minimum
            lr.append(lr_)
        return Tensor(np.array(lr).astype(np.float32))

    def train(self):
        # model
        glow_net = self.graph
        weight_init = Tensor((np.ones([1]) * (0.5)).astype(np.float32))
        glow_loss = GlowLoss(y_condition=True, y_briterion="multi-classes", weight_y=weight_init)
        net_with_loss = WithLossCell(glow_net, glow_loss, self.batch_size)
        # optimizer
        beta1 = initializer(Tensor(np.array([0.9]).astype(np.float32)), [1])
        beta2 = initializer(Tensor(np.array([0.9999]).astype(np.float32)), [1])
        steps = self.multistep(self.hparams.Train.num_batches)
        lr = self.multisteplr(0.001)
        optimizer = Adamax(filter(lambda x: x.requires_grad, glow_net.get_parameters()), lr, beta1, beta2, steps)
        #optimizer = Momentum(filter(lambda x: x.requires_grad, glow_net.get_parameters()), 0.0001, 0.0)
        # build train net
        train_net = TrainOneStepCell(net_with_loss, optimizer)
        train_net.set_train()
        if self.hparams.Train.enable_distributed is True:
            self.set_distributed_train()
        if self.hparams.Train.enable_checkpoint is True:
            run_context, cb_params, ckpoint_cb = self.set_checkpoint(train_net)

        data_loader = self.get_dataloader(self.dataset_root)
        for epoch in range(self.n_epoches):
            for data in data_loader:
                # get batch data
                if self.hparams.Train.enable_minddata is True:
                    x = Tensor(data["image"] + np.random.normal(0., 1.0 / 256, size=(self.batch_size, 3, 64, 64)).astype(np.float32))
                    y_onehot = Tensor(data["attr"].astype(np.float32))
                else:
                    x = Tensor(data["image"].numpy() + np.random.normal(0., 1.0 / 256, size=(self.batch_size, 3, 64, 64)).astype(np.float32))
                    y_onehot = Tensor(data["attr"].numpy().astype(np.float32))
                loss = train_net(x, y_onehot)
                print("epoch = {0}, iter = {1}, loss = {2}".format(epoch, self.global_step, loss))
                if self.hparams.Train.enable_checkpoint is True:
                    self.save_ckpt(run_context, cb_params, ckpoint_cb)
                self.global_step += 1

    def infer(self):
        image_shape = self.hparams.Glow.image_shape
        glow_decoder = self.graph_decoder
        if self.hparams.Infer.reconstruct is True:
            data_loader = self.get_dataloader(self.dataset_root)
            for data in data_loader:
                # get batch data
                if self.hparams.Train.enable_minddata is True:
                    x = Tensor(data["image"].astype(np.float32))
                    y_onehot = Tensor(data["attr"].astype(np.float32))
                else:
                    x = Tensor(data["image"].numpy())
                    y_onehot = Tensor(data["attr"].numpy().astype(np.float32))
                # we only get one image
                break

            glow_encoder = self.graph
            z, _, _ = glow_encoder(x, y_onehot)
            images = glow_decoder(z, y_onehot)
        else:
            batch_size = self.batch_size
            np_x = np.random.rand(batch_size, image_shape[2], image_shape[0], image_shape[1]).astype(np.float32)
            np_y_onehot = np.zeros((batch_size, 40)).astype(np.float32)
            data_loader = self.get_dataloader(self.dataset_root)
            for data in data_loader:
                if self.hparams.Train.enable_minddata is True:
                    x = Tensor(data["image"])
                    y_onehot = Tensor(data["attr"].astype(np.float32))
                else:
                    x = Tensor(data["image"].numpy())
                    y_onehot = Tensor(data["attr"].numpy().astype(np.float32))
                # we only get one image
                break
            x = Tensor(np_x)
            y_onehot = Tensor(np_y_onehot)
            images = glow_decoder(x, y_onehot)

        images = images.asnumpy()
        images = np.transpose(images[0], (1, 2, 0))
        images = images[:, :, ::-1]
        images = cv2.resize(images, (256, 256))
        images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("out_cv.jpg", images)
