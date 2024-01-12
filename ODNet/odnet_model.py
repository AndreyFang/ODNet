import os
import cv2
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import utils
import torchvision.utils as vutils
from ODNet.criterion import CtoFCriterion
from ODNet.network import ODNet
import os

class ODNetModel():

    def name(self):
        return 'ODNet Model'

    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.MULTI_GPU = False
        if torch.cuda.device_count() > 1:
            self.MULTI_GPU = True
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIABLE_DEVICES"] = "0,1"
            self.device_ids = [0, 1]
        self.lowestloss = 1.0

        # init model, optimizer, scheduler
        self.model = ODNet(args, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)
        if self.MULTI_GPU:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # reloading from checkpoints
        self.start_step = self.load_from_ckpt()

        # define loss function
        self.criterion = CtoFCriterion(args).to(self.device)

    def set_input(self, data):
        self.im1 = Variable(data['im1'].to(self.device))
        self.im2 = Variable(data['im2'].to(self.device))

        self.coord1 = Variable(data['coord1'].to(self.device))
        self.fmatrix = data['F'].cuda()
        self.pose = Variable(data['pose'].to(self.device))
        self.intrinsic1 = data['intrinsic1'].to(self.device)
        self.intrinsic2 = data['intrinsic2'].to(self.device)

        self.im1_ori = data['im1_ori']
        self.im2_ori = data['im2_ori']
        self.batch_size = len(self.im1)
        self.imsize = self.im1.size()[2:]

    def forward(self):
        self.out, self.f11, self.f12, self.f21, self.f22 = self.model.forward(self.im1, self.im2, self.coord1)

    def backward_net(self):
        loss = self.criterion(self.coord1, self.out, self.fmatrix, self.pose, self.imsize)
        self.j_loss, self.eloss_c, self.eloss_f, self.closs_c, self.closs_f, self.std_loss = loss
        # self.j_loss, self.eloss_f, self.closs_f, self.std_loss = loss
        alpha_orth = 1e-10

        # orth_loss_11 = torch.abs(torch.sum(torch.mul(self.c11, self.c12)))
        orth_loss_12 = torch.abs(torch.sum(torch.mul(self.f11, self.f12))) / (
                    self.f11.size()[0] * self.f11.size()[2] * self.f11.size()[3])
        # orth_loss_21 = torch.abs(torch.sum(torch.mul(self.c21, self.c22)))
        orth_loss_22 = torch.abs(torch.sum(torch.mul(self.f21, self.f22))) / (
                    self.f11.size()[0] * self.f11.size()[2] * self.f11.size()[3])

        self.ort_loss = alpha_orth * (orth_loss_12 + orth_loss_22)
        self.j_loss = self.j_loss + self.ort_loss
        self.j_loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward_net()
        self.optimizer.step()
        self.scheduler.step()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            coord2_e, std = self.model.test(self.im1, self.im2, self.coord1)
        return coord2_e, std

    def extract_features(self, im, coord):
        self.model.eval()
        with torch.no_grad():
            feat_c, feat_f, angle = self.model.extract_features(im, coord)
        return feat_c, feat_f, angle

    def write_summary(self, writer, n_iter):
        print("%s | Step: %d, Loss: %2.5f" % (self.args.exp_name, n_iter, self.j_loss.item()))
        if (self.j_loss.item() < self.lowestloss):
            self.lowestloss = self.j_loss.item()
            ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name + "best")
            os.makedirs(ckpt_folder, exist_ok=True)

            save_path = os.path.join(ckpt_folder, "best_{:06d}.pth".format(n_iter))
            print('saving ckpts {}...'.format(save_path))
            torch.save({'step': n_iter,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        },
                       save_path,
                       _use_new_zipfile_serialization=False)

        # write scalar
        if n_iter % self.args.log_scalar_interval == 0:
            writer.add_scalar('Total_loss', self.j_loss.item(), n_iter)
            writer.add_scalar('epipolar_loss_coarse', self.eloss_c.item(), n_iter)
            writer.add_scalar('epipolar_loss_fine', self.eloss_f.item(), n_iter)
            writer.add_scalar('cycle_loss_coarse', self.closs_c.item(), n_iter)
            writer.add_scalar('cycle_loss_fine', self.closs_f.item(), n_iter)
            writer.add_scalar('ort_loss', self.ort_loss.item(), n_iter)


        # write image
        if n_iter % self.args.log_img_interval == 0:
            # this visualization shows a number of query points in the first image,
            # and their predicted correspondences in the second image,
            # the groundtruth epipolar lines for the query points are plotted in the second image
            num_kpts_display = 20
            im1_o = self.im1_ori[0].numpy()
            im2_o = self.im2_ori[0].numpy()
            kpt1 = self.coord1.cpu().numpy()[0][:num_kpts_display, :]
            # predicted correspondence
            correspondence = self.out['coord2_ef']
            kpt2 = correspondence.detach().cpu().numpy()[0][:num_kpts_display, :]
            lines2 = cv2.computeCorrespondEpilines(kpt1.reshape(-1, 1, 2), 1, self.fmatrix[0].cpu().numpy())
            lines2 = lines2.reshape(-1, 3)
            im2_o, im1_o = utils.drawlines(im2_o, im1_o, lines2, kpt2, kpt1)
            vis = np.concatenate((im1_o, im2_o), 1)
            vis = torch.from_numpy(vis.transpose(2, 0, 1)).float().unsqueeze(0)
            x = vutils.make_grid(vis, normalize=True)
            writer.add_image('Image', x, n_iter)

    def load_model(self, filename):
        to_load = torch.load(filename)
        self.model.load_state_dict(to_load['state_dict'])
        if 'optimizer' in to_load.keys():
            self.optimizer.load_state_dict(to_load['optimizer'])
        if 'scheduler' in to_load.keys():
            self.scheduler.load_state_dict(to_load['scheduler'])
        return to_load['step']

    def load_from_ckpt(self):
        '''
        load model from existing checkpoints and return the current step
        :param ckpt_dir: the directory that stores ckpts
        :return: the current starting step
        '''

        # load from the specified ckpt path
        if self.args.ckpt_path != "":
            print("Reloading from {}".format(self.args.ckpt_path))
            if os.path.isfile(self.args.ckpt_path):
                step = self.load_model(self.args.ckpt_path)
            else:
                raise Exception('no checkpoint found in the following path:{}'.format(self.args.ckpt_path))

        else:
            ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
            os.makedirs(ckpt_folder, exist_ok=True)
            # load from the most recent ckpt from all existing ckpts
            ckpts = [os.path.join(ckpt_folder, f) for f in sorted(os.listdir(ckpt_folder)) if f.endswith('.pth')]
            if len(ckpts) > 0:
                fpath = ckpts[-1]
                step = self.load_model(fpath)
                print('Reloading from {}, starting at step={}'.format(fpath, step))
            else:
                print('No ckpts found, training from scratch...')
                step = 0

        return step

    def save_model(self, step):
        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        save_path = os.path.join(ckpt_folder, "{:06d}.pth".format(step))
        print('saving ckpts {}...'.format(save_path))
        torch.save({'step': step,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    },
                   save_path,
                   _use_new_zipfile_serialization=False)
