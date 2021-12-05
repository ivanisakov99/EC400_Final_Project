import torch
import torch.nn as NN
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

        super().__init__()

        layers = []
        
        # Lowest Loss = 0.014
        # input = 3        
        # output = 32
        # kernel_size = 5
        # stride = 2
    
        # layers.append(NN.Conv2d(input, output, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))
        # layers.append(NN.BatchNorm2d(output))
        # layers.append(NN.ReLU())

        # layers.append(NN.Conv2d(output, 2 * output, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))
        # layers.append(NN.BatchNorm2d(2 * output))
        # layers.append(NN.ReLU())

        # layers.append(NN.Conv2d(2 * output, 4 * output, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))
        # layers.append(NN.BatchNorm2d(4 * output))
        # layers.append(NN.ReLU())

        # layers.append(NN.Conv2d(4 * output, 2 * output, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))
        # layers.append(NN.BatchNorm2d(2 * output))
        # layers.append(NN.ReLU())

        # layers.append(NN.Conv2d(2 * output, output, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))
        # layers.append(NN.BatchNorm2d(output))
        # layers.append(NN.ReLU())

        # layers.append(NN.Conv2d(output, 1, kernel_size=kernel_size,
        #                         padding=kernel_size // 2, stride=stride))

        # self._conv = torch.nn.Sequential(*layers)

        input = 3
        kernel_size = 3
        output = 1

        self.conv11 = NN.Conv2d(input, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn11 = NN.BatchNorm2d(64)
        self.conv12 = NN.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn12 = NN.BatchNorm2d(64)

        self.conv21 = NN.Conv2d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn21 = NN.BatchNorm2d(128)
        self.conv22 = NN.Conv2d(128, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn22 = NN.BatchNorm2d(128)

        self.conv31 = NN.Conv2d(128, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn31 = NN.BatchNorm2d(256)
        self.conv32 = NN.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn32 = NN.BatchNorm2d(256)
        self.conv33 = NN.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn33 = NN.BatchNorm2d(256)

        self.conv33d = NN.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn33d = NN.BatchNorm2d(256)
        self.conv32d = NN.Conv2d(256, 256, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn32d = NN.BatchNorm2d(256)
        self.conv31d = NN.Conv2d(256,  128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn31d = NN.BatchNorm2d(128)

        self.conv22d = NN.Conv2d(128, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn22d = NN.BatchNorm2d(128)
        self.conv21d = NN.Conv2d(128, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn21d = NN.BatchNorm2d(64)

        self.conv12d = NN.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn12d = NN.BatchNorm2d(64)
        self.conv11d = NN.Conv2d(64, output, kernel_size=kernel_size, padding=kernel_size // 2)


    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(img)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(
            x12, kernel_size=2, stride=2, return_indices=True)
        size1 = x12.size()

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(
            x22, kernel_size=2, stride=2, return_indices=True)
        size2 = x22.size()

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(
            x33, kernel_size=2, stride=2, return_indices=True)
        size3 = x33.size()

        # Stage 3d
        x3d = F.max_unpool2d(x3p, id3, kernel_size=2,
                             stride=2, output_size=size3)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2,
                             stride=2, output_size=size2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2,
                             stride=2, output_size=size1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        
        x = x11d


        # Previous
        # x = self._conv(img)

        # print(img.shape)
        # print(x.shape)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(
                t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
