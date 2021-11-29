# EC400_Final_Project

Data:

    python3 -m homework.utils zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland

Train:

    python3 -m homework.planner [TRACK_NAME] -v

Test:

    python3 -m grader homework


Initial NN:

```
input = 3        
output = 32
kernel_size = 5
stride = 2

layers.append(NN.Conv2d(input, output, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))
layers.append(NN.BatchNorm2d(output))
layers.append(NN.ReLU())

layers.append(NN.Conv2d(output, 2 * output, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))
layers.append(NN.BatchNorm2d(2 * output))
layers.append(NN.ReLU())

layers.append(NN.Conv2d(2 * output, 4 * output, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))
layers.append(NN.BatchNorm2d(4 * output))
layers.append(NN.ReLU())

layers.append(NN.Conv2d(4 * output, 2 * output, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))
layers.append(NN.BatchNorm2d(2 * output))
layers.append(NN.ReLU())

layers.append(NN.Conv2d(2 * output, output, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))
layers.append(NN.BatchNorm2d(output))
layers.append(NN.ReLU())

layers.append(NN.Conv2d(output, 1, kernel_size=kernel_size,
                        padding=kernel_size // 2, stride=stride))

self._conv = torch.nn.Sequential(*layers)

```