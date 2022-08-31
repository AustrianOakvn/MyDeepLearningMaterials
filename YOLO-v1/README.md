# YOLO algorithm

## Overview

### **Paper**: You only look once, Unified, Real-time Object Detection.

![image info](./imgs/s_by_s_grid.png)

- An image is going to be split into SxS grid. For example if you split an image into 3x3 grid, you will have 9 cells
- Each cell will output a prediction with a corresponding bounding box. But an object may span over multiple cells and we only want to have one bounding box so the. We would find one cell that responsible for that object and the cell is chosen to be the cell which contains the mid-point of the object.
- Each cell would have a top left corner with (0, 0) and the bottom right with coordinate (1, 1)
- Each output and label will be relative to the cell. Each bounding box for each cell will have $[x, y, w, h]$. $x$ and $y$ are the coordinates for object midpoint in cell. $x$ and $y$ are in range [0, 1] but w and h can be greater than 1 if object is taller than the cell.

- Labels format for one cell:

$label_{cell} = [c_1, c_2,...,c_{20}, p_c, x, y, w, h]$

Where ${c_1, c_2,..., c_{20}}$ is for 20 different classes. $p_c$ is the probability that there is an object (0 or 1). $x, y, w, h$ are for the bounding box coordinates

- The predictions will look very similar but we will output two bounding boxes to cover cases. Prediction format:

$pred_{cell}=[c_1, c_2,...,c_{20}, p_{c1}, x, y, w, h, p_{c2}, x, y, w, h]$

Note: A cell can only detect one object

For every cell:

- Target shape for **one image**: $(S, S, 25)$
- Prediction shape for **one image**: $(S, S, 30)$

## Architecture and implementation

![image info](./imgs/yolo_architecture.png)

Reminder formula to calculate the output of convolutional and max pooling layer:
$Output_size = floor([(W-K+2P)/S])+1$

- $W$ is the input size
- $K$ is the kernel size
- $P$ is the padding
- $S$ is the stride

Input image --> resize (448, 448, 3)
24 Layers to extract features:

| Layer number   | Kernel size | Number of kernel | Stride | Padding | Input size        | Output size       |
| -------------- | ----------- | ---------------- | ------ | ------- | ----------------- | ----------------- |
| $1$ (conv)     | $(7, 7)$    | $64$             | $2$    | $3$     | $(3, 448, 448)$   | $(64, 224, 224)$  |
| $2$ (maxpool)  | $(2, 2)$    | $NA$             | $2$    | $0$     | $(64, 224, 224)$  | $(64, 112, 112)$  |
| $3$ (conv)     | $(3, 3)$    | $192$            | $1$    | $1$     | $(64, 112, 112)$  | $(192, 112, 112)$ |
| $4$ (maxpool)  | $(2, 2)$    | $NA$             | $2$    | $0$     | $(192, 112, 112)$ | $(192, 56, 56)$   |
| $5$ (conv)     | $(1, 1)$    | $128$            | $1$    | $0$     | $(192, 56, 56)$   | $(128, 56, 56)$   |
| $6$ (conv)     | $(3, 3)$    | $256$            | $1$    | $1$     | $(128, 56, 56)$   | $(256, 56, 56)$   |
| $7$ (conv)     | $(1, 1)$    | $256$            | $1$    | $0$     | $(256, 56, 56)$   | $(256, 56, 56)$   |
| $8$ (conv)     | $(3, 3)$    | $512$            | $1$    | $1$     | $(256, 56, 56)$   | $(512, 56, 56)$   |
| $9$ (maxpool)  | $(2, 2)$    | $NA$             | $2$    | $0$     | $(512, 56, 56)$   | $(512, 28, 28)$   |
| $10$ (conv)    | $(1, 1)$    | $256$            | $1$    | $0$     | $(512, 28, 28)$   | $(256, 28, 28)$   |
| $11$ (conv)    | $(3, 3)$    | $512$            | $1$    | $1$     | $(256, 28, 28)$   | $(512, 28, 28)$   |
| $12$ (conv)    | $(1, 1)$    | $256$            | $1$    | $0$     | $(512, 28, 28)$   | $(256, 28, 28)$   |
| $13$ (conv)    | $(3, 3)$    | $512$            | $1$    | $1$     | $(256, 28, 28)$   | $(512, 28, 28)$   |
| $14$ (conv)    | $(1, 1)$    | $256$            | $1$    | $0$     | $(512, 28, 28)$   | $(256, 28, 28)$   |
| $15$ (conv)    | $(3, 3)$    | $512$            | $1$    | $1$     | $(256, 28, 28)$   | $(512, 28, 28)$   |
| $16$ (conv)    | $(1, 1)$    | $256$            | $1$    | $0$     | $(512, 28, 28)$   | $(256, 28, 28)$   |
| $17$ (conv)    | $(3, 3)$    | $512$            | $1$    | $1$     | $(256, 28, 28)$   | $(512, 28, 28)$   |
| $18$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(512, 28, 28)$   | $(1024, 28, 28)$  |
| $19$ (maxpool) | $(2, 2)$    | $NA$             | $2$    | $0$     | $(1024, 28, 28)$  | $(1024, 14, 14)$  |
| $20$ (conv)    | $(1, 1)$    | $512$            | $1$    | $0$     | $(1024, 14, 14)$   | $(512, 14, 14)$   |
| $21$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(512, 14, 14)$   | $(1024, 14, 14)$  |
| $22$ (conv)    | $(1, 1)$    | $512$            | $1$    | $0$     | $(1024, 14, 14)$   | $(512, 14, 14)$   |
| $23$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(512, 14, 14)$   | $(1024, 14, 14)$  |
| $24$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(1024, 14, 14)$   | $(1024, 14, 14)$  |
| $25$ (conv)    | $(3, 3)$    | $1024$           | $2$    | $1$     | $(1024, 14, 14)$   | $(1024, 7, 7)$  |
| $26$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(1024, 7, 7)$   | $(1024, 7, 7)$  |
| $27$ (conv)    | $(3, 3)$    | $1024$           | $1$    | $1$     | $(1024, 7, 7)$   | $(1024, 7, 7)$  |

2 Fully connected layers



## Loss function

Loss function of YOLO-v1 from the paper:

![image info](./imgs/loss_function.png)

## Acknowledgement
