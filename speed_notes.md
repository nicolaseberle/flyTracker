# Implementation comparison

Here we compare various implementations across machines and tasks to compare speeds.

| Type                               | Loading data | Complete run (1000 frames) | Complete run (whole movie) |
| ---------------------------------- | -----------: | -------------------------: | -------------------------: |
| Baseline-workstation               |        3.4 s |                     6.00 s |                            |
| Pytorch-workstation with undistort |        3.9 s |                            |                            |
| Pytorch-workstation sans undistort |        3.4 s |
