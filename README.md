# Flytracker

This repo contains the code for the flytracker project. 

## How to install 
We don't have a list of packages we require (yet), but we don't use anything special so you might have all the packages already. We use
* **OpenCV** - To load videos, correct for lens abberations and its blobtracker.
* **NumPy** - Because it's fucking NumPy.
* **Scipy** - To perform the tracking.
* **Scikit-learn** - For kmeans benchmarking.
* **PyTorch** - To run everything on a gpu.

To use the code, clone it by running:
```
git clone https://github.com/nicolaseberle/flyTracker.git
```

Then go into the folder and run 

```
pip install -e .
```

This installs it as a development package which, since we're still in development, is the easiest for now.

## How to use

You can find a few notebooks with examples how to use the code in the examples folder.
