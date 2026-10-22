# DCGAN Fashion-MNIST Practical

A compact, cell-by-cell Jupyter notebook that implements and trains a Deep Convolutional Generative Adversarial Network (DCGAN) on Fashion-MNIST.

## Contents

- `DCGAN_FashionMNIST_Practical.ipynb` — dataset preparation, Generator and Discriminator, training, generated samples, loss curves, diversity check, and analysis/viva notes.
- `requirements.txt` — Python packages needed to run the notebook.
- `.gitignore` — excludes downloaded data, generated images, model checkpoints, and notebook cache files.

## Run the practical

1. Install Python 3.10 or later and Jupyter Notebook or JupyterLab.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start Jupyter in this project folder and open `DCGAN_FashionMNIST_Practical.ipynb`.
4. Run the cells from top to bottom. The first dataset cell downloads Fashion-MNIST automatically.

A CUDA-enabled GPU is recommended, but the notebook also runs on CPU. CPU training can take considerably longer. The default training length is 10 epochs for a practical demonstration; increase `epochs` in the notebook for longer training.

## Outputs

The notebook saves these files in an `outputs/` folder when the final save cell runs:

- `dcgan_fashion_samples.png` — generated image grid.
- `dcgan_fashion_mnist.pt` — Generator and Discriminator weights plus recorded loss values.

These generated files and the downloaded dataset are excluded from Git by `.gitignore`. Commit them only if you intentionally want to include them in your repository.

## Dataset

Fashion-MNIST is provided through `torchvision.datasets.FashionMNIST`. It contains 28×28 grayscale clothing images across ten categories. Images are normalized to `[-1, 1]` to match the Generator's `Tanh` output.

## Notes on evaluation

GAN loss curves can oscillate and should not be read as ordinary classification accuracy. Inspect generated grids for visual quality and variety. The notebook's pixel-wise diversity value is only a rough indicator; it is not a substitute for a perceptual metric such as FID.
