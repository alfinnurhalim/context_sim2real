
# Sim-to-Real Transfer in Underwater Scene Analysis

## Project Structure

- `config/`: Configuration files
- `Dataset/`: Dataset files
- `model/`: Model architecture and related files
- `README.md`: Project documentation
- `test.py`: Script for testing the model
- `train.py`: Script for training the model
- `weight/`: Pre-trained model weights

## Setup Instructions

### 1. Clone the Repository
```bash
git https://github.com/alfinnurhalim/context_sim2real
cd context_sim2real
```

### 2. Download the Dataset
- Download the dataset and place it in the `Dataset/` directory.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Training the Model
To train the model, run the following command:
```bash
python train.py
```

## Testing the Model
To test the model, run the following command:
```bash
python test.py
```

## Configuration
Modify the configuration files in the `config/` directory to change training/testing parameters.

## Model Architecture
The model architecture is defined in the `model/` directory. Most of the code related to model architecture is borrowed from the StyleFlow original repository. Additional code has been developed to match our objectives of sim-to-real transfer in underwater scenes.

## Results
The results of the experiments, including metrics like SSIM and FID, will be saved in the appropriate directory specified in the configuration files.

## Acknowledgements
This project is based on the following research papers:
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. ArXiv. /abs/1508.06576
- Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ArXiv. /abs/1703.10593
- Fan, W., Chen, J., Ma, J., Hou, J., & Yi, S. (2022). StyleFlow For Content-Fixed Image to Image Translation. ArXiv. /abs/2207.01909
- Fan, W., Chen, J., & Liu, Z. (2023). Hierarchy Flow For High-Fidelity Image-to-Image Translation. ArXiv. /abs/2308.06909
- Zhang, Y., Huang, N., Tang, F., Huang, H., Ma, C., Dong, W., & Xu, C. (2022). Inversion-Based Style Transfer with Diffusion Models. ArXiv. /abs/2211.13203

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
