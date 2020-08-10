# German Traffic Sign Recognition



## <u>About the Project</u>

This project is an implementation of recognition of traffic signs using deep learning. We are using the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=news) along with the Spatial Transformer Network that is proposed in [Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods.](https://www.sciencedirect.com/science/article/pii/S0893608018300054?casa_token=afsB9kq_U2EAAAAA:DV_6RTfBv_uXzZ27SAHceBr4l5zjfvqzfNGm90WnUWZEOigpjX73pUPEDFlWre82oLrqhTN5-P-7).



***This project uses the PyTorch Deep Learning Framework.***



## <u>Project Structure</u>

* ***Note: The input folder in this repo will be empty. I have included all the sub-folders without any files so that you can easily set-up the project directory as I have. But you are free to setup your input data directory as you like it. You just have to change the paths in the python programs.***

* **LINK TO DOWNLOAD ALL THE TRAINING AND TEST DATA => https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html**

* The following is the structure that I have used.

  ```
  ├───input
  │   └───german_traffic_sign
  │       └───GTSRB
  │           ├───Final_Test
  │           │   └───Images
  │           └───Final_Training
  │               └───Images
  ├───notebooks
  │   └───...
  ├───outputs
  └───src
      └───*.py files
  ```

### Important Not About the Paths in the Python Scripts

* **In the python scripts, you may find that the paths to files and images may be in the following structure**:
  * `../../input/path/to/files/and/images`
* **The extra `../`  is because of how I arrange my vision datasets globally** **You can easily omit the first `../` So, for you, the path will become**:
  * `../input/path/to/files/and/images`
* **That's it. You are good to go.**



## <u>References</u>

* [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=news).

* [Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods.](https://www.sciencedirect.com/science/article/pii/S0893608018300054?casa_token=afsB9kq_U2EAAAAA:DV_6RTfBv_uXzZ27SAHceBr4l5zjfvqzfNGm90WnUWZEOigpjX73pUPEDFlWre82oLrqhTN5-P-7).
* [ppriyank](https://github.com/ppriyank)/[Deep-neural-network-for-traffic-sign-recognition-systems](https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems) for model code.
* [LCN (Local Contrast Normalization) code.](https://github.com/dibyadas/Visualize-Normalizations)

