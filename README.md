# German Traffic Sign Recognition

[![xs:code](https://img.shields.io/static/v1?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAiIGhlaWdodD0iNDUiIHZpZXdCb3g9IjAgMCAzMCA0NSIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwLjgzMjEgMzcuNDg4OVYyOS40NzJDMjAuODMyMSAyNi43NjQ1IDE5LjI5NjggMjQuNjY5MiAxNi44NTMxIDI0Ljk1NDJWMjAuMTgzMUMxOS4yOTY4IDIwLjQ2MjggMjAuODMyMSAxOC4zNTcgMjAuODMyMSAxNS42NDQyVjcuNjI3MjhDMjAuODMyMSAwLjI5MTE2NiAyNS4xMjQ2IDAuODEzNjY1IDI5LjM3NDYgMC44MTM2NjVWNC44MDM2NkMyNi45MzA5IDQuNTIzOTQgMjUuMzk1NiA0LjkxNDUgMjUuMzk1NiA3LjYyNzI4VjE1LjY0NDJDMjUuMzk1NiAxOC43NDc2IDI0LjQyODcgMjEuMTE3MyAyMi4zODM0IDIyLjUzMTdDMjQuNDI4NyAyMy45OTg5IDI1LjM5NTYgMjYuMzY4NyAyNS4zOTU2IDI5LjQ3MlYzNy40ODg5QzI1LjM5NTYgNDAuMTk2NCAyNi45MzA5IDQwLjU5MjMgMjkuMzc0NiA0MC4zMTI2VjQ0LjYwMzRDMjUuNjkzMSA0NC41OTgxIDIwLjgzMjEgNDQuODI1MSAyMC44MzIxIDM3LjQ4ODlaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMC41MDY0MDkgNDQuNTk4MVY0MC4zMDczQzIuOTUwMTYgNDAuNTg3IDQuNDg1NDcgNDAuMTk2NCA0LjQ4NTQ3IDM3LjQ4MzZWMjkuNDcyQzQuNDg1NDcgMjYuMzY4NiA1LjQ1MjM1IDIzLjk5ODkgNy40OTc2NiAyMi41MzE3QzUuNDUyMzUgMjEuMTIyNSA0LjQ4NTQ3IDE4Ljc0NzUgNC40ODU0NyAxNS42NDQyVjcuNjI3MjVDNC40ODU0NyA0LjkxOTc1IDIuOTUwMTYgNC41MjM5MiAwLjUwNjQwOSA0LjgwMzY0VjAuODEzNjM4QzQuNzU2NDEgMC44MTM2MzggOS4wNDg5MSAwLjI4NTg2IDkuMDQ4OTEgNy42MjcyNVYxNS42NDQyQzkuMDQ4OTEgMTguMzUxNyAxMC41ODQyIDIwLjQ2MjggMTMuMDI4IDIwLjE4MzFWMjQuOTU0MkMxMC41ODQyIDI0LjY3NDUgOS4wNDg5MSAyNi43NjQ1IDkuMDQ4OTEgMjkuNDcyVjM3LjQ4ODlDOS4wNDg5MSA0NC44MjUgNC4xOTMyOCA0NC41OTgxIDAuNTA2NDA5IDQ0LjU5ODFaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K&message=xs:code&label=Ping+me+on&color=%23007EFF)](https://xscode.com/profile/sovit-123)

## <u>About the Project</u>

This project is an implementation of recognition of traffic signs using deep learning. We are using the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=news) along with the Spatial Transformer Network that is proposed in [Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods.](https://www.sciencedirect.com/science/article/pii/S0893608018300054?casa_token=afsB9kq_U2EAAAAA:DV_6RTfBv_uXzZ27SAHceBr4l5zjfvqzfNGm90WnUWZEOigpjX73pUPEDFlWre82oLrqhTN5-P-7).



**Feel free to use, modify, distribute, and build upon the project in whatever way you feel it will be useful and better.**

***[Download the trained weights file from here](https://drive.google.com/file/d/1W1E3_0VLqVj4ERh60uMA1WiNFUtxJ1Gy/view?usp=sharing)***.



## <u>Contents</u>

* **[Dependencies and Frameworks](#Dependencies-and-Frameworks).**
* **[Project Structure](#Project-Structure).**
  * **[Important Note About the Paths in the Python Scripts](#Important-Note-About-the-Paths-in-the-Python-Scripts).**
* **[Steps to Train and Test](#Steps-to-Train-and-Test).**
* **[Results](#Results).**
* **[Some Results on the Test Data](#Some-Results-on-the-Test-Data).**
* **[References](#References).**



## <u>Dependencies and Frameworks</u>

* [PyTorch >= 1.4](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00008.png?raw=true).
* [Albumentations](https://albumentations.ai/).
* [Scikit-Learn](https://scikit-learn.org/stable/index.html).



***Now, just to give an idea of what to results expect from this project:***

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00008.png?raw=true)

***In my opinion, it would have been pretty difficult for a human to tell what the sign is***.



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

### Important Note About the Paths in the Python Scripts

* **In the python scripts, you may find that the paths to files and images may be in the following structure**:
  * `../../input/path/to/files/and/images`
* **The extra `../`  is because of how I arrange my vision datasets globally** **You can easily omit the first `../` So, for you, the path will become**:
  * `../input/path/to/files/and/images`
* **That's it. You are good to go.**



## <u>Steps to Train and Test</u>

* **If you have the compute power and want to train your own network, then execute the files in the following order:**
  * Execute `preprocess.py` just one.
    * `python preprocess.py`
  * Then execute `train.py`.
    * `python train.py`
* **Now, if you just want to test the network by loading the trained weights, then:**
  * Download the weights from [**here**](https://drive.google.com/file/d/1srFZ95FDiRkRClyseotxkUExVUwIYjEk/view?usp=sharing).
  * Then just execute `test.py`.
    * `python test.py`



## <u>Results</u>

| After 20 epochs | Accuracy | Loss   |
| --------------- | -------- | ------ |
| **Training**    | 99.63%   | 0.0001 |
| **Validation**  | 98.72%   | 0.0002 |

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/accuracy.png?raw=true)

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/loss.png?raw=true)



## <u>Some Results on the Test Data</u>

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00000.png?raw=true)

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00001.png?raw=true)

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00004.png?raw=true)

![](https://github.com/sovit-123/German-Traffic-Sign-Recognition-with-Deep-Learning/blob/master/outputs/00009.png?raw=true)





## <u>References</u>

* [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=news).

* [Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods.](https://www.sciencedirect.com/science/article/pii/S0893608018300054?casa_token=afsB9kq_U2EAAAAA:DV_6RTfBv_uXzZ27SAHceBr4l5zjfvqzfNGm90WnUWZEOigpjX73pUPEDFlWre82oLrqhTN5-P-7).
* [ppriyank](https://github.com/ppriyank)/[Deep-neural-network-for-traffic-sign-recognition-systems](https://github.com/ppriyank/Deep-neural-network-for-traffic-sign-recognition-systems) for model code.
* [LCN (Local Contrast Normalization) code.](https://github.com/dibyadas/Visualize-Normalizations)

