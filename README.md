# Pop-Out Motion
## Pop-Out Motion: 3D-Aware Image Deformation via Learning the Shape Laplacian (CVPR 2022) ##

[Jihyun Lee](https://jyunlee.github.io/)\*, [Minhyuk Sung](https://mhsung.github.io/)\*, Hyunjin Kim, [Tae-Kyun (T-K) Kim](https://sites.google.com/view/tkkim/home) (*: equal contributions)

**[\[Project Page\]](https://jyunlee.github.io/projects/pop-out-motion) [\[Paper\]](https://arxiv.org/abs/2203.15235) [\[Supplementary Video Results\]](https://youtu.be/gHxwHxIZiuM)**

---

**CVPR 2022 Materials: [\[Presentation Video\]](https://jyunlee.github.io/projects/pop-out-motion) **<a href="https://jyunlee.github.io/projects/pop-out-motion/data/cvpr2022_poster.pdf" class="image fit" type="application/pdf">**\[Poster\]**</a>


<p align="center">
  <img src="teaser.gif" alt="animated" />
</p>

> We present a framework that can deform an object in a 2D image as it exists in 3D space. While our method leverages 2D-to-3D reconstruction, we argue that reconstruction is not sufficient for realistic deformations due to the vulnerability to topological errors. Thus, we propose to take a supervised learning-based approach to predict the shape Laplacian of the underlying volume of a 3D reconstruction represented as a point cloud. Given the deformation energy calculated using the predicted shape Laplacian and user-defined deformation handles (e.g., keypoints), we obtain bounded biharmonic weights to model plausible handle-based image deformation.

&nbsp;

## Environment Setup  
Clone this repository and install the dependencies specified in `requirements.txt`.
<pre><code> git clone https://github.com/jyunlee/Pop-Out-Motion.git
 mv Pop-Out-Motion
 pip install -r requirements.txt </pre></code>

&nbsp;

## Data Pre-Processing  
### Training Data
1. Build executables from the c++ files in `data_preprocessing` directory. After running the commands below, you should have  `normalize_bin` and `calc_l_minv_bin` executables.
<pre><code> cd data_preprocessing
 mkdir build
 cd build
 cmake ..
 make
 cd ..</pre></code>
2. Clone and build [Manifold](https://github.com/hjwdzh/Manifold) repository to obtain `manifold` executable.

3. Clone and build [fTetWild](https://github.com/wildmeshing/fTetWild) repository to obtain `FloatTetwild_bin` executable.

4. Run `preprocess_train_data.py` to prepare your training data. This should perform (1) shape normalization into a unit bounding sphere, (2) volume mesh conversion, and (3) cotangent Laplacian and inverse mass calculation.
<pre><code> python preprocess_train_data.py </code></pre>
 
 
### Test Data
1. Build executables from the c++ files in `data_preprocessing` directory. After running the commands below, you should have  `normalize_bin` executable.
<pre><code> cd data_preprocessing
 mkdir build
 cd build
 cmake ..
 make
 cd ..</pre></code>

2. Run `preprocess_test_data.py` to prepare your test data. This should perform (1) shape normalization into a unit bounding sphere and (2) pre-computation of KNN-Based Point Pair Sampling (KPS).
<pre><code> python preprocess_test_data.py </code></pre>

&nbsp;

## Network Training
Run `network/train.py` to train your own Laplacian Learning Network.
<pre><code> cd network
 python train.py </pre></code>
The pre-trained model on DFAUST dataset is also available [here](https://drive.google.com/drive/folders/1pMVi9b4DH6bIrkgkWuEzYB9LtFdGwjhC?usp=sharing).

&nbsp;

## Network Inference
**Deformation Energy Inference**
1. Given an input image, generate its 3D reconstruction via running [PIFu](https://github.com/shunsukesaito/PIFu). It is also possible to directly use point cloud data obtained from other sources.

2. Pre-process the data obtained from *Step 1* -- please refer to [this section](#test-data).

3. Run `network/a_inference.py` to predict the deformation energy matrix.
<pre><code> cd network
 python a_inference.py </pre></code>
 
 
**Handle-Based Deformation Weight Calculation**

1. Build an executable from the c++ file in `bbw_calculation` directory. After running the commands below, you should have  `calc_bbw_bin` executable.
<pre><code> cd bbw_calculation
 mkdir build
 cd build
 cmake ..
 make
 cd ..</pre></code>

2. (Optional) Run `sample_pt_handles.py` to obtain deformation control handles sampled by farthest point sampling.

3. Run `calc_bbw_bin` to calculate handle-based deformation weights using the predicted deformation energy. 
```
./build/calc_bbw_bin <shape_path> <handle_path> <deformation_energy_path> <output_weight_path>
```

&nbsp;

## Citation
If you find this work useful, please consider citing our paper.
```
@InProceedings{lee2022popoutmotion,
    author = {Lee, Jihyun and Sung, Minhyuk and Kim, Hyunjin and Kim, Tae-Kyun},
    title = {Pop-Out Motion: 3D-Aware Image Deformation via Learning the Shape Laplacian},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```
&nbsp;

## Acknowledgements

 - Parts of our data-preprocessing code are adopted from [DeepMetaHandles](https://github.com/Colin97/DeepMetaHandles).
 - Parts of our network code are adopted from [Point-Transformer](https://github.com/POSTECH-CVLab/point-transformer).
