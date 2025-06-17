# RESAMPL-UDA : Leveraging foundation models for unsupervised domain adaptation in biomedical images
## Pattern Recognition Letters
This is the repository for the following paper : https://hal.science/hal-05104207/

### Environement

This code was runned using ```python-3.11.4```
libraries used can be find in the file ```requirements.txt```

### Extracting shapes with SAM
To use SAM in "segment anything mode", we used ```SamAutomaticMaskGenerator``` from the segment anything official repo : https://github.com/facebookresearch/segment-anything
Nonetheless, you can use any code that allows you to extract the shapes using SAM without human prompt.
Once you generated your SAM predictions, please put them in the file structure described bellow

### Training instructions
Once you have put your data in the structure discribed at the bottom of the repo, if you are using slurm, please modify the file ```generate_slurm.sh``` in order to generate automatically the slurm scripts to train on your own data.
Then, you can launch ```sh generate_slurm.sh``` and it will generate you two files : ```launch_train_jobs.sh``` and ```launch_test_jobs.sh```. You can launch them first to train the model and then to test it.
If you are not using slurm, you can manually use the python lauching commands that are in the ```generate_slurm.sh```

### Source-Free UDA extension
In the same way that explained above, there is a ```generate_slurm_sfuda.sh``` that allows you to perform this SFUDA extension.


### Dataset 
The data should be in the following structure :

```
├── train
    ├── img
        ├── images
            ├── 1.png
            ├── 2.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 1.png
            ├── 2.png
            ├── ...
    ├── pred_sam
        ├── sam
            ├── 1.png
            ├── 2.png
            ├── ...
├── val
    ├── img
        ├── images
            ├── 5001.png
            ├── 5002.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 5001.png
            ├── 5002.png
            ├── ...
    ├── pred_sam
        ├── sam
            ├── 5001.png
            ├── 5002.png
            ├── ...
├── test
    ├── img
        ├── images
            ├── 6001.png
            ├── 6002.png
            ├── ...
    ├── lbl
        ├── labels
            ├── 6001.png
            ├── 6002.png
            ├── ...
    ├── pred_sam
        ├── sam
            ├── 6001.png
            ├── 6002.png
            ├── ...

```
