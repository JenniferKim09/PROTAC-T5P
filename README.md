The code provided here is for the  **A Data-Driven Hierarchical Framework for PROTAC Degradation Prediction and Target-Specific Candidate Prioritization** paper. First of all, prepare the environment:

- ```
  git clone https://github.com/JenniferKim09/PROTAC-T5P.git
  cd PROTAC-T5P
  conda env create -f environment.yml -n protac-t5
  conda activate protac-t5
  ```

## 1a. Direct download of DATA and CKPT

Datasets used for the training of the model (put under `./data`) and the trained checkpoint (put under `./ckpt`) can be downloaded [here](https://doi.org/10.5281/zenodo.20304396). 


## 1b. Manual preparation of the datasets
If you want to try the data preparation for your own data, use the code as:

- ```
  python lig2ecloud.py 
  ```

The `.csv` file and `.sdf` of your data should be previously prepared and the path mentioned in `lig2ecloud.py` should be changed to your path. Users can refer to the examples `./data/protac-fine4.csv` and `./data/protac-fine4.sdf`. 


## 2. Train and test of PROTAC-T5P

* After the preparation of the datasets, the training and testing of PROTAC-T5P(base4) can be performed by:

- ```
  python property_prediction.py 
  ```

* After the preparation of the datasets, the training and testing of PROTAC-T5P(base4_pro) can be performed by:

- ```
  python property_prediction_pro.py 
  ```

## 3. Machine learning

* Refer to `ml.py` to perform the machine learning. BTK dataset was not provided due to patent-related restrictions but can be replaced with data for the user's own target of interest.













