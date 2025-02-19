# DS5230

## Setup
* Please check the [Project Plan](doc/Project%20plan.md).
* For using command line, follow instructions in this [GitHub page](https://github.com/ds5110/git-intro/blob/main/setup.md). "Make" package is highly recommended for the reproducibility using command line.
* You also have an option to simply use Colab.
* If you want to convert .ipynb file(Jupyter notebook) into a .py file(script), you can use [nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html#convert-notebook). Following command is an example of converting a '.ipynb' file to a '.py' file:
```
jupyter nbconvert --to script EDA_Shill_Bidding_Sun830pm.ipynb
```

* Import environment.yml with following command:
```
conda env create --name ds5230 -f environment.yml
```
* You can activate the environment with following command:
```
conda activate ds5230
```

## Group Assignment 1
* [Link](https://colab.research.google.com/drive/186SsqkoZwK0R5gcvTO00hBqIVFs_oFEs?usp=sharing) to Colab version.
* [Link](https://docs.google.com/presentation/d/1tvC9Ljs2UG3cjI59K5eEALSGA9bNqlZChpDjtmt5LuE/edit?usp=sharing) to slidedeck.

* Setup data within the directory by following command:
```
make data
```

* After running the above command, you should have a 'data' directory with 'Online Retail.xlsx' and 'Shill Bidding Dataset.csv' in it now.

* Preprocess datasets with following command:
```
make preprocess
```

* Jupyter version of the preprocessed datasets can be found here:
    * [Shill Bidding Dataset](src/OnlineRetail.ipynb)
    * [Online Retail](src/EDA_Shill_Bidding_Sun830pm.ipynb)
