# Data Engineering and Machine Learning with Airflow & PySpark

## Purpose
Build a full data pipeline that extracts Bitcoin values in a five years interval, also making value predictions using Machine Learning into a 30 days interval, and returns as a final product an interactive Plotly graph in HTML format and a Bitcoin data table in a csv file. These files are on $HOME/BitcoinPredictions.  
  
Another purpose for building this repo is to test Airflow with Pyspark. The final version is very different from the one I had planned to do, using Object-Oriented Programming. In the time I developed this project, I needed to do a lot of adjusts because I had some difficulties to write a more atomized code that could have a good integration between Airflow and PySpark.  
  
The final version is in the 'bitcoin_pyspark_dag.py' file.  
   
I used PySpark Machine Learning framework called Mlib (pyspark.ml), and I also used Gradient Boosting Tree Regressor model to make Bitcoin value predictions, making also hyperparameters adjustment like tree number and cross-validation from this model.  

## Resources
- Visual Studio Code
- python3.9
- virtualenv
- pip3: python3.x packages manager

## Python packages
- airflow
- os
- datetime
- calendar
- plotly
- pandas
- pyspark

## Images from this project
<img src="plot1.png" />

## Running this repo in your local machine
- clone this repo:    
> git clone https://github.com/rafaelcoelho1409/DataEngineering.git  
- install required packages that are in 'dataeng_requirements.txt' file:  
> pip3 install -r dataeng_requirements.txt  
- choose your python interpreter (python3.x)  
- install Airflow:  
> pip3 install apache-airflow  
- Find the Airflow DAGs folder in your machine (generally, $AIRFLOW_HOME/dags)    
- copy and paste 'bitcoin_pyspark_dag.py' file into this folder ($AIRFLOW_HOME/dags)  
- access Airflow on your browser (https://localhost:8080)  
- Activate the DAG and trigger it  
- More orientations about Airflow data automation:  
> https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html
