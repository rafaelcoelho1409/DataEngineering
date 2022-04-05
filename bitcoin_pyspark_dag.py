import airflow
import os
import datetime
import calendar
import plotly.express as px
import pandas as pd
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

args = {
    'owner': 'rafaelcoelho',
    'start_date': airflow.utils.dates.days_ago(3),
    'depends_on_past': False,
    'retries': 5,
    'retries_delay': timedelta(seconds = 30)}

dag = DAG(
    dag_id = 'bitcoin_dag',
    default_args = args,
    schedule_interval = timedelta(days = 1))

################## Arguments ############################
home = os.environ.get('HOME')
spark = SparkSession.builder.appName('BitcoinPredictions').getOrCreate()
#########################################################

############ Functions ##################################
def create_folder():
    #Cria uma pasta para armazenar as informações sobre bitcoin
    os.chdir(home)
    try:
        os.mkdir('BitcoinPredictions')
        os.chdir('BitcoinPredictions')
    except:
        pass

def get_data():
    #Baixa um csv com os dados de bitcoin dentro de um período de 5 anos
    os.chdir(f'{home}/BitcoinPredictions')
    now = datetime.datetime.now()
    start_date = calendar.timegm((now.year - 5, now.month, now.day, 0, 0, 0))
    end_date = calendar.timegm((now.year, now.month, now.day, 0, 0, 0))
    df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true')
    df.to_csv('BTC-USD.csv')

def pyspark_model_data(**context):
    data = spark.read.csv(f'{home}/BitcoinPredictions/BTC-USD.csv', header = True, inferSchema = True)
    data = data.withColumnRenamed('Close', 'label')
    assembler = VectorAssembler(inputCols = ['Open', 'High', 'Low', 'Volume'], outputCol = 'features')
    processed_data = assembler.transform(data)
    _train = processed_data.collect()[:-31]
    _test = processed_data.collect()[-31:]
    train = spark.createDataFrame(_train)
    test = spark.createDataFrame(_test)
    evaluator = RegressionEvaluator()
    model = GBTRegressor()
    paramGrid = ParamGridBuilder().addGrid(model.maxDepth, [10,15,20]).build()
    cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = paramGrid,
        evaluator = evaluator,
        numFolds = 3)
    cv_model = cv.fit(train)
    best_model = cv_model.bestModel
    predictions = best_model.transform(test)
    metrics = {}
    for metric in ['mse', 'rmse', 'mae', 'r2']:
        metrics[metric] = RegressionEvaluator(metricName = metric).evaluate(predictions)
        print(f'{metric}: {metrics[metric]}')
    plot_train = train.orderBy(train['Date'].desc()).limit(30).toPandas().sort_values(by = 'Date')
    plot_test = test.toPandas()
    plot_predicted = predictions.toPandas()[:-1]
    plot_today_predicted = predictions.toPandas()[-2:]
    plot_train = plot_train.append(plot_test.iloc[0], ignore_index = True)
    fig1 = px.line(
        plot_train.rename(columns = {'label': 'Treino'}), 
        x = 'Date', 
        y = 'Treino', 
        markers = True, 
        title = 'Valor do Bitcoin (em US$)')
    fig1.data[0].line.color = '#0000ff'
    fig2 = px.line(
        plot_test.rename(columns = {'label': 'Teste'}), 
        x = 'Date', 
        y = 'Teste', 
        markers = True)
    fig2.data[0].line.color = '#ff0000'
    fig1 = fig1.add_trace(fig2.data[0])
    fig3 = px.line(
        plot_today_predicted.rename(columns = {'prediction': 'Predição de hoje'}), 
        x = 'Date', 
        y = 'Predição de hoje',
        markers = True)
    fig3.data[0].line.color = '#00ff00'
    fig1 = fig1.add_trace(fig3.data[0])
    fig4 = px.line(
        plot_predicted.rename(columns = {'prediction': 'Predição de teste'}), 
        x = 'Date', 
        y = 'Predição de teste',
        markers = True)
    fig4.data[0].line.color = '#ffa500'
    fig1 = fig1.add_trace(fig4.data[0])
    fig1 = fig1.update_xaxes(title_text = 'Azul: Treino<br>Vermelho: Teste<br>Laranja: Predição de teste<br>Verde: Predição de hoje')
    fig1 = fig1.update_yaxes(title_text = 'Valor (em US$)')
    fig1 = fig1.update_traces(showlegend = True)
    date_path = '{}/{}'.format(f'{home}/BitcoinPredictions', datetime.datetime.now())
    os.mkdir(date_path)
    os.chdir(date_path)
    fig1.write_html('final_plot.html')
#########################################################

_requirements = BashOperator(
    task_id = 'requirements',
    bash_command = 'cd ~/airflow/dags && pip3 install -r dataeng_requirements.txt',
    dag = dag)

_create_folder = PythonOperator(
    task_id = 'create_folder',
    python_callable = create_folder,
    dag = dag)

_get_data = PythonOperator(
    task_id = 'get_data',
    python_callable = get_data,
    dag = dag)

_pyspark_model_data = PythonOperator(
    task_id = 'pyspark_model_data',
    python_callable = pyspark_model_data,
    dag = dag)

#_gradient_boosting = PythonOperator(
#    task_id = 'gradient_boosting',
#    python_callable = gradient_boosting,
#    dag = dag)
#
#_predict = PythonOperator(
#    task_id = 'predict',
#    python_callable = predict,
#    dag = dag)
#
#_evaluate = PythonOperator(
#    task_id = 'evaluate',
#    python_callable = evaluate,
#    dag = dag)
#
#_plotly_results = PythonOperator(
#    task_id = 'plotly_results',
#    python_callable = plotly_results,
#    dag = dag)
#
#_save_plot = PythonOperator(
#    task_id = 'save_plot',
#    python_callable = save_plot,
#    dag = dag)


_requirements >> _create_folder >>_get_data >> _pyspark_model_data


