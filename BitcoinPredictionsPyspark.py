#Autor: Rafael Silva Coelho
########### Imports ############################################
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.types as T
import pyspark.sql.functions as F
import plotly.express as px
import os
import wget
import datetime
import calendar

########################### Actions ###################################################
class BitcoinPredictionsPyspark:
    def __init__(self):
        self.spark = SparkSession.builder.appName('BitcoinPredictions').getOrCreate()

    def create_folder(self):
        #Cria uma pasta para armazenar as informações sobre bitcoin
        self.home = os.environ.get('HOME')
        os.chdir(self.home)
        if 'BitcoinPredictions' not in os.listdir(self.home):
            os.mkdir('BitcoinPredictions')
            os.chdir('BitcoinPredictions')
        self.btc_path = f'{self.home}/BitcoinPredictions'

    def get_data(self):
        #Baixa um csv com os dados de bitcoin dentro de um período de 5 anos
        os.chdir(self.btc_path)
        now = datetime.datetime.now()
        start_date = calendar.timegm((now.year - 5, now.month, now.day, 0, 0, 0))
        end_date = calendar.timegm((now.year, now.month, now.day, 0, 0, 0))
        url = f'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
        filename = 'BTC-USD.csv'
        wget.download(url, filename)

    def pyspark_model_data(self):
        self.data = self.spark.read.csv('BTC-USD.csv', header = True, inferSchema = True)
        self.data = self.data.withColumnRenamed('Close', 'label')
        assembler = VectorAssembler(inputCols = ['Open', 'High', 'Low', 'Volume'], outputCol = 'features')
        processed_data = assembler.transform(self.data)
        train = processed_data.collect()[:-31]
        test = processed_data.collect()[-31:]
        self.train = self.spark.createDataFrame(train)
        self.test = self.spark.createDataFrame(test)

    def gradient_boosting(self):
        self.evaluator = RegressionEvaluator()
        self.model = GBTRegressor()
        self.paramGrid = ParamGridBuilder().addGrid(self.model.maxDepth, [5,10,15,20,25]).build()
        self.cv = CrossValidator(
            estimator = self.model,
            estimatorParamMaps = self.paramGrid,
            evaluator = self.evaluator,
            numFolds = 5)
    
    def train_model(self):
        self.cv_model = self.cv.fit(self.train)
        self.best_model = self.cv_model.bestModel

    def predict(self):
        self.predictions = self.best_model.transform(self.test)

    def evaluate(self):
        self.metrics = {}
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            self.metrics[metric] = RegressionEvaluator(metricName = metric).evaluate(self.predictions)
            print(f'{metric}: {self.metrics[metric]}')

    def plotly_results(self):
        plot_train = self.train.orderBy(self.train['Date'].desc()).limit(30).toPandas().sort_values(by = 'Date')
        plot_test = self.test.toPandas()
        plot_predicted = self.predictions.toPandas()[:-1]
        plot_today_predicted = self.predictions.toPandas()[-2:]
        plot_train = plot_train.append(plot_test.iloc[0], ignore_index = True)
        self.fig1 = px.line(
            plot_train.rename(columns = {'label': 'Treino'}), 
            x = 'Date', 
            y = 'Treino', 
            markers = True, 
            title = 'Valor do Bitcoin (em US$)')
        self.fig1.data[0].line.color = '#0000ff'
        self.fig2 = px.line(
            plot_test.rename(columns = {'label': 'Teste'}), 
            x = 'Date', 
            y = 'Teste', 
            markers = True)
        self.fig2.data[0].line.color = '#ff0000'
        self.fig1 = self.fig1.add_trace(self.fig2.data[0])
        self.fig3 = px.line(
            plot_today_predicted.rename(columns = {'prediction': 'Predição de hoje'}), 
            x = 'Date', 
            y = 'Predição de hoje',
            markers = True)
        self.fig3.data[0].line.color = '#00ff00'
        self.fig1 = self.fig1.add_trace(self.fig3.data[0])
        self.fig4 = px.line(
            plot_predicted.rename(columns = {'prediction': 'Predição de teste'}), 
            x = 'Date', 
            y = 'Predição de teste',
            markers = True)
        self.fig4.data[0].line.color = '#ffa500'
        self.fig1 = self.fig1.add_trace(self.fig4.data[0])
        self.fig1 = self.fig1.update_xaxes(title_text = 'Azul: Treino<br>Vermelho: Teste<br>Laranja: Predição de teste<br>Verde: Predição de hoje')
        self.fig1 = self.fig1.update_yaxes(title_text = 'Valor (em US$)')
        self.fig1 = self.fig1.update_traces(showlegend = True)

    def save_plot(self):
        os.chdir(self.btc_path)
        self.date_path = '{}/{}'.format(self.btc_path, datetime.datetime.now().strftime('%Y-%m-%d'))
        os.mkdir(self.date_path)
        os.chdir(self.date_path)
        self.fig1.write_html('final_plot.html')