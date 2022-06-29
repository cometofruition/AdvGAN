#importing libraries
import numpy as np
import pandas as pd
import pickle
from blackboxmodel import blackboxdetector
import os
import warnings
import sys
from keras.models import load_model
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input, Activation
from numpy import unique
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from tensorflow.keras.optimizers import Adam
#ignore warning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
    
#class 
class Gan():
        def __init__(self, algo = None):
            #path
            self.input_or_model_path = 'gan/'
            self.output_or_csv_path = './../../GANG/src/modified/'
            self.black_box_model_path = 'blackboxmodel/'
            self.feature_path = 'feature/'

            self.dataset1 = './../../input/feature_vector/dataset1.csv'
            self.dataset2 = './../../input/feature_vector/dataset2.csv'
            self.dataset3 = './../../input/feature_vector/dataset3.csv'
            self.dataset1Mal = './../../input/feature_vector/dataset1_mal.csv'
            self.dataset1Ben = './../../input/feature_vector/dataset1_ben.csv'
            self.dataset2Mal = './../../input/feature_vector/dataset2_mal.csv'
            self.dataset2Ben = './../../input/feature_vector/dataset2_ben.csv'
            self.dataset3Mal = './../../input/feature_vector/dataset3_mal.csv'
            self.dataset3Ben = './../../input/feature_vector/dataset3_ben.csv'

            self.algo = algo           
            self.black_box_type = 0 
            self.csv_writing = 1
            self.noise_dimension = 2434
            self.optimizer = Adam(lr=0.001)
            self.epochs = 10
            self.batch_size = 128
            self.selectEpoch = 1
            self.figNum = 0
            
            #load features 
            with open (self.feature_path +'feature_list', 'rb') as fp:
                self.featurelist_train = pickle.load(fp)
            # load blackbox_detector
            if(self.black_box_type == 0):
                self.blackbox_detector = blackboxdetector
            self.generator = load_model(self.input_or_model_path + 'Generator_model_gan_.h5', compile = False)

            self.discriminator = self.create_discriminator()
            self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

            # The generator takes malware and noise as input and generates adversarial malware examples
            example = Input(shape=(self.noise_dimension,))
            noise = Input(shape=(self.noise_dimension,))
            input = [example, noise]
            malware_examples = self.generator(input)

            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # The discriminator takes generated images as input and determines validity
            validity = self.discriminator(malware_examples)

            # The combined model  (stacked generator and substitute_detector)
            # Trains the generator to fool the discriminator
            self.combined = Model(input, validity)
            self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

            #create folder for csv saving
            if not os.path.exists(self.output_or_csv_path):
                os.makedirs(self.output_or_csv_path)
            
        def create_discriminator(self):
            input = Input(shape=(self.noise_dimension,))
            x = input
            for dim in [50, 1]:
                x = Dense(dim)(x)
            x = Activation(activation='sigmoid')(x)
            discriminator = Model(input, x, name='discriminator')
            return discriminator

        #dataframe to csv
        def to_CsvFormat(self, dataframe1, dataframe2, csvName, index_csv = False):
            result_gan_final_train_df_final = pd.concat([dataframe1, dataframe2], ignore_index=True, axis=0).fillna(0)
            result_gan_final_train_df_final.to_csv(self.output_or_csv_path+ csvName+".csv", index = index_csv)
            
        def choose_dataset(self, datasetNum = 3):
            if datasetNum == 1:
                return self.dataset1
            elif datasetNum == 2:
                return self.dataset2
            elif datasetNum == 3:
                return self.dataset3
            elif datasetNum == 4:
                return self.dataset1Mal
            elif datasetNum == 5:
                return self.dataset1Ben
            elif datasetNum == 6:
                return self.dataset2Mal
            elif datasetNum == 7:
                return self.dataset2Ben
            elif datasetNum == 8:
                return self.dataset3Mal
            else:
                return self.dataset3Ben
                
        def pre_process_data(self, dataframe = None):
            if dataframe is None:
                dataframe = self.dataset3
            dataFrame_Test = pd.read_csv(dataframe) 
            apk_name = dataFrame_Test['apk_name']
            df1 = pd.DataFrame()
            feature_dataset_train = self.featurelist_train
            df_test_category = dataFrame_Test.select_dtypes(exclude =["number"])
            for index_feat in range(len(feature_dataset_train)):
                if(feature_dataset_train[index_feat] in dataFrame_Test):
                    df1[feature_dataset_train[index_feat]] =  dataFrame_Test[feature_dataset_train[index_feat]].values
                else:    
                    df1[feature_dataset_train[index_feat]] = np.zeros((dataFrame_Test.shape[0],), dtype=int)
            dataFrame_Test = df1
            #droping Categorical feature
            df_test = dataFrame_Test.select_dtypes("number")
            if('apk_name' in df_test):
                df_test = df_test.drop(['apk_name'], axis =1)
            if('apkname' in df_test):
                df_test = df_test.drop(['apkname'], axis =1)
            if('SHA256' in df_test):
                df_test = df_test.drop(['SHA256'], axis =1) 
                
            shape_dataSet_test = df_test.shape[1] - 1            
            shape_dataSet_category_test = df_test_category.shape[1] - 1
            df_test_categoryNumpy = df_test_category.values
            if('Class' in df_test ):
                xmal = df_test.drop(['Class'], axis = 1)
                ymal = df_test['Class']
            else:
                xmal = df_test
                ymal = np.ones(df_test.shape[0])
            #extracting feature names
            xmal = xmal.values
            ymal = ymal.values
            return {'xmal': xmal, 'ymal': ymal, 'apk_name': apk_name}

        def test(self, choose = 'blackbox', datasetNum = 3):
            if choose == 'blackbox':
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(datasetNum))
                xmal_test = data_processed["xmal"]
                ymal_test = data_processed["ymal"]
                score = self.blackbox_detector.score(xmal_test, ymal_test, self.algo)
                predict = self.blackbox_detector.predict(xmal_test, self.algo)
                return {'score': score, 'predict': predict}
            else:
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(datasetNum))
                xmal_test = data_processed["xmal"]
                ymal_test = data_processed["ymal"]
                score = self.combined.score(xmal_test, ymal_test)
                predict = self.combined.predict(xmal_test)
                return {'score': score, 'predict': predict}

        def train(self, choose = 'blackbox', datasetNum = 3, retrain = False):
            if choose == 'blackbox':
                if retrain == False:
                    data_processed = self.pre_process_data(dataframe = self.choose_dataset(datasetNum)) 
                    xmal_train = data_processed['xmal']
                    ymal_train = data_processed['ymal']
                    self.blackbox_detector.fit(xmal_train, ymal_train, self.algo)
                else:
                    data_processed = self.pre_process_data(dataframe = self.choose_dataset(4))
                    xmal_train = data_processed["xmal"]
                    ymal_train = data_processed["ymal"]
                    data_processed = self.pre_process_data(dataframe = self.choose_dataset(5))
                    xben_train = data_processed["xmal"]
                    yben_train = data_processed["ymal"]
                    data_processed = self.pre_process_data(dataframe = self.choose_dataset(8))
                    xmal_test = data_processed["xmal"]
                    ymal_test = data_processed["ymal"]
                    #data_processed = self.pre_process_data(dataframe = self.choose_dataset(9))
                    #xben_test = data_processed["xmal"]
                    #yben_test = data_processed["ymal"]

                    # Generate Train Adversarial Examples
                    noise = np.random.normal(0, 1, (xmal_train.shape[0], self.noise_dimension))
                    gen_examples = self.generator.predict([xmal_train, noise])
                    gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.7)
                    self.blackbox_detector.fit(np.concatenate([xmal_train, xben_train, gen_examples]),
                                               np.concatenate([ymal_train, yben_train, ymal_train]), self.algo)

                    # Compute Train TPR
                    train_TPR = self.blackbox_detector.score(gen_examples, ymal_train, self.algo)

                    # Compute Test TPR
                    noise = np.random.normal(0, 1, (xmal_test.shape[0], self.noise_dimension))
                    gen_examples = self.generator.predict([xmal_test, noise])
                    gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.7)
                    test_TPR = self.blackbox_detector.score(gen_examples, ymal_test, self.algo)
                    print('\n---TPR after the black-box detector is retrained(Before Retraining MalGAN).')
                    print('\nTrain_TPR: {0}, Test_TPR: {1}'.format(train_TPR, test_TPR))
            else:
            
                retrain_index = 0
                #train original dataset
                if retrain == False:
                    ymal_train = self.test(choose = 'blackbox', datasetNum = datasetNum)['predict']
                    xmal_train = self.pre_process_data(dataframe = self.choose_dataset(datasetNum))['xmal']
                    self.discriminator.fit(xmal_train, ymal_train, batch_size=self.batch_size, epochs=10)
                    retrain_index = 1

                #train adv dataset
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(4))
                xmal_train = data_processed['xmal']
                ymal_train = data_processed['ymal']
                data_processed1 = self.pre_process_data(dataframe = self.choose_dataset(8))
                xmal_test = data_processed1['xmal']
                ymal_test = data_processed1['ymal']
                apk_name_mal = data_processed['apk_name']
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(7))
                xben_train = data_processed['xmal']
                yben_train = self.blackbox_detector.predict(xben_train, self.algo)
                apk_name_ben = data_processed ['apk_name']
                lowest_TPR = 100

                g_loss_result = []
                d_loss_result = []
                #TPR_result = []
                #TPR_test_result = []
                for epoch in range(self.epochs):
                    print(f'Epoch: {epoch}')
                    xtrain = DataLoader(xmal_train, batch_size=self.batch_size)
                    ytrain = DataLoader(ymal_train, batch_size=self.batch_size)

                    xben_batch = DataLoader(xben_train, batch_size=int(xben_train.shape[0]/(xtrain.dataset.data.shape[0]/self.batch_size)))
                    yben_batch = DataLoader(yben_train, batch_size=int(xben_train.shape[0]/(xtrain.dataset.data.shape[0]/self.batch_size)))     

                    for ((idx, X), (idy, y), (idxb, Xben), (idyb, yben)) in zip(enumerate(xtrain), enumerate(ytrain), enumerate(xben_batch), enumerate(yben_batch)):                    		
                        noise = np.random.normal(0, 1, (X.shape[0], self.noise_dimension))
                        X = np.array(X)
                        gen_examples = self.generator.predict([X, noise])
                        gen_examples = np.ones((gen_examples.shape), dtype=int) *(gen_examples > 0.7)

                        y = self.blackbox_detector.predict(gen_examples, self.algo)
                        # Train the discriminator
                        for i in range(5):
                            d_loss_fake = self.discriminator.train_on_batch(gen_examples, y)
                            d_loss_real = self.discriminator.train_on_batch(Xben, yben)
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                        # Train the generator
                        noise = np.random.normal(0, 1, (X.shape[0], self.noise_dimension))
                        g_loss = self.combined.train_on_batch([X, noise], np.zeros((X.shape[0], 1)))

                        dataFrameIndex_I = idy
                        gen_examples_train_df = pd.DataFrame(gen_examples, columns = self.featurelist_train[1:-1])
                        gen_examples_train_df['Class'] = y
                        if(idy == 0):
                        	result_gan_final_train_df = gen_examples_train_df
                        else:
                        	result_gan_final_train_df = result_gan_final_train_df.append(gen_examples_train_df, sort=False)
                    
                    result_gan_final_train_df['apk_name'] = apk_name_mal
                    apk_name_column = result_gan_final_train_df.pop('apk_name')
                    result_gan_final_train_df.insert(0, 'apk_name', apk_name_column)

                    ben_df = pd.DataFrame(xben_train, columns = self.featurelist_train[1:-1])
                    ben_df['Class'] = yben_train
                    ben_df['apk_name'] = apk_name_ben
                    apk_name_column = ben_df.pop('apk_name')
                    ben_df.insert(0, 'apk_name', apk_name_column)
                    
                    # Compute Test TPR
                    noise = np.random.normal(0, 1, (xmal_test.shape[0], self.noise_dimension))
                    gen_examples = self.generator.predict([xmal_test, noise])
                    TPR_test = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.7), ymal_test, self.algo)
                    print(f'TPR_test: {TPR_test}')

                    # Compute Train TPR
                    noise = np.random.normal(0, 1, (xmal_train.shape[0], self.noise_dimension))
                    gen_examples = self.generator.predict([xmal_train, noise])
                    TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.7), ymal_train, self.algo)
                    print(f'TPR: {TPR}')
                    
                    if retrain_index == 1:
                        if (g_loss < 1 and TPR < lowest_TPR):
                            self.selectEpoch = epoch + 1
                            self.to_CsvFormat(result_gan_final_train_df, ben_df, f"train_modified_feature_vectors_{self.algo}")
                            self.combined.save_weights(f'./saves/{self.algo}/malgan{epoch + 1}.h5')
                            lowest_TPR = TPR
                    else:
                        if (g_loss < 1 and TPR < lowest_TPR):
                            self.selectEpoch = epoch + 1
                            self.to_CsvFormat(result_gan_final_train_df, ben_df, f"train_modified_feature_vectors_{self.algo}_retrain")
                            self.combined.save_weights(f'./saves/{self.algo}_retrain/malgan{epoch + 1}.h5')
                            lowest_TPR = TPR

                    # Plot the progress
                    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                    d_loss_result.append(d_loss[0])
                    g_loss_result.append(g_loss)
                    #TPR_result.append(TPR)
                    #TPR_test_result.append(TPR_test)
                    	
                """plt.rcParams["figure.figsize"] = [7.50, 3.50]
                plt.rcParams["figure.autolayout"] = True

                #x = np.arange(self.epochs)
                #y = np.array(TPR_result)
                x1 = np.arange(self.epochs)
                y1 = np.array(g_loss_result)
                
                plt.figure()
                #plt.title("Line graph")
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                #plt.plot(x, y, color="red", label = 'Training Set')
                plt.plot(x1, y1, color = 'red', label = 'Training Set')
                plt.legend()
                plt.savefig(f'./pictures/{self.algo}_retrain1={retrain}.png')
                #self.figNum = 1"""
             
        def generate_adv_mal(self, retrain_index = False):
                if retrain_index == False:
                    self.combined.load_weights(f'./saves/{self.algo}/malgan{self.selectEpoch}.h5')
                    print(self.selectEpoch)
                else:
                    self.combined.load_weights(f'./saves/{self.algo}_retrain/malgan{self.selectEpoch}.h5')
                    print(self.selectEpoch)
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(4))
                xmal_test = data_processed['xmal']
                ymal_test = data_processed['ymal']
                apk_name_mal = data_processed['apk_name']
                data_processed = self.pre_process_data(dataframe = self.choose_dataset(9))
                xben_test = data_processed['xmal']
                yben_test = data_processed['ymal']
                apk_name_ben = data_processed['apk_name']

                noise = np.random.normal(0, 1, (xmal_test.shape[0], self.noise_dimension))
                gen_examples = self.generator.predict([xmal_test, noise])
                gen_examples = np.ones((gen_examples.shape), dtype=int) *(gen_examples > 0.7)

                """gen_examples_test_df = pd.DataFrame(gen_examples, columns = self.featurelist_train[1:-1])
                gen_examples_test_df['Class'] = '1'
                gen_examples_test_df['apk_name'] = apk_name_mal
                apk_name_column = gen_examples_test_df.pop('apk_name')
                gen_examples_test_df.insert(0, 'apk_name', apk_name_column)

                ben_df = pd.DataFrame(xben_test, columns = self.featurelist_train[1:-1])
                ben_df['Class'] = '0'
                ben_df['apk_name'] = apk_name_ben
                apk_name_column = ben_df.pop('apk_name')
                ben_df.insert(0, 'apk_name', apk_name_column)

                self.to_CsvFormat(gen_examples_test_df, ben_df, f"modified_feature_vectors_{self.algo}_{self.selectEpoch}")"""
                ADR = self.blackbox_detector.score(gen_examples, ymal_test, self.algo)
                return ADR

        	
if __name__ == "__main__":
        #algos = ['RF', 'DT', 'XgBoost', 'LR', 'CNN']
        algos = ['SVM']
        for algo in algos:
            ganObject = Gan(algo = algo)
            print("================ Train blackbox ================")
            #ganObject.train(choose = 'blackbox', datasetNum = 1)
            print(f"+++++++++++++++++++++++++++ {algo} +++++++++++++++++++++++++++")
            ganObject.train(choose = 'GANmodel', datasetNum = 2)
            ODR = ganObject.test(choose = 'blackbox', datasetNum = 8)['score']
            print("================ ODR ================")
            print(ODR)
            ADR = ganObject.generate_adv_mal()
            print("================ ADR ================")
            print(ADR)
            print("================ Retrain blackbox ================")
            ganObject.train(choose = 'blackbox', retrain = True)
            """print("================ Retrain GAN Model ================")
            ganObject.train(choose = 'GANmodel', retrain = True)
            ADR = ganObject.generate_adv_mal(retrain_index = True)
            print("================ ADR after retraining ================")
            print(ADR)"""
