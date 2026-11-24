#V3 PCA DNN + SNN model 
import numpy as np 
import pandas as pd 

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten

import time
#from functions import *

np.random.seed(111)
tf.random.set_seed(111)


df_save_file_name = "3_PCA_"

# sim loop parameters 
mean_shift_step = 0.1 # how many steps 
mean_shift_range = 2 # from 0.1 to X
no_diff_collumns_steps = 10 # from 5 to 126 diff columns 
repetitions_per_setting = 10 # repetitions per seting 

model_input_dim = 19


################################################################ MODEL

def build_model_DNN(input_shape):        
        model = keras.Sequential([
            Input(shape=(input_shape,)),            
            Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            Dropout(rate=0.25),
            Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            Dense(5, activation='softmax')
        ])


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def build_model_SNN(input_shape):        
        model = keras.Sequential([
            Input(shape=(input_shape,)),            
            Dense(19, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            Dropout(rate=0.25),
            Dense(5, activation='softmax')
        ])


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

############################################################


# Simulation Function V2 (zero mean)
def simulate_data(mean_shift_vec, no_diff, control_sd_vec, class_size = 25, class_no = 5):
    
    no_diff_not = len(control_sd_vec) - no_diff    # number of features without difference
    no_sets = len(mean_shift_vec)

    # simulating control 
    sim_cont = np.random.normal(loc = 0, scale=control_sd_vec, size=(class_size, (no_diff+no_diff_not)))

    #print(f'shape of sim control: {sim_cont.shape}')
    
    # seting column names and creating dataframe
    colnames_diff = []
    for i in range(0, no_diff):
        colnames_diff.append(f'diff_{i}')
    colnames_diff_not = []
    for i in range(0, no_diff_not):
        colnames_diff_not.append(f'no_diff_{i}')

    simulated_df = pd.DataFrame(sim_cont,columns=colnames_diff + colnames_diff_not)
    simulated_df.insert(0, 'class', 0)


    # simulating other classes
    for i in range(0, no_sets):
        sim_diff = np.random.normal(loc = mean_shift_vec[i], scale=control_sd_vec[:no_diff], size=(class_size, no_diff))
        sim_diff_not = np.random.normal(loc = 0, scale=control_sd_vec[no_diff:], size=(class_size, no_diff_not))
        sim_set = np.concatenate((sim_diff, sim_diff_not), axis=1)
        sim_df = pd.DataFrame(sim_set, columns=colnames_diff + colnames_diff_not)
        sim_df.insert(0, 'class', i+1)
        simulated_df = pd.concat([simulated_df, sim_df], ignore_index=True)
        simulated_df['class'] = simulated_df['class'].astype('category')

    sim_predictors = simulated_df.iloc[:, 1:].columns

    
    return simulated_df, sim_predictors

#simulated_df, sim_predictors = simulate_data(mean_shift_vec=mean_shift_vec, no_diff=no_diff, control_sd_vec=cont_sd_vec_p, class_size=25, class_no=5)



cont_sd_vec = np.array([1.15693235, 1.05500268, 1.7416794 , 2.02711463, 1.89803897,
       2.09672919, 2.71297395, 1.70754195, 1.75781168, 1.66783655,
       2.64169587, 2.02930585, 0.99278733, 1.66503435, 1.24323687,
       1.99152683, 1.16981103, 0.70516529, 0.37134141, 0.31048821,
       0.31048821, 0.92605786, 0.70929293, 0.31048821, 1.24366267,
       1.50757082, 0.54502209, 0.31048821, 0.31048821, 1.37363651,
       0.80037693, 1.82446074, 1.16256951, 0.31048821, 1.6548613 ,
       0.31048821, 0.31048821, 1.30088483, 0.60066226, 0.91722705,
       0.47135157, 1.15096605, 2.1050664 , 1.07579632, 0.95622453,
       0.53943466, 0.31048821, 0.31048821, 1.25483198, 0.87208199,
       2.91022276, 0.31048821, 0.37745608, 0.61829634, 1.54480233,
       0.31048821, 0.31048821, 0.31048821, 0.31048821, 1.19239847,
       1.16677074, 0.31048821, 0.31048821, 0.52448312, 0.31048821,
       1.00953577, 0.41309526, 0.31048821, 1.83994088, 0.96428176,
       0.42613804, 0.42613804, 0.40827678, 0.42508987, 0.45608239,
       0.58194415, 1.05762236, 0.31048821, 1.54564574, 0.31048821,
       0.31048821, 1.29064277, 0.73924572, 0.31048821, 0.60021065,
       0.31048821, 0.31048821, 0.31048821, 0.31048821, 0.31048821,
       1.09006886, 0.31048821, 0.31048821, 0.45949104, 0.65011202,
       0.31048821, 0.31048821, 0.31048821, 0.31048821, 0.31048821,
       0.31048821, 0.31048821, 0.31048821, 0.31048821, 0.31048821,
       0.44862142, 0.31048821, 0.31048821, 0.31048821, 0.54222221,
       0.31048821, 0.31048821, 0.83029022, 0.66179732, 0.51085034,
       0.48680427, 0.48680427, 0.45949104, 0.45949104, 0.75914654,
       0.67642638, 0.82997493, 0.63937359, 0.56377073, 0.31048821])


#cont_sd_vec_df = pd.read_csv('cont_sd_vec.csv')
#cont_sd_vec = cont_sd_vec_df.to_numpy().flatten()

cont_sd_vec_p = np.random.permutation(cont_sd_vec)
#print(f"length of control sd vector: {len(cont_sd_vec)}")
#print(f"first 10 values of control sd vector: {cont_sd_vec[:10]}")
# resource estimates
shift_steps = int((mean_shift_range - 0.1) / mean_shift_step) + 1
diff_column_steps = round(120/no_diff_collumns_steps)
total_loops = shift_steps*diff_column_steps*repetitions_per_setting
single_loop_time = 0.15 # in minutes, estimated from test runs
estimated_time = total_loops*single_loop_time/60


print(f"shift steps: {shift_steps}")
print(f"number of diff column steps: {diff_column_steps}")

# model parameters
# early stopping parameters
ES_monitor = 'val_loss'
ES_patience = 20
fit_param_y_name = 'class'




print(f"total loops = {total_loops}")
print(f"estimated time (hrs): {estimated_time}")
results_DNN = []
results_SNN = []



for diff_column_step in np.arange(5, 130, no_diff_collumns_steps):
    no_diff = diff_column_step

    for shift_step in np.arange(0.1, mean_shift_range+0.1, mean_shift_step):
        shift_step = round(shift_step, 4)

        # modify in loop parameters
        mean_shift_vec = [(shift_step*-2),(shift_step*-1),(shift_step*1),(shift_step*2)] 
        #print(f"mean shift vector: {mean_shift_vec}")

        # save result arrays 
        train_accuracy_list_DNN = []
        test_accuracy_list_DNN = []
        train_loss_list_DNN = []
        test_loss_list_DNN = []
        val_accuracy_list_DNN = []
        val_loss_list_DNN = []
        no_epochs_list_DNN = []


        train_accuracy_list_SNN = []
        test_accuracy_list_SNN = []
        train_loss_list_SNN = []
        test_loss_list_SNN = []
        val_accuracy_list_SNN = []
        val_loss_list_SNN = []
        no_epochs_list_SNN = []



        repetition_start = time.time()
        for repetition in range(repetitions_per_setting):
            tf.keras.backend.clear_session()

            
            
            # simulate the data 
            simulated_df, sim_predictors = simulate_data(mean_shift_vec=mean_shift_vec, no_diff=no_diff, control_sd_vec=cont_sd_vec_p, class_size=25, class_no=5)
            #print(f"simulated data shape: {simulated_df.shape}")


            # prepare 
            df_train_val, df_test = train_test_split(simulated_df, test_size=0.2, random_state=111, stratify=simulated_df['class'], shuffle=True)
            X_test = df_test[sim_predictors].values
            y_test = keras.utils.to_categorical(df_test[fit_param_y_name].values)
            X_train = df_train_val[sim_predictors].values
            y_train = keras.utils.to_categorical(df_train_val[fit_param_y_name].values)


            # build autoencoder 
            pca = PCA(n_components=model_input_dim)
            embeddings_train = pca.fit_transform(df_train_val[sim_predictors])
            embeddings_test = pca.transform(df_test[sim_predictors])

            '''
            print(f"embedings train shape: {embeddings_train.shape}")
            print(f"embedings test shape: {embeddings_test.shape}")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"y_test shape: {y_test.shape}")
            '''
            

            # DNN MODEL 
            
            model_DNN = build_model_DNN(model_input_dim)

            ES_DNN = tf.keras.callbacks.EarlyStopping(
                        monitor=ES_monitor,
                        patience=ES_patience,
                        restore_best_weights=True
                    )

            history_DNN = model_DNN.fit(
                x=embeddings_train,
                y=y_train,
                validation_split=0.2,
                epochs=500,
                batch_size=16,
                callbacks=[ES_DNN],
                verbose=0
            )

            best_epoch = np.argmin(history_DNN.history['val_loss']) + 1
            no_epochs_list_DNN.append(best_epoch)
            train_accuracy_list_DNN.append(history_DNN.history['accuracy'][best_epoch-1])
            val_accuracy_list_DNN.append(history_DNN.history['val_accuracy'][best_epoch-1])
            train_loss_list_DNN.append(history_DNN.history['loss'][best_epoch-1])
            val_loss_list_DNN.append(history_DNN.history['val_loss'][best_epoch-1])

            # evaluate model DNN
            test_result = model_DNN.evaluate(x=embeddings_test, y=y_test, verbose=0)            #save results
            test_accuracy_list_DNN.append(test_result[1])
            test_loss_list_DNN.append(test_result[0])

            # SNN MODEL
            model_SNN = build_model_SNN(X_train.shape[1])
            ES_SNN = tf.keras.callbacks.EarlyStopping(
                        monitor=ES_monitor,
                        patience=ES_patience,
                        restore_best_weights=True
                    )
            history_SNN = model_SNN.fit(
                x=X_train,
                y=y_train,
                validation_split=0.2,
                epochs=500,
                batch_size=16,
                callbacks=[ES_SNN],
                verbose=0
            )
            best_epoch_SNN = np.argmin(history_SNN.history['val_loss']) + 1
            no_epochs_list_SNN.append(best_epoch_SNN)
            train_accuracy_list_SNN.append(history_SNN.history['accuracy'][best_epoch_SNN-1])
            val_accuracy_list_SNN.append(history_SNN.history['val_accuracy'][best_epoch_SNN-1])
            train_loss_list_SNN.append(history_SNN.history['loss'][best_epoch_SNN-1])
            val_loss_list_SNN.append(history_SNN.history['val_loss'][best_epoch_SNN-1])

            # evaluate model SNN
            test_result_SNN = model_SNN.evaluate(x=X_test, y=y_test, verbose=0)            #save results
            test_accuracy_list_SNN.append(test_result_SNN[1])
            test_loss_list_SNN.append(test_result_SNN[0])   



            # delete variables to save memory
            del model_DNN
            del model_SNN
            del history_DNN
            del history_SNN
            del simulated_df
            del df_train_val
            del df_test
            del X_train
            del y_train
            del X_test
            del y_test
            del pca
            

        repetition_end = time.time()
        repetition_time = repetition_end - repetition_start
        #save mean results from repetitions DNN
        epochs_mean_DNN = round(np.mean(no_epochs_list_DNN), 4)

        train_accuracy_mean_DNN = round(np.mean(train_accuracy_list_DNN), 4)
        train_acccuracy_std_DNN = round(np.std(train_accuracy_list_DNN), 4)
        train_loss_mean_DNN     = round(np.mean(train_loss_list_DNN), 4)
        train_loss_std_DNN      = round(np.std(train_loss_list_DNN), 4)

        val_accuracy_mean_DNN   = round(np.mean(val_accuracy_list_DNN), 4)
        val_accuracy_std_DNN    = round(np.std(val_accuracy_list_DNN), 4)
        val_loss_mean_DNN       = round(np.mean(val_loss_list_DNN), 4)
        val_loss_std_DNN        = round(np.std(val_loss_list_DNN), 4)

        test_accuracy_mean_DNN  = round(np.mean(test_accuracy_list_DNN), 4)
        test_accuracy_std_DNN   = round(np.std(test_accuracy_list_DNN), 4)
        test_loss_mean_DNN      = round(np.mean(test_loss_list_DNN), 4)
        test_loss_std_DNN       = round(np.std(test_loss_list_DNN), 4)
        

        results_DNN.append([epochs_mean_DNN, repetition_time,
                        diff_column_step, shift_step,
                        train_accuracy_mean_DNN, train_acccuracy_std_DNN,
                        val_accuracy_mean_DNN, val_accuracy_std_DNN, 
                        test_accuracy_mean_DNN, test_accuracy_std_DNN, 
                        train_loss_mean_DNN, train_loss_std_DNN, 
                        val_loss_mean_DNN, val_loss_std_DNN,
                        test_loss_mean_DNN, test_loss_std_DNN])
        

        # SNN
        epochs_mean_SNN = round(np.mean(no_epochs_list_SNN), 4)

        train_accuracy_mean_SNN = round(np.mean(train_accuracy_list_SNN), 4)
        train_acccuracy_std_SNN = round(np.std(train_accuracy_list_SNN), 4)
        train_loss_mean_SNN     = round(np.mean(train_loss_list_SNN), 4)
        train_loss_std_SNN      = round(np.std(train_loss_list_SNN), 4)

        val_accuracy_mean_SNN   = round(np.mean(val_accuracy_list_SNN), 4)
        val_accuracy_std_SNN    = round(np.std(val_accuracy_list_SNN), 4)
        val_loss_mean_SNN       = round(np.mean(val_loss_list_SNN), 4)
        val_loss_std_SNN        = round(np.std(val_loss_list_SNN), 4)

        test_accuracy_mean_SNN  = round(np.mean(test_accuracy_list_SNN), 4)
        test_accuracy_std_SNN   = round(np.std(test_accuracy_list_SNN), 4)
        test_loss_mean_SNN      = round(np.mean(test_loss_list_SNN), 4)
        test_loss_std_SNN       = round(np.std(test_loss_list_SNN), 4)

        results_SNN.append([epochs_mean_SNN, repetition_time,
                        diff_column_step, shift_step,
                        train_accuracy_mean_SNN, train_acccuracy_std_SNN,
                        val_accuracy_mean_SNN, val_accuracy_std_SNN, 
                        test_accuracy_mean_SNN, test_accuracy_std_SNN, 
                        train_loss_mean_SNN, train_loss_std_SNN, 
                        val_loss_mean_SNN, val_loss_std_SNN,
                        test_loss_mean_SNN, test_loss_std_SNN])


        #print(f"diff column step:{diff_column_step},  shift step: {shift_step} - train acc: {train_accuracy_mean}, val acc: {val_accuracy_mean}, test acc: {test_accuracy_mean}, epochs: {epochs_mean}, repetition time (min): {round(repetition_time/60, 4)}")
        # log print
        print("model, mean epochs, repetition time (min), no diff columns, shift step, train accuracy mean, train accuracy std, val accuracy mean, val accuracy std, test accuracy mean, test accuracy std, train loss mean, train loss std, val loss mean, val loss std, test loss mean, test loss std")
        print(f"DNN,{epochs_mean_DNN},{round(repetition_time/60, 4)},{diff_column_step}, {shift_step},{train_accuracy_mean_DNN},{train_acccuracy_std_DNN},{val_accuracy_mean_DNN},{val_accuracy_std_DNN},{test_accuracy_mean_DNN},{test_accuracy_std_DNN},{train_loss_mean_DNN},{train_loss_std_DNN},{val_loss_mean_DNN},{val_loss_std_DNN},{test_loss_mean_DNN},{test_loss_std_DNN}")
        print(f"SNN,{epochs_mean_SNN},{round(repetition_time/60, 4)},{diff_column_step}, {shift_step},{train_accuracy_mean_SNN},{train_acccuracy_std_SNN},{val_accuracy_mean_SNN},{val_accuracy_std_SNN},{test_accuracy_mean_SNN},{test_accuracy_std_SNN},{train_loss_mean_SNN},{train_loss_std_SNN},{val_loss_mean_SNN},{val_loss_std_SNN},{test_loss_mean_SNN},{test_loss_std_SNN}")
        
    
    

# save results from diff column step
results_df_DNN = pd.DataFrame(
    results_DNN,
    columns=[
        "epochs_mean",
        "repetition_time",
        "no_diff_columns",
        "shift_step",
        "train_accuracy",
        "train_accuracy_std",
        "val_accuracy",
        "val_accuracy_std",
        "test_accuracy",
        "test_accuracy_std",
        "train_loss",
        "train_loss_std",
        "val_loss",
        "val_loss_std",
        "test_loss",
        "test_loss_std"
    ]
)
            

results_df_SNN = pd.DataFrame(
    results_SNN,
    columns=[
        "epochs_mean",
        "repetition_time",
        "no_diff_columns",
        "shift_step",
        "train_accuracy",
        "train_accuracy_std",
        "val_accuracy",
        "val_accuracy_std",
        "test_accuracy",
        "test_accuracy_std",
        "train_loss",
        "train_loss_std",
        "val_loss",
        "val_loss_std",
        "test_loss",
        "test_loss_std"
    ]
)
            

results_df_DNN.to_csv(f'{df_save_file_name}_DNN.csv', index=False)
results_df_SNN.to_csv(f'{df_save_file_name}_SNN.csv', index=False)