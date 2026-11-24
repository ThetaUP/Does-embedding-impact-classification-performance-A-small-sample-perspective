import numpy as np 
import pandas as pd 

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

import time
#from functions import *

np.random.seed(111)
tf.random.set_seed(111)


df_save_file_name = "0_SNN"

# sim loop parameters 
mean_shift_step = 0.1 # how many steps 
mean_shift_range = 2 # from 0.1 to X
no_diff_collumns_steps = 10 # from 5 to 126 diff columns 
repetitions_per_setting = 10 # repetitions per seting 

model_input_dim = 125


################################################################ MODEL

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(5, activation ='softmax')
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

'''
# load data from github 
df = pd.read_csv('https://raw.githubusercontent.com/paq88/Fish_neural_network_classifier/refs/heads/main/Gut_analysis/Data/clr_data_bacteria_gut.csv')
print(f"shape:{df.shape}")
print(df.dtypes.value_counts())
df.head()

predictors = df.columns[2:128]
print(f"Number of predictors: {len(predictors)}")
predictors


cont_sd_vec = df.loc[df['pond_5'] == 0, predictors].std().values
'''


cont_sd_vec_df = pd.read_csv('cont_sd_vec.csv')
cont_sd_vec = cont_sd_vec_df.to_numpy().flatten()

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
results = []



for diff_column_step in np.arange(5, 130, no_diff_collumns_steps):
    no_diff = diff_column_step

    for shift_step in np.arange(0.1, mean_shift_range+0.1, mean_shift_step):
        shift_step = round(shift_step, 4)

        # modify in loop parameters
        mean_shift_vec = [(shift_step*-2),(shift_step*-1),(shift_step*1),(shift_step*2)] 
        #print(f"mean shift vector: {mean_shift_vec}")

        # save result arrays 
        train_accuracy_list = []
        test_accuracy_list = []
        train_loss_list = []
        test_loss_list = []
        val_accuracy_list = []
        val_loss_list = []
        no_epochs_list = []
        repetition_start = time.time()
        for repetition in range(repetitions_per_setting):
            
            
            
            # simulate the data 
            simulated_df, sim_predictors = simulate_data(mean_shift_vec=mean_shift_vec, no_diff=no_diff, control_sd_vec=cont_sd_vec_p, class_size=25, class_no=5)
            #print(f"simulated data shape: {simulated_df.shape}")

            # reduce dimensionality if nedded

            # prepare 
            df_train_val, df_test = train_test_split(simulated_df, test_size=0.2, random_state=111, stratify=simulated_df['class'], shuffle=True)
            X_test = df_test[sim_predictors].values
            y_test = keras.utils.to_categorical(df_test[fit_param_y_name].values)
            X_train = df_train_val[sim_predictors].values
            y_train = keras.utils.to_categorical(df_train_val[fit_param_y_name].values)
            

            # build model 
            tf.keras.backend.clear_session()
            model = build_model(model_input_dim)

            ES = tf.keras.callbacks.EarlyStopping(
                        monitor=ES_monitor,
                        patience=ES_patience,
                        restore_best_weights=True
                    )

            history = model.fit(
                x=X_train,
                y=y_train,
                validation_split=0.2,
                epochs=500,
                batch_size=16,
                callbacks=[ES],
                verbose=0
            )
            best_epoch = np.argmin(history.history['val_loss']) + 1
            no_epochs_list.append(best_epoch)
            train_accuracy_list.append(history.history['accuracy'][best_epoch-1])
            val_accuracy_list.append(history.history['val_accuracy'][best_epoch-1])
            train_loss_list.append(history.history['loss'][best_epoch-1])
            val_loss_list.append(history.history['val_loss'][best_epoch-1])
            # evaluate model
            test_result = model.evaluate(x=X_test, y=y_test, verbose=0)            #save results
            test_accuracy_list.append(test_result[1])
            test_loss_list.append(test_result[0])
            # delete variables to save memory
            del model 
            del history
            del simulated_df
            del df_train_val
            del df_test
            del X_train
            del y_train
            del X_test
            del y_test

        repetition_end = time.time()
        repetition_time = repetition_end - repetition_start
        #save mean results from repetitions 
        epochs_mean = round(np.mean(no_epochs_list), 4)

        train_accuracy_mean = round(np.mean(train_accuracy_list), 4)
        train_acccuracy_std = round(np.std(train_accuracy_list), 4)
        train_loss_mean     = round(np.mean(train_loss_list), 4)
        train_loss_std      = round(np.std(train_loss_list), 4)

        val_accuracy_mean   = round(np.mean(val_accuracy_list), 4)
        val_accuracy_std    = round(np.std(val_accuracy_list), 4)
        val_loss_mean       = round(np.mean(val_loss_list), 4)
        val_loss_std        = round(np.std(val_loss_list), 4)

        test_accuracy_mean  = round(np.mean(test_accuracy_list), 4)
        test_accuracy_std   = round(np.std(test_accuracy_list), 4)
        test_loss_mean      = round(np.mean(test_loss_list), 4)
        test_loss_std       = round(np.std(test_loss_list), 4)
        

        results.append([epochs_mean, repetition_time,
                        diff_column_step, shift_step,
                        train_accuracy_mean, train_acccuracy_std,
                        val_accuracy_mean, val_accuracy_std, 
                        test_accuracy_mean, test_accuracy_std, 
                        train_loss_mean, train_loss_std, 
                        val_loss_mean, val_loss_std,
                        test_loss_mean, test_loss_std])
        #print(f"diff column step:{diff_column_step},  shift step: {shift_step} - train acc: {train_accuracy_mean}, val acc: {val_accuracy_mean}, test acc: {test_accuracy_mean}, epochs: {epochs_mean}, repetition time (min): {round(repetition_time/60, 4)}")
        # log print
        print(f"epochs_mean: {epochs_mean}, repetition_time: {round(repetition_time/60, 4)}, no_diff_columns: {diff_column_step}, shift_step: {shift_step}, train_accuracy_mean: {train_accuracy_mean}, train_acccuracy_std: {train_acccuracy_std}, val_accuracy_mean: {val_accuracy_mean}, val_accuracy_std: {val_accuracy_std}, test_accuracy_mean: {test_accuracy_mean}, test_accuracy_std: {test_accuracy_std}, train_loss_mean: {train_loss_mean}, train_loss_std: {train_loss_std}, val_loss_mean: {val_loss_mean}, val_loss_std: {val_loss_std}, test_loss_mean: {test_loss_mean}, test_loss_std: {test_loss_std}")
        # save results from shift step 
        
    
    

# save results from diff column step
results_df = pd.DataFrame(
    results,
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
            

results_df.to_csv(f'Simulation_results/{df_save_file_name}.csv', index=False)