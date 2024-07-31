import os
import tensorflow as tf
import numpy as np
import datetime
import voxelmorph as vxm
import distutils.dir_util
import pandas as pd
from tensorflow.keras.models import save_model

from NetworkBasis import config as cfg
import NetworkBasis.util as util

import evaluation

import NetworkBasis.loadsavenii as loadsave
from buildmodel import buildmodel
from datagenerator import DataGenerator, get_test_images
import processing


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

logs_path = os.path.join(cfg.path_result, cfg.experiment_name)

distutils.dir_util.mkpath(logs_path)

def training(f, losses,  loss_weights, learning_rate, nb_epochs, batch_size, seed=42):
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training = True

    traindata_files = pd.read_csv(cfg.train_csv, dtype=object).values

    valdata_files = pd.read_csv(cfg.vald_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)

    vxm_model=buildmodel(inshape)

    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

    vxm_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)

    training_batch_generator = DataGenerator(traindata_files, cfg.path, cfg.nb, inshape)
    validation_batch_generator = DataGenerator(valdata_files, cfg.path, cfg.nb, inshape)

    log_dir = logs_path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #Different callbacks
    callback_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                              restore_best_weights=True)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=logs_path + 'checkpoint',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss', mode='min',
                                                                   save_best_only=True)

    callback = [callback_earlystopping, model_checkpoint_callback, callback_tensorboard]

    validation_steps = cfg.number_of_vald // cfg.nb
    vxm_model.fit(training_batch_generator, validation_data=validation_batch_generator,
                         validation_steps=validation_steps,
                         epochs=nb_epochs, batch_size=batch_size,
                         callbacks=[callback], verbose=2)

    vxm_model.save_weights(logs_path + str(f) + '/weights.h5')
    try:
        save_model(vxm_model, logs_path + str(f) + "/" )
    except:
        print("model save failed")

def apply(f, batch_size,seed=42):
    '''!
    predict images, (segmentations, ) displacementfields for test files
    use shape fixed image
    '''
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training=False
    cfg.output_disp_all=False

    test_data = pd.read_csv(cfg.test_csv, dtype=object).values
    if cfg.seg_available:
        test_data_seg = pd.read_csv(cfg.test_seg_csv, dtype=object).values

    inshape = (cfg.height,cfg.width,cfg.numb_slices)
    vxm_model = buildmodel(inshape)

    print("Test file size: ", len(test_data))
    vxm_model.load_weights(logs_path + str(f) + '/weights.h5')

    predict_path = logs_path+'predict/'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    for i in range(int(len(test_data) / cfg.nb)):
        test_images = get_test_images(test_data, cfg.path, i)
        predictions = vxm_model.predict(test_images, steps=1,
                                        batch_size=batch_size, verbose=2)

        for j in range(cfg.nb):
            filename = test_data[i * cfg.nb + j][0][2:-1]
            cfg.orig_filepath = cfg.path + filename

            if cfg.seg_available:
                filename_seg = test_data_seg[i * cfg.nb + j][0][2:-1]

            loadsave.save_pred_disp([predictions[j*2], predictions[j*2+1]], predict_path, filename)
            processing.warp_img(predict_path, filename)

            if cfg.seg_available:
                processing.warp_seg(predict_path + "/seg/", filename, filename_seg)

def apply_seg_CDH(f, batch_size,seed=42):
    '''!
    predict segmentations for test files
    use shape fixed image
    CDH: lung segmentation available for first image of time series
    '''
    tf.keras.backend.clear_session()
    # inits
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg.training=False
    cfg.output_disp_all=False

    test_data_all = pd.read_csv(cfg.test_csv, dtype=object).values

    patients=[]
    test_data=[]

    for i in range(len(test_data_all)):
        patient = str(test_data_all[i]).rsplit("_", 1)[0][4:]
        if patient not in patients:
            patients.append(patient)

    for i in range(len(patients)):
        for j in range(cfg.nb):
            test_data.append(["b'"+patients[i] + "_" + str(j + 1) + ".nii.gz'"])

    inshape = (cfg.height,cfg.width,cfg.numb_slices)
    vxm_model = buildmodel(inshape)

    print("Test file size: ", len(test_data))
    vxm_model.load_weights(logs_path + str(f) + '/weights.h5')

    predict_path = logs_path+'predict/'
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    for i in range(int(len(test_data) / cfg.nb)):
        test_images = get_test_images(test_data, cfg.path, i)
        predictions = vxm_model.predict(test_images, steps=1,
                                        batch_size=batch_size, verbose=2)

        filename = test_data[i * cfg.nb][0][2:-1]
        cfg.orig_filepath = cfg.path + filename

        loadsave.save_pred_disp([predictions[0], predictions[1]], predict_path, filename)
        processing.warp_img(predict_path, filename)

        processing.warp_seg(predict_path + "/seg/", filename,filename.rsplit("_",1)[0]+".nii.gz")

def evaluate(f):
    '''!
    evaluate predicted images with metrics
    used shape fixed image
    '''

    np.random.seed(42)

    cfg.training = False

    test_data = pd.read_csv(cfg.test_csv, dtype=object).values

    distutils.dir_util.mkpath(logs_path + 'eval/')
    eval_file_path = logs_path + 'eval/' + 'eval-' + str(f) + '.csv'

    header_row = evaluation.make_csv_header()
    util.make_csv_file(eval_file_path, header_row)

    predict_path = logs_path + 'predict/'

    for i in range(len(test_data)//cfg.nb):

        filenames=[]

        for j in range(cfg.nb):
            filenames.append(test_data[i*cfg.nb+j][0][2:-1])

        template_img=loadsave.load_template(filenames)

        for j in range(cfg.nb):
            try:
                result_metrics = {}
                result_metrics['FILENAME_FIXED'] = "Template"
                result_metrics['FILENAME_MOVING'] = filenames[j]


                result_metrics = evaluation.evaluate_prediction_template(result_metrics, predict_path,filenames[j], template_img)
                util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)
                print('Finished Evaluation for ', filenames[j])
            except RuntimeError as err:
                print("    !!! Evaluation of ", filenames[j], ' failed', err)

    # read csv
    header = pd.read_csv(eval_file_path, dtype=object, sep=';')
    header = header.columns.values
    values = pd.read_csv(eval_file_path, dtype=object, sep=';').values
    np_values = np.empty(values.shape)

    result_metrics['FILENAME_FIXED'] = 'min'
    result_metrics['FILENAME_MOVING'] = ' '

    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.min(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'mean'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.average(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

    result_metrics['FILENAME_FIXED'] = 'max'
    result_metrics['FILENAME_MOVING'] = ' '
    for i in range(values.shape[1] - 2):
        for j in range(values.shape[0]):
            np_values[j, i + 2] = float(values[j, i + 2])
        metrics_np = np_values[0:values.shape[0], i + 2]
        try:
            result_metrics[header[i + 2]] = np.max(metrics_np[np.nonzero(metrics_np)])
        except:
            result_metrics[header[i + 2]] = -1
    util.write_metrics_to_csv(eval_file_path, header_row, result_metrics)

def experiment_1(data, losses, loss_weights, learning_rate, nb_epochs,
                 batch_size, is_training, is_apply, is_evaluate):
    k_fold=5
    np.random.seed(42)
    patient_indices= np.random.permutation(range(0, cfg.nb_patients))
    test_folds = np.array_split(patient_indices, k_fold)

    if cfg.seg_available:
        data_seg = loadsave.getdatalist_from_csv(cfg.seg_csv)

    for f in range(k_fold):
        test_patient_indices = test_folds[f]
        remaining_patient_indices = np.random.permutation(np.setdiff1d(patient_indices, test_folds[f]))
        vald_patient_indices = remaining_patient_indices[:cfg.number_of_vald_patients]
        train_patient_indices = remaining_patient_indices[cfg.number_of_vald_patients:]

        cfg.number_of_vald=cfg.number_of_vald_patients*nb_image_pairs_per_patient

        train_indices=[]
        vald_indices=[]
        test_indices=[]

        for i in range(len(train_patient_indices)):
            train_indices.extend(range(train_patient_indices[i]*nb_image_pairs_per_patient, train_patient_indices[i]*nb_image_pairs_per_patient+nb_image_pairs_per_patient))

        for i in range(len(vald_patient_indices)):
            vald_indices.extend(range(vald_patient_indices[i]*nb_image_pairs_per_patient, vald_patient_indices[i]*nb_image_pairs_per_patient+nb_image_pairs_per_patient))

        for i in range(len(test_patient_indices)):
            test_indices.extend(range(test_patient_indices[i]*nb_image_pairs_per_patient, test_patient_indices[i]*nb_image_pairs_per_patient+nb_image_pairs_per_patient))

        train_indices = util.random_permutation_with_groups(train_indices, cfg.nb)
        vald_indices = util.random_permutation_with_groups(vald_indices, cfg.nb)
        test_indices = util.random_permutation_with_groups(test_indices, cfg.nb)[90:]

        train_files=np.empty(len(train_indices), dtype = "S70")
        vald_files=np.empty(len(vald_indices), dtype = "S70")
        test_files=np.empty(len(test_indices), dtype = "S70")

        if cfg.seg_available:
            train_files_seg = np.empty(len(train_indices), dtype="S70")
            vald_files_seg = np.empty(len(vald_indices), dtype="S70")
            test_files_seg = np.empty(len(test_indices), dtype="S70")

        for i in range(len(train_indices)):
            train_files[i] = data[train_indices[i]]
            if cfg.seg_available:
                train_files_seg[i] = data_seg[train_indices[i]]

        for i in range(len(vald_indices)):
            vald_files[i] = data[vald_indices[i]]
            if cfg.seg_available:
                vald_files_seg[i] = data_seg[vald_indices[i]]

        for i in range(len(test_indices)):
            test_files[i] = data[test_indices[i]]
            if cfg.seg_available:
                test_files_seg[i] = data_seg[test_indices[i]]

        np.savetxt(cfg.train_csv, train_files, fmt='%s', header='path')
        np.savetxt(cfg.vald_csv, vald_files, fmt='%s', header='path')
        np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

        if cfg.seg_available:
            np.savetxt(cfg.train_seg_csv, train_files_seg, fmt='%s', header='path')
            np.savetxt(cfg.vald_seg_csv, vald_files_seg, fmt='%s', header='path')
            np.savetxt(cfg.test_seg_csv, test_files_seg, fmt='%s', header='path')

        cfg.num_train_files = len(train_indices)

        print(str(len(train_indices)) + ' train cases, '
                + str(len(test_indices))
                + ' test cases, ' + str(len(vald_indices)) + ' vald cases')

        distutils.dir_util.mkpath(logs_path+'/'+ str(f))

        if is_training:
            training(f, losses, loss_weights, learning_rate, nb_epochs,batch_size, seed=f)

        if is_apply:
            apply(f, batch_size, seed=f)
            if cfg.seg_available:
                apply_seg_CDH(f, batch_size, seed=f)

        if is_evaluate:
            evaluate(f)

        try:
            for f in os.listdir(logs_path + "/predict/displacementfields/"):
                os.remove(os.path.join(logs_path + "/predict/displacementfields/", f))
        except:
            pass

    if is_evaluate:
        evaluation.combine_evaluation_results_from_folds(logs_path+'eval/')
        evaluation.combine_evaluation_results_in_file(logs_path+'eval/')
        evaluation.make_boxplot_graphic(logs_path+'eval/')

#main

is_training = True
is_apply = True
is_evaluate= True

data = loadsave.getdatalist_from_csv(cfg.csv)

cfg.height=192
cfg.width=192
cfg.numb_slices=64

batch_size=1
nb_epochs = 200

cfg.nb = 5 #group size

cfg.seg_available = True

cfg.nb_patients=30
cfg.number_of_vald_patients=3
nb_image_pairs_per_patient=50

#for MI:
bin_centers_mi_glob = np.linspace(0, 1 - 1 / cfg.nb_bins_glob, cfg.nb_bins_glob)
bin_centers_mi_glob = bin_centers_mi_glob + 0.5 * (bin_centers_mi_glob[1] - bin_centers_mi_glob[0])

learning_rate = 1e-4

losses = []
loss_weights = []

for i in range(cfg.nb):
    losses.append(vxm.losses.NMI(bin_centers_mi_glob, (cfg.height, cfg.width, cfg.numb_slices)).loss)
    losses.append(vxm.losses.Grad('l2').loss)
    loss_weights.append(1)
    loss_weights.append(2)

loss_names="NMI_GradL2"

distutils.dir_util.mkpath(logs_path)

experiment_1(data, losses, loss_weights, learning_rate, nb_epochs, batch_size,
             is_training=is_training, is_apply=is_apply, is_evaluate=is_evaluate)



