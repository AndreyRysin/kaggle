import pandas as pd
import numpy as np
import pydicom as dcm
import cv2 as cv
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
from tqdm import tqdm
from IPython.display import display as disp


class Detector:
    """
    Main processing class.
    """

    def __init__(
        self,
        path_data,
        path_models,
        first_several_images=-1,  # change only for debugging!
        ignore_errors=False,
    ):
        """
        Init function.
        """
        self.path_data = path_data
        self.path_models = path_models
        self.target_size = (224, 224)
        self.first_several_images = first_several_images
        self.ignore_errors = ignore_errors
        print("v.02")

    def read_filenames(self):
        """
        Forms the table (dataframe) contains all the names:
            | study | folder | image |

        Also, here is defining the image size array.
        """
        names_arr = np.empty((0, 3), dtype="object")
        study_names = listdir_without_ipynb_checkpoints(self.path_data)
        for study in study_names:
            image_folders = listdir_without_ipynb_checkpoints(
                os.path.join(self.path_data, study)
            )
            for folder in image_folders:
                image_names = listdir_without_ipynb_checkpoints(
                    os.path.join(self.path_data, study, folder)
                )
                for image in image_names:
                    names_arr = np.vstack(
                        (
                            names_arr,
                            np.array(
                                (study, folder, image[:-4]), dtype="object"
                            ).reshape(1, -1),
                        )
                    )
        self.df_names = pd.DataFrame(names_arr, columns=["study", "folder", "img"])
        self.image_sizes = np.zeros((names_arr.shape[0], 2), dtype="int")
        self.errors = np.full(names_arr.shape[0], False, dtype="bool")

    def load_pixel_array(self, index):
        """
        Loads the image (pixel array) from a disk.

        index : int
            Index in the table "df_names".
        """
        path = os.path.join(
            self.path_data,
            self.df_names.loc[index, "study"],
            self.df_names.loc[index, "folder"],
            f"{self.df_names.loc[index, 'img']}.dcm",
        )
        if self.ignore_errors:
            try:
                with dcm.dcmread(path) as ds:
                    self.pixel_arr = ds.pixel_array
            except:
                self.pixel_arr = np.random.rand(2500, 2800)
                self.errors[index] = True
        else:
            with dcm.dcmread(path) as ds:
                self.pixel_arr = ds.pixel_array

    def get_image_size(self):
        """
        Returns the size (array shape) of `self.pixel_arr`.

        The array structure:
            [width, height]

        width (xmin, xmax) : self.pixel_arr.shape[1]
        height (ymin, ymax) : self.pixel_arr.shape[0]
        """
        return np.array((self.pixel_arr.shape[1], self.pixel_arr.shape[0]), dtype="int")

    def frags_into_samples(self, frag_dicts):
        """
        Transforms given fragment dicts to the samples X and y ready to be fed
        neural network.
        """
        # X, y init
        X = np.zeros(
            (len(frag_dicts), 1, self.target_size[0], self.target_size[1]),
            dtype="float32",
        )
        y = np.zeros((len(frag_dicts), 4), dtype="float32")
        # Forming the arrays, collecting pixel array means
        for i, frag in enumerate(frag_dicts):
            X[i, 0] = frag["pixel_arr"]
            y[i] = frag["fbox"]
        return X, y

    def bbox_pred_flatten(self, bbox_pred):
        """
        Flattens the bbox predictions (coordinates) from several models into
        a single row.
        """
        bbox_flat = np.zeros(
            (
                bbox_pred.shape[0],
                bbox_pred.shape[1] * bbox_pred.shape[3],
                bbox_pred.shape[2],
            ),
            dtype="float32",
        )
        for i in np.arange(bbox_flat.shape[0]):
            for j in np.arange(bbox_flat.shape[2]):
                bbox_flat[i, :, j] = bbox_pred[i, :, j, :].flatten()
        return bbox_flat

    def compute_predictions_label(self):
        """
        Splits the image on fragments and computing the predictions for
        each one.

        `label_pred` array structure:
            [imgs, models, frags, targets]
        """
        # Init common
        batch_size = 125
        stride = 0.25
        first_several_images = (
            self.first_several_images
            if self.first_several_images > -1
            else self.df_names.shape[0]
        )
        # Init of label
        frag_calibers_label = [2, 3]
        label_pred_arr_size = (first_several_images, 6, 106, 1)
        label_pred = np.zeros(label_pred_arr_size, dtype="float32")
        path_models_label = os.path.join(self.path_models, "01_class")
        # Loop
        for i in tqdm(np.arange(self.df_names[:first_several_images].shape[0])):
            # Init
            self.load_pixel_array(i)
            self.image_sizes[i] = self.get_image_size()
            img_proc = Image_processing(self.pixel_arr, self.target_size)
            # raw image --> samples X, y
            img_proc.compute_fragment_dicts(
                frag_calibers=frag_calibers_label, stride=stride
            )
            X, _ = self.frags_into_samples(img_proc.get_fragment_dicts())
            # samples X, y --> ovr probas
            # neg
            inference = Inference(Model_ovrneg())
            inference.predict(X, batch_size, [78, 87], path_models_label)
            y_pred = inference.get_predictions()
            label_pred[i, np.array((0, 1))] = y_pred.reshape(1, *y_pred.shape)
            # typ
            inference = Inference(Model_ovr1())
            inference.predict(X, batch_size, [120, 143], path_models_label)
            y_pred = inference.get_predictions()
            label_pred[i, np.array((2, 3))] = y_pred.reshape(1, *y_pred.shape)
            # det
            inference = Inference(Model_ovr2())
            inference.predict(X, batch_size, [146], path_models_label)
            y_pred = inference.get_predictions()
            label_pred[i, np.array((4,))] = y_pred.reshape(1, *y_pred.shape)
            # atp
            inference = Inference(Model_ovr3())
            inference.predict(X, batch_size, [149], path_models_label)
            y_pred = inference.get_predictions()
            label_pred[i, np.array((5,))] = y_pred.reshape(1, *y_pred.shape)
        # Return
        return label_pred

    def define_label(self, label_fl_proba):
        """
        Defines the label using the fragment-level predictions.
        """
        # Init
        batch_size = 200
        path_models = os.path.join(self.path_models, "03_class_defining")
        label_fl = np.squeeze(label_fl_proba)  # result: [imgs, models, frags]
        X = label_fl.reshape(label_fl.shape[0], 1, label_fl.shape[1], label_fl.shape[2])
        # Computing
        inference = Inference(Model_label_defining())
        inference.predict(X, batch_size, [213, 214, 215, 216, 217, 218], path_models)
        label_pred = inference.get_predictions()
        label_pred = np.mean(label_pred, axis=0)
        # Return
        return label_pred

    def compute_predictions_bbox(self, label_il):
        """
        Splits the image on fragments and computing the predictions for
        each one.

        `bbox_pred` array structure:
            [imgs, models, frags, targets]
        """
        # Init common
        batch_size = 125
        stride = 0.25
        first_several_images = (
            self.first_several_images
            if self.first_several_images > -1
            else self.df_names.shape[0]
        )
        # Init of bbox
        frag_calibers_bbox = [1, 2]
        bbox_pred_arr_size = (first_several_images, 5, 27, 4)
        bbox_pred = np.zeros(bbox_pred_arr_size, dtype="float32")
        fbox_true = np.zeros(bbox_pred_arr_size, dtype="float32")
        neg_dummy = np.zeros(bbox_pred_arr_size[1:], dtype="float32")
        neg_dummy[:, :, 2:] = 1.0
        path_models_bbox = os.path.join(self.path_models, "02_bbox")
        # Loop
        for i in tqdm(np.arange(self.df_names[:first_several_images].shape[0])):
            # Init
            if label_il[i] == 0:
                bbox_pred[i] = neg_dummy
                fbox_true[i] = neg_dummy
            else:
                self.load_pixel_array(i)
                img_proc = Image_processing(self.pixel_arr, self.target_size)
                # raw image --> samples X, y
                img_proc.compute_fragment_dicts(
                    frag_calibers=frag_calibers_bbox, stride=stride
                )
                X, fbox = self.frags_into_samples(img_proc.get_fragment_dicts())
                # samples X, y --> bbox coordinates
                inference = Inference(Model_bbox())
                inference.predict(
                    X, batch_size, [134, 136, 151, 152, 153], path_models_bbox
                )
                y_pred = inference.get_predictions()
                bbox_pred[i] = y_pred
                fbox_true[i] = fbox
        # Return
        return bbox_pred, fbox_true

    def define_bbox_amount(self, bbox_fl, fbox_fl):
        """
        Defines the amount of bounding boxes within an image using the
        fragment-level predictions.
        """
        # Init
        batch_size = 200
        path_models = os.path.join(self.path_models, "04_bbox_amount_defining")
        X = np.append(
            self.bbox_pred_flatten(bbox_fl),
            np.swapaxes(fbox_fl[:, 0, :, :], 1, 2),
            axis=1,
        )  # result: [imgs, models, frags]
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        # Computing
        inference = Inference(Model_bbox_amount_defining())
        inference.predict(X, batch_size, [189, 190, 191, 192, 193], path_models)
        bbox_amount_pred = inference.get_predictions()
        bbox_amount_pred = np.argmax(np.mean(bbox_amount_pred, axis=0), axis=1)
        # Return
        return bbox_amount_pred

    def bbox_centres(self, bbox_fl, fbox_fl):
        """
        Finds relative (within the fragment) and absolute (within the whole
        image) centres of the bounding boxes.

        Input arrays shape:
            [imgs, models, frags, bbox_coordinates]
            (e.g.: [5881, 5, 108, 4])

        Output arrays shape is the same as of the input except the dimension 3
        that is of 2: x and y coordinates of the centre.
            [imgs, models, frags, centre_coordinates]
            (e.g.: [5881, 5, 108, 2])
        """
        # Init
        centres = np.zeros((*bbox_fl.shape[:3], 2), dtype="float32")
        centres_abs = np.zeros(centres.shape, dtype="float32")
        # Computing
        centres[:, :, :, 0] = np.mean(bbox_fl[:, :, :, np.array((0, 2))], axis=3)
        centres[:, :, :, 1] = np.mean(bbox_fl[:, :, :, np.array((1, 3))], axis=3)
        centres_abs[:, :, :, 0] = fbox_fl[:, :, :, 0] + centres[:, :, :, 0] * (
            fbox_fl[:, :, :, 2] - fbox_fl[:, :, :, 0]
        )
        centres_abs[:, :, :, 1] = fbox_fl[:, :, :, 1] + centres[:, :, :, 1] * (
            fbox_fl[:, :, :, 3] - fbox_fl[:, :, :, 1]
        )
        # Return
        return centres, centres_abs

    def clusterization(self, bbox_fl, fbox_fl, bbox_amount):
        """
        Defines what bounding box the fragments are related to;
        rearranges the arrays for being fed the model.
        """
        # Finding centers of the bounding boxes
        centres, centres_abs = self.bbox_centres(bbox_fl, fbox_fl)
        # Splitting indices by `bbox_amount`
        bbox_amount = pd.Series(bbox_amount)
        index_am1 = bbox_amount[bbox_amount == 1].index.values
        index_am2 = bbox_amount[bbox_amount > 1].index.values
        # Clusterization
        X = np.swapaxes(self.bbox_pred_flatten(centres_abs), 1, 2)[index_am2]
        clust = KMeans(n_clusters=2)
        cluster_labels = np.zeros(X.shape[:2], dtype="int")
        cluster_centres = np.zeros((X.shape[0], 2, 2), dtype="float32")
        for i in tqdm(np.arange(X.shape[0])):
            cluster_labels[i] = clust.fit_predict(X[i])
            cluster_centres[i] = np.mean(
                clust.cluster_centers_.reshape(2, 5, 2), axis=1
            )
        # Rearranging the arrays
        X = np.append(
            self.bbox_pred_flatten(bbox_fl),
            np.swapaxes(fbox_fl[:, 0, :, :], 1, 2),
            axis=1,
        )  # result: [imgs, models, frags]
        X_am1 = np.swapaxes(X[index_am1], 1, 2)
        X_am2 = np.swapaxes(X[index_am2], 1, 2)
        X_am2_label0 = np.zeros(X_am2.shape, dtype="float32")
        X_am2_label1 = np.zeros(X_am2.shape, dtype="float32")
        for i in np.arange(X_am2_label0.shape[0]):
            X_am2_label0[i] = X_am2[i].copy()
            X_am2_label0[i][cluster_labels[i] == 1] = 0
        for i in np.arange(X_am2_label1.shape[0]):
            X_am2_label1[i] = X_am2[i].copy()
            X_am2_label1[i][cluster_labels[i] == 0] = 0
        X = np.vstack((X_am1, X_am2_label0, X_am2_label1))
        X = X.reshape(X.shape[0], 1, X.shape[2], X.shape[1])
        # Return
        return X

    def define_bbox(self, bbox_fl_clustered):
        """
        Defines the bounding boxes using the fragment-level predictions.
        """
        # Init
        batch_size = 200
        path_models = os.path.join(self.path_models, "05_bbox_defining")
        X_flat = np.zeros(
            (
                bbox_fl_clustered.shape[0],
                bbox_fl_clustered.shape[2] * bbox_fl_clustered.shape[3],
            ),
            dtype="float32",
        )
        for i in np.arange(X_flat.shape[0]):
            X_flat[i] = bbox_fl_clustered[i, 0].flatten()
        # Computing
        inference = Inference(Model_bbox_defining())
        inference.predict(
            X_flat,
            batch_size,
            [234, 236, 244, 245, 247, 252, 254, 255, 256, 258],
            path_models,
        )
        bbox_pred = inference.get_predictions()
        bbox_pred = np.mean(bbox_pred, axis=0)
        # Return
        return bbox_pred

    def form_predictions_array(self, label_il, label_il_proba, bbox_il, bbox_amount):
        """
        Forms and returns the image-wise array of the predictions.
        One array item <==> one image (not study!)

        Each array item contains the dictionary of the structure as follows:
            predictions[i] = {
                "label": `label number`,
                "conf": `label confidence (proba)`,
                "bbox": `np.array of the bounding boxes`,
            }

        `label number` : int
            Can be of range of 4 according to the following dict:
            labels = {0: "neg", 1: "typ", 2: "det", 3: "atp"}

        `label confidence (proba)` : float
            The probability of class (i.e. of the label).

        `np.array of the bounding boxes` : numpy.array of floats
            Array of shape (n_bboxes_in_image, 4).
        """
        # Init
        predictions = np.empty(self.df_names.shape[0], dtype="object")
        for i in np.arange(predictions.shape[0]):
            predictions[i] = {"label": None, "conf": None, "bbox": None}
        # Splitting indices by `bbox_amount`
        bbox_amount = pd.Series(bbox_amount)
        index_am0 = bbox_amount[bbox_amount == 0].index.values
        index_am1 = bbox_amount[bbox_amount == 1].index.values
        index_am2 = bbox_amount[bbox_amount > 1].index.values
        # Unstacking the predicted bounding boxes array
        bbox_il_am1 = bbox_il[: index_am1.shape[0]]
        bbox_il_am2_label0 = bbox_il[
            index_am1.shape[0] : index_am1.shape[0] + index_am2.shape[0]
        ]
        bbox_il_am2_label1 = bbox_il[index_am1.shape[0] + index_am2.shape[0] :]
        # Forming the resulting array
        # no opacities
        for idx in index_am0:
            predictions[idx]["label"] = 0
            predictions[idx]["conf"] = 1.0
            predictions[idx]["bbox"] = np.array(((0, 0, 1, 1),))
        # one opacity
        for i, idx in enumerate(index_am1):
            bbox_arr = np.zeros((1, 4), dtype="float32")
            bbox_arr[0] = bbox_il_am1[i]
            predictions[idx]["label"] = label_il[idx]
            predictions[idx]["conf"] = (
                label_il_proba[idx, label_il[idx]]
                if label_il_proba[idx, label_il[idx]] > 0.51
                else 0.51
            )
            predictions[idx]["bbox"] = bbox_arr
        # two opacities
        for i, idx in enumerate(index_am2):
            bbox_arr = np.zeros((2, 4), dtype="float32")
            bbox_arr[0] = bbox_il_am2_label0[i]
            bbox_arr[1] = bbox_il_am2_label1[i]
            predictions[idx]["label"] = label_il[idx]
            predictions[idx]["conf"] = (
                label_il_proba[idx, label_il[idx]]
                if label_il_proba[idx, label_il[idx]] > 0.51
                else 0.51
            )
            predictions[idx]["bbox"] = bbox_arr
        # if error occurs while reading the pixel array (this is a dummy)
        for i in np.arange(self.df_names.shape[0]):
            if self.errors[i]:
                predictions[idx]["label"] = 0
                predictions[idx]["conf"] = 1.0
                predictions[idx]["bbox"] = np.array(((0, 0, 1, 1),))
        # Return
        return predictions

    def predictions_computing_pipeline(self):
        """
        Computes the predictions and returns the list of the results.
        """
        # Init
        self.read_filenames()
        # Label: fragment-level predictions
        label_fl_proba = self.compute_predictions_label()
        # Label: image-level predictions
        label_il_proba = self.define_label(label_fl_proba)
        label_il = np.argmax(label_il_proba, axis=1)
        # Bbox: fragment-level predictions
        bbox_fl, fbox_fl = self.compute_predictions_bbox(label_il)
        bbox_amount = self.define_bbox_amount(bbox_fl, fbox_fl)
        bbox_fl_clustered = self.clusterization(bbox_fl, fbox_fl, bbox_amount)
        # Bbox: image-level predictions
        bbox_il = self.define_bbox(bbox_fl_clustered)
        # Forming the predictions list
        self.predictions = self.form_predictions_array(
            label_il, label_il_proba, bbox_il, bbox_amount
        )

    def form_csv(self):
        """
        Forms the csv-file with predictions according to the format
        requirements.
        """
        # Init
        first_several_images = (
            self.first_several_images
            if self.first_several_images > -1
            else self.df_names.shape[0]
        )
        index_study = (
            self.df_names[:first_several_images]
            .drop_duplicates(subset="study")
            .index.values
        )
        study_level_labels = {
            0: "negative",
            1: "typical",
            2: "indeterminate",
            3: "atypical",
        }
        image_level_labels = {0: "none", 1: "opacity", 2: "opacity", 3: "opacity"}
        results_arr = np.empty((0, 2), dtype="object")
        # Study level
        for idx in index_study:
            study_sample = self.df_names[
                self.df_names["study"] == self.df_names.loc[idx, "study"]
            ]
            std_lvl_pred = ""
            for j in study_sample.index:
                std_lvl_pred += "{} {} 0 0 1 1 ".format(
                    study_level_labels[self.predictions[j]["label"]],
                    "1"
                    if self.predictions[j]["label"] == 0
                    else self.predictions[j]["conf"],
                )
            std_lvl_pred = str.strip(std_lvl_pred)
            std_lvl_id = "{}_study".format(self.df_names.loc[idx, "study"])
            results_arr = np.append(
                results_arr,
                np.array(((std_lvl_id, std_lvl_pred),), dtype="object"),
                axis=0,
            )
        # Image level
        for i in np.arange(self.df_names[:first_several_images].shape[0]):
            img_lvl_pred = ""
            for j in np.arange(self.predictions[i]["bbox"].shape[0]):
                bbox = self.predictions[i]["bbox"][j].copy()
                if self.predictions[i]["label"] > 0:
                    for k in np.arange(4):
                        bbox[k] = np.around(
                            self.predictions[i]["bbox"][j, k]
                            * self.image_sizes[i, k % 2],
                            0,
                        )
                    bbox = bbox.astype("int")
                img_lvl_pred += "{} {} {} {} {} {} ".format(
                    image_level_labels[self.predictions[i]["label"]],
                    "1"
                    if self.predictions[i]["label"] == 0
                    else self.predictions[i]["conf"],
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                )
            img_lvl_pred = str.strip(img_lvl_pred)
            img_lvl_id = "{}_image".format(self.df_names.loc[i, "img"])
            results_arr = np.append(
                results_arr,
                np.array(((img_lvl_id, img_lvl_pred),), dtype="object"),
                axis=0,
            )
        # Forming the dataframe
        self.df_results = pd.DataFrame(results_arr, columns=["id", "PredictionString"])

    def get_csv(self):
        """
        Returns the csv-file with predictions.
        """
        return self.df_results

    def save_csv(self, path):
        """
        Saves the csv-file with predictions on a disk.
        """
        self.df_results.to_csv(os.path.join(path, "submission.csv"), index=False)


class Image_processing:
    """
    Implementation of image processing.
    Generally, it does splitting and resizing.
    """

    def __init__(self, img_orig_pixel_array, target_size):
        """
        - img_orig_pixel_array
            raw image array (ds.pixel_array) from dcm file;
        - target_size -> tuple (height, width)
            the size that image is transformed into to be fit for the neural
            network input.
        """
        # Checking up
        assert len(target_size) == 2
        assert len(target_size) == img_orig_pixel_array.ndim
        # image general vars
        self.img_orig = img_orig_pixel_array.copy()
        self.target_size = target_size

    def compute_fragment_dicts(self, frag_calibers, stride):
        """
        Implementation of the fragments generation pipeline for inference.
        """
        # scaling the pixels into relative values (of the range [0;1])
        self.scaling_img_orig()
        # generating the fragments
        frags = self.fragment_generator(
            frag_calibers=frag_calibers,
            stride=stride,
        )
        # forming the list of fragment dicts (also, normalizing and resizing)
        self.frag_dicts = []
        for i in np.arange(frags.shape[0]):
            frag_dict = self.fragment_extraction(frags[i])
            frag_dict["pixel_arr"] = self.resize_pixel_arr(
                self.normalize_pixel_arr(frag_dict["pixel_arr"])
            )
            self.frag_dicts.append(frag_dict)

    def scaling_img_orig(self):
        """
        Scaling pixel levels.
        """
        self.img = (self.img_orig / self.img_orig.max()).astype("float32")

    def fragment_generator(self, frag_calibers, stride):
        """
        Returns coordinates of the fragments for inference.
        """
        frags_inference = np.zeros((0, 4), dtype="float32")
        for caliber in frag_calibers:
            if caliber == 1:
                frags_inference = np.vstack(
                    (frags_inference, np.array((0.0, 0.02, 0.6, 0.98)))
                )
                frags_inference = np.vstack(
                    (frags_inference, np.array((0.4, 0.02, 1.0, 0.98)))
                )
            else:
                size = 1 / caliber
                for i in np.arange(int((1 - size) / (size * stride) + 1)):
                    xmin = i * size * stride
                    for j in np.arange(int((1 - size) / (size * stride) + 1)):
                        ymin = j * size * stride
                        frags_inference = np.vstack(
                            (
                                frags_inference,
                                np.array((xmin, ymin, xmin + size, ymin + size)),
                            )
                        )
        return frags_inference

    def fragment_extraction(self, fragment):
        """
        Extracts the fragment and returns the dict.
        One dict <==> one fragment.

        The structure of the dict:
        'pixel_arr': pixel array of the fragment (np.array([]) of size of the fragment);
        'fbox': array of the fragment coordinates within the image (np.array([xmin, ymin, xmax, ymax,])).
        """
        # init
        frag_dict = {}
        xmin_frag = fragment[0]
        ymin_frag = fragment[1]
        xmax_frag = fragment[2]
        ymax_frag = fragment[3]
        # pixel_arr
        frag_dict["pixel_arr"] = self.img[
            int(np.around(ymin_frag * self.img.shape[0], 0)) : int(
                np.around(ymax_frag * self.img.shape[0], 0)
            ),
            int(np.around(xmin_frag * self.img.shape[1], 0)) : int(
                np.around(xmax_frag * self.img.shape[1], 0)
            ),
        ]
        # fbox
        frag_dict["fbox"] = np.array((xmin_frag, ymin_frag, xmax_frag, ymax_frag))
        # return
        return frag_dict

    def resize_pixel_arr(self, pixel_arr):
        """
        Resizes the image (pixel_arr) to the target size (target_size).
        If the size of the input pixel array and the target size are the same
        resizing is unnecessary, and the input pixel array is returned.
        """
        if pixel_arr.shape == self.target_size:
            return pixel_arr
        else:
            # Choosing the interpolation method
            interpolation_method = cv.INTER_CUBIC
            if (pixel_arr.shape[0] > self.target_size[0]) & (
                pixel_arr.shape[1] > self.target_size[1]
            ):
                interpolation_method = cv.INTER_AREA
            # Resizing
            resized_pixel_arr = cv.resize(
                pixel_arr,
                self.target_size,
                interpolation=interpolation_method,
            )
            # Return
            return resized_pixel_arr

    def normalize_pixel_arr(self, pixel_arr):
        """
        Normalizes the image (pixel_arr).
        """
        arr_min = np.min(pixel_arr)
        arr_max = np.max(pixel_arr)

        if arr_max > arr_min:
            return (pixel_arr - arr_min) / (arr_max - arr_min)
        else:
            return pixel_arr

    def get_fragment_dicts(self):
        """
        Returns the list of the fragment dicts.
        """
        return self.frag_dicts


class Inference:
    """
    Performs neural network working.
    """

    def __init__(self, model):
        """
        - model
            pytorch model, pretrained or not.
        """
        self.cuda_number = torch.cuda.current_device()
        self.model = model.float().cuda(self.cuda_number)
        self.model.eval()

    def epoch_func(self, loader):
        """
        Implementation of the training and the validation within one epoch.
        """
        y_pred = torch.FloatTensor([])
        for batch in loader:
            y_pred_batch = self.batch_func(batch)
            y_pred = torch.cat((y_pred, y_pred_batch), dim=0)
        self.predictions_epoch = y_pred

    def batch_func(self, batch):
        """
        Implementation of the training and the validation within one batch.
        """
        X_batch = batch[0].cuda(self.cuda_number)
        y_pred_batch = self.model(X_batch)
        return y_pred_batch.detach().cpu()

    def torch_dataloader(self, X_input):
        """
        Initialization of the torch data loaders (TensorDataset --> DataLoader).
        """
        X = torch.FloatTensor(self.transform_input_X(X_input))
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        return X, loader

    def clear_memory(self):
        """
        Clears the RAM and the CUDA cache.
        """
        try:
            del self.loader
            del self.X
        except:
            None
        torch.cuda.empty_cache()

    def load_and_assign_from_folder(self, number_of_model, path_models_selected):
        """
        Loads from a disk and assigns the model state dict that was copied to
        an independent folder "path_models_selected".
        """
        number = "{:03d}".format(number_of_model)
        path = os.path.join(path_models_selected)
        if os.path.exists(path):
            try:
                self.model.load_state_dict(
                    torch.load(os.path.join(path, f"{number}_MODEL.pt"))
                )
            except:
                print("No MODEL")
        else:
            print("The folder is not existing")

    def predict(
        self,
        X,
        batch_size,
        model_numbers_list,
        path_models_selected,
    ):
        """
        Implementation of prediction with the trained model.
        """
        # Init
        self.batch_size = batch_size
        # Inference pass
        self.X, self.loader = self.torch_dataloader(X)
        try:
            self.full_inference_pass(
                numbers=model_numbers_list,
                path_models_selected=path_models_selected,
            )
        except KeyboardInterrupt:
            None
        self.clear_memory()

    def full_inference_pass(self, numbers, path_models_selected=None):
        """
        Full accomplished inference pass.

        - numbers
            the numbers of models that were copied priorly and are kept in a
            specified independent folder "path_models_selected".
        """
        # Iterating over the state dicts (loading, assigning, computing, collecting)
        predictions_chunk_list = []
        for i, number in enumerate(numbers):
            self.load_and_assign_from_folder(number, path_models_selected)
            self.epoch_func(self.loader)
            predictions_chunk_list.append(self.get_y(self.predictions_epoch))
        # Forming the predictions chunk array (one epoch, a bunch of models)
        predictions_chunk = np.zeros(
            (len(numbers), *predictions_chunk_list[0].shape), dtype="float32"
        )
        for i in np.arange(len(predictions_chunk_list)):
            predictions_chunk[i] = predictions_chunk_list[i]
        # Appending chunk predictions to the general array
        try:
            self.predictions = np.vstack((self.predictions, predictions_chunk))
        except:
            self.predictions = predictions_chunk

    def get_predictions(self):
        """
        Returns the array of the last predictions (y_pred).
        If predictions over several models the model-level dimension is 0.
        """
        return self.predictions

    def get_y(self, y, mode="array"):
        """
        Gets the array of the predictions directly from the epoch function and
        returns it in the proper format.

        The array has to be computed priorly using the "predict" method.
        The array can be output as "numpy.ndarray" or as "torch.tensor"
        according to "mode".

        Possible values of "mode":
        "array" -> numpy.array
        "tensor" -> torch.tensor
        """
        try:
            if mode == "array":
                return y.detach().cpu().numpy()
            elif mode == "tensor":
                return y.detach()
            else:
                print('Choose correct "mode".')
        except:
            print('Array to output doesn\'t exist. Run "predict" first.')

    def transform_input_X(self, X_input):
        """
        Transforms the input feature array (X) just before turning into
        a tensor.

        By default, returns the input as is, and can be overridden if
        necessary.

        X_input : numpy.array
            Raw input feature array in the state it is loaded.

        return : numpy.array
            The feature array is in the final state that is ready to be turned
            into a tensor.
        """
        # Subtraction of the global dataset mean, that is of 0.527877857
        return X_input - 0.527877857

    def transform_input_y(self, y_input):
        """
        Transforms the input target array (y) just before turning into
        a tensor.

        By default, returns the input as is, and can be overridden if
        necessary.

        y_input : numpy.array
            Raw input target array in the state it is loaded.

        return : numpy.array
            The target array is in the final state that is ready to be turned
            into a tensor.
        """
        return y_input


def listdir_without_ipynb_checkpoints(path):
    """
    Implements "os.listdir()" method with two features:
    - ".ipynb_checkpoints" item is removed;
    - the list is sorted.

    path : string
        Full path to the files whose list is required.

    return : list of strings
        List of the file names.
    """
    filenames = sorted(os.listdir(path))
    i = 0
    while i < len(filenames):
        if filenames[i] == ".ipynb_checkpoints":
            del filenames[i]
        else:
            i += 1
    return filenames


def Model_ovrneg():
    """
    Hard-fixed structure of the model.
    """
    params_conv_sequential = [
        {
            "ci": 1,
            "co": 40,
            "ck": 5,
            "cs": 2,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "avg",
            "pk": 4,
            "ps": 4,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 40,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "avg",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "avg",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 60,
            "co": 90,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_lin_sequential = [
        {"dp": 0.4, "li": 810, "lo": 256, "at": "sigm"},
        {"dp": 0.0, "li": 256, "lo": 1, "at": "sigm"},
    ]
    model = nn.Sequential(
        Sequential_Modules(Conv2d_Module, params_conv_sequential),
        nn.Flatten(),
        Sequential_Modules(Linear_Module, params_lin_sequential),
    )
    return model


def Model_ovr1():
    """
    Hard-fixed structure of the model.
    """
    params_conv_parallel_0 = [
        {
            "ci": 1,
            "co": 5,
            "ck": 1,
            "cs": 1,
            "cp": 0,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 9,
            "cs": 1,
            "cp": 4,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 13,
            "cs": 1,
            "cp": 6,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_parallel_1 = [
        {
            "ci": 20,
            "co": 10,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 7,
            "cs": 1,
            "cp": 3,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_sequential = [
        {
            "ci": 30,
            "co": 30,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 30,
            "co": 30,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 30,
            "co": 40,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 1,
            "pd": 1,
        },
        {
            "ci": 60,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_lin_sequential = [
        {"dp": 0.4, "li": 240, "lo": 256, "at": "sigm"},
        {"dp": 0.4, "li": 256, "lo": 64, "at": "sigm"},
        {"dp": 0.4, "li": 64, "lo": 1, "at": "sigm"},
    ]
    model = nn.Sequential(
        Parallel_Modules(Conv2d_Module, params_conv_parallel_0),
        Parallel_Modules(Conv2d_Module, params_conv_parallel_1),
        Sequential_Modules(Conv2d_Module, params_conv_sequential),
        nn.Flatten(),
        Sequential_Modules(Linear_Module, params_lin_sequential),
    )
    return model


def Model_ovr2():
    """
    Hard-fixed structure of the model.
    """
    params_conv_parallel_0 = [
        {
            "ci": 1,
            "co": 5,
            "ck": 1,
            "cs": 1,
            "cp": 0,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 9,
            "cs": 1,
            "cp": 4,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 13,
            "cs": 1,
            "cp": 6,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_parallel_1 = [
        {
            "ci": 20,
            "co": 10,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 7,
            "cs": 1,
            "cp": 3,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_sequential = [
        {
            "ci": 30,
            "co": 40,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 60,
            "co": 80,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 80,
            "co": 120,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 1,
            "pd": 1,
        },
        {
            "ci": 120,
            "co": 150,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_lin_sequential = [
        {"dp": 0.4, "li": 600, "lo": 256, "at": "sigm"},
        {"dp": 0.4, "li": 256, "lo": 64, "at": "sigm"},
        {"dp": 0.4, "li": 64, "lo": 1, "at": "sigm"},
    ]
    model = nn.Sequential(
        Parallel_Modules(Conv2d_Module, params_conv_parallel_0),
        Parallel_Modules(Conv2d_Module, params_conv_parallel_1),
        Sequential_Modules(Conv2d_Module, params_conv_sequential),
        nn.Flatten(),
        Sequential_Modules(Linear_Module, params_lin_sequential),
    )
    return model


def Model_ovr3():
    """
    Hard-fixed structure of the model.
    """
    params_conv_parallel_0 = [
        {
            "ci": 1,
            "co": 5,
            "ck": 1,
            "cs": 1,
            "cp": 0,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 9,
            "cs": 1,
            "cp": 4,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 13,
            "cs": 1,
            "cp": 6,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_parallel_1 = [
        {
            "ci": 20,
            "co": 10,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 20,
            "co": 10,
            "ck": 7,
            "cs": 1,
            "cp": 3,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_sequential = [
        {
            "ci": 30,
            "co": 40,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 60,
            "co": 80,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 80,
            "co": 120,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 1,
            "pd": 1,
        },
        {
            "ci": 120,
            "co": 150,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_lin_sequential = [
        {"dp": 0.4, "li": 600, "lo": 256, "at": "sigm"},
        {"dp": 0.4, "li": 256, "lo": 64, "at": "sigm"},
        {"dp": 0.4, "li": 64, "lo": 1, "at": "sigm"},
    ]
    model = nn.Sequential(
        Parallel_Modules(Conv2d_Module, params_conv_parallel_0),
        Parallel_Modules(Conv2d_Module, params_conv_parallel_1),
        Sequential_Modules(Conv2d_Module, params_conv_sequential),
        nn.Flatten(),
        Sequential_Modules(Linear_Module, params_lin_sequential),
    )
    return model


def Model_bbox():
    """
    Hard-fixed structure of the model.
    """
    params_conv_parallel = [
        {
            "ci": 1,
            "co": 5,
            "ck": 1,
            "cs": 1,
            "cp": 0,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 5,
            "cs": 1,
            "cp": 2,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 9,
            "cs": 1,
            "cp": 4,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 1,
            "co": 5,
            "ck": 13,
            "cs": 1,
            "cp": 6,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
    ]
    params_conv_sequential = [
        {
            "ci": 20,
            "co": 30,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 30,
            "co": 30,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 30,
            "co": 40,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "frc",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 40,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 0,
            "pd": 1,
        },
        {
            "ci": 60,
            "co": 60,
            "ck": 3,
            "cs": 1,
            "cp": 1,
            "cd": 1,
            "at": "prelu",
            "pt": "max",
            "pk": 2,
            "ps": 2,
            "pp": 1,
            "pd": 1,
        },
    ]
    params_lin_sequential = [
        {"dp": 0.4, "li": 960, "lo": 256, "at": "sigm"},
        {"dp": 0.4, "li": 256, "lo": 64, "at": "sigm"},
        {"dp": 0.4, "li": 64, "lo": 4, "at": "sigm"},
    ]
    model = nn.Sequential(
        Parallel_Modules(Conv2d_Module, params_conv_parallel),
        Sequential_Modules(Conv2d_Module, params_conv_sequential),
        nn.Flatten(),
        Sequential_Modules(Linear_Module, params_lin_sequential),
    )
    return model


def Model_label_defining():
    """
    Hard-fixed structure of the model.
    """
    model = nn.Sequential(
        nn.Conv2d(1, 100, (6, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.Conv2d(100, 100, (1, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.Conv2d(100, 100, (1, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.Conv2d(100, 100, (1, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.MaxPool2d((1, 2), stride=(1, 2)),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(5300, 64),
        nn.Sigmoid(),
        nn.Linear(64, 4),
        nn.Softmax(dim=1),
    )
    return model


def Model_bbox_amount_defining():
    """
    Hard-fixed structure of the model.
    """
    model = nn.Sequential(
        nn.Conv2d(1, 100, (24, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.Conv2d(100, 100, (1, 1)),
        nn.BatchNorm2d(100),
        nn.PReLU(),
        nn.Conv2d(100, 200, (1, 1)),
        nn.BatchNorm2d(200),
        nn.PReLU(),
        nn.Conv2d(200, 200, (1, 1)),
        nn.BatchNorm2d(200),
        nn.PReLU(),
        nn.MaxPool2d((1, 2), padding=(0, 1), stride=(1, 2)),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(2800, 512),
        nn.Sigmoid(),
        nn.Linear(512, 64),
        nn.Sigmoid(),
        nn.Linear(64, 4),
        nn.Softmax(dim=1),
    )
    return model


def Model_bbox_defining():
    """
    Hard-fixed structure of the model.
    """
    model = nn.Sequential(
        nn.Linear(648, 1024),
        nn.PReLU(),
        nn.Linear(1024, 2048),
        nn.PReLU(),
        nn.Linear(2048, 2048),
        nn.PReLU(),
        nn.Linear(2048, 1024),
        nn.PReLU(),
        nn.Linear(1024, 1024),
        nn.PReLU(),
        nn.Linear(1024, 4),
        nn.Sigmoid(),
    )
    return model


class Parallel_Modules(nn.Module):
    def __init__(self, module, params):
        super(Parallel_Modules, self).__init__()
        self.layers = nn.ModuleList([module(**params[i]) for i in range(len(params))])

    def forward(self, x):
        x_out = []
        for i, layer in enumerate(self.layers):
            x_out.append(layer(x))
        x = torch.cat(x_out, dim=1)
        return x


class Sequential_Modules(nn.Module):
    def __init__(self, module, params):
        super(Sequential_Modules, self).__init__()
        self.layers = nn.ModuleList([module(**params[i]) for i in range(len(params))])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv2d_Module(nn.Module):
    def __init__(self, ci, co, ck, cs, cp, cd, at, pt, pk, ps, pp, pd):
        """
        ci, co, ck, cs, cp, cd:
            c* - "conv" layer;
            *i - in_channels;
            *o - out_channels;
            etc... (see torch docs)
        pk, ps, pp, pd:
            p* - "pool" layer;
            pt - type: "max" or "avg" (my own, no docs);
            *k - kernel_size;
            etc... (see torch docs)
        """
        super(Conv2d_Module, self).__init__()
        EPS = 1e-10
        self.poolings = nn.ModuleDict(
            {
                "max": nn.MaxPool2d(pk, stride=ps, padding=pp, dilation=pd),
                "avg": nn.AvgPool2d(pk, stride=ps, padding=pp),
                "frc": nn.FractionalMaxPool2d(
                    pk, output_ratio=(1 / (ps + EPS), 1 / (ps + EPS))
                ),
            }
        )
        self.activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "prelu": nn.PReLU(),
                "sigm": nn.Sigmoid(),
                "tanh": nn.Tanh(),
            }
        )
        self.module = nn.Sequential(
            nn.Conv2d(ci, co, ck, stride=cs, padding=cp, dilation=cd),
            nn.BatchNorm2d(co),
            self.activations[at],
            self.poolings[pt],
        )

    def forward(self, x):
        return self.module(x)


class Conv1d_Module(nn.Module):
    def __init__(self, ci, co, ck, cs, cp, cd, at, pt, pk, ps, pp, pd):
        """
        ci, co, ck, cs, cp, cd:
            c* - "conv" layer;
            *i - in_channels;
            *o - out_channels;
            etc... (see torch docs)
        pk, ps, pp, pd:
            p* - "pool" layer;
            pt - type: "max" or "avg" (my own, no docs);
            *k - kernel_size;
            etc... (see torch docs)
        """
        super(Conv2d_Module, self).__init__()
        EPS = 1e-10
        self.poolings = nn.ModuleDict(
            {
                "max": nn.MaxPool1d(pk, stride=ps, padding=pp, dilation=pd),
                "avg": nn.AvgPool1d(pk, stride=ps, padding=pp),
                "frc": nn.FractionalMaxPool1d(
                    pk, output_ratio=(1 / (ps + EPS), 1 / (ps + EPS))
                ),
            }
        )
        self.activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "prelu": nn.PReLU(),
                "sigm": nn.Sigmoid(),
                "tanh": nn.Tanh(),
            }
        )
        self.module = nn.Sequential(
            nn.Conv1d(ci, co, ck, stride=cs, padding=cp, dilation=cd),
            nn.BatchNorm1d(co),
            self.activations[at],
            self.poolings[pt],
        )

    def forward(self, x):
        return self.module(x)


class Linear_Module(nn.Module):
    def __init__(self, dp, li, lo, at):
        """
        dp - dropout proba;
        li - linear in_units;
        lo - linear out_units;
        at - type of the activation function.
        """
        super(Linear_Module, self).__init__()
        self.activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "prelu": nn.PReLU(),
                "sigm": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "soft": nn.Softmax(),
            }
        )
        self.module = nn.Sequential(
            nn.Dropout(dp),
            nn.Linear(li, lo),
            self.activations[at],
        )

    def forward(self, x):
        return self.module(x)
