from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from keras.models import load_model
import numpy as np
import argparse
from keras.layers import Input, Dense, Dropout, Activation, ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential

def getModel():
    inputs = Input(shape=(100,), name='network_input')
    x = Dense(2048, kernel_initializer="uniform", name='hiddenLayer1')(inputs)
    bn1 = BatchNormalization()(x)
    act = Activation("relu")(bn1)
    d1 = Dropout(0.5)(act)

    y = Dense(1024, kernel_initializer="uniform", name='hiddenLayer2')(d1)
    bn2 = BatchNormalization()(y)
    act2 = Activation("relu")(bn2)
    d2 = Dropout(0.5)(act2)

    z = Dense(512, kernel_initializer="uniform", name='hiddenLayer3')(d2)
    bn3 = BatchNormalization()(z)
    act3 = Activation("relu")(bn3)
    prefinal = Dense(1, name="outputLayer")(act3)
    output =ReLU(max_value=1.0, negative_slope=0.0, threshold=0.0)(prefinal)
    model = Model(inputs=inputs, outputs=output)
    return model

def main():
    if args.out_file:
        f = open(args.out_file, "w")
    else:
        f = sys.stdout
    model = getModel()
    model.load_weights(args.model_path, by_name=True)
    model.compile(optimizer="Adam", loss="mean_absolute_error")
    model.summary(print_fn=lambda x: print(x,file=f))
    features = np.load(args.features_path)
    predictions = model.predict(features)
    if args.labels_path:
        labels = np.load(args.labels_path)
        mae = np.mean(np.abs(labels - predictions))
        print("MAE: {}\n".format(mae), file=f)
    print("Predictions:",file=f)
    for prediction in predictions:
        print(prediction[0],file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to model weights")
    parser.add_argument("--features_path", type=str, help="path to dataset feature")
    parser.add_argument("--labels_path", type=str, help="optional: path to dataset labels")
    parser.add_argument("--out_file", type=str, help="path to output file", default="")
    args = parser.parse_args()
    main()
