import argparse
from preprocessing import Preprocessing
from network import MatchBox
from train import Train
from test import Test
import custom_layers

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', nargs='?', type=str)
checkpoint_path = parser.parse_args().test
if checkpoint_path is None:
    train_bool = True
else:
    train_bool = False

#this file can stay full
prep = Preprocessing()
# feature_instance = Feature(prep)
# train_dataset, val_dataset, test_dataset = feature_instance.create_iterators()
train_dataset, val_dataset, test_dataset = prep.create_iterators()
for i in train_dataset.take(1):
    print(i[0].shape)
    print(i[1].shape)
    # print(i[1])
    # exit()
feature_layer = custom_layers.FeatureLayer()
network = MatchBox(feature_layer)

if train_bool:
    # runs Train.py
    train_instance = Train(network, train_dataset, val_dataset)
    train_instance.train()
else:
    print('test.......')
    test_obj = Test(network, test_dataset)
    test_obj.test(checkpoint_path)

