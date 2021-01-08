#!encoding=utf-8

from LoadAllImage import *
from Model.CustomModels import *
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    import os,glob
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Parameters
    params = {'n_channels': 3,
              'shuffle': True,
              'dim': (224,224),
              'n_classes': 5,
              'one_hot': True}
    label_dic = {"消散":0,"小雾":1,"中雾":2,"大雾":3,"超大雾":4}
    # Datasets
    prefix = "./MerchData/*/*.jpg" 
    paths = glob.glob(prefix)
    x = {}
    x['train'] = [(path[:-4].split("/")[-2],path[:-4].split("/")[-1]) for path in paths]
    N = len(x['train'])
    y = {}
    y['train'] = {(path[:-4].split("/")[-2],path[:-4].split("/")[-1]):label_dic[path[:-4].split("/")[-2]] for path in paths}
    prefix = prefix.replace("*","{}")
    # Generators
    image_loader = ImageLoader(prefix,x['train'],y['train'],**params)
    X,y = image_loader.load_data()
    # Design model
    model = AlexNet((params["dim"]+(params["n_channels"],)),params["n_classes"])
    # Train model on dataset
    checkpoint = ModelCheckpoint("./alex.ckp",monitor="val_acc",save_best_only=True)
    model.fit(X,y,batch_size=128,epochs=30,validation_split=0.3,callbacks=[checkpoint,])
    X_train = X[:1941]
    y_train = y[:1941]
    y_train = np.argmax(y_train,axis=1)
    X_val = X[1941:]
    y_val = y[1941:]
    y_val = np.argmax(y_val,axis=1)
    np.savez("MerchData/MerchData",X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val)

