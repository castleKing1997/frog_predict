#!encoding=utf-8
from keras.models import load_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
if __name__ == "__main__":
    label_dic = {0:u"消散",1:u"小雾",2:u"中雾",3:u"大雾",4:u"超大雾"}
    # data
    data = np.load("./MerchData/MerchData.npz",allow_pickle=True)
    X_val = data["X_val"]
    y_val = data["y_val"]
    # model
    model_best = load_model("./alex.ckp")
    y_pred = np.argmax(model_best.predict(X_val),axis=1)
    cfr = classification_report(y_val,y_pred,target_names=[u"消散",u"小雾",u"中雾",u"大雾",u"超大雾"])
    print(cfr)
    indexes = np.arange(len(y_val))
    indexes_tmp = np.random.choice(indexes,4)
    n_fig = 0
    plt.figure(figsize=(15,15))
    plt.tight_layout()
    for idx in indexes_tmp:
        n_fig += 1
        plt.subplot(2,2,n_fig)
        plt.imshow(X_val[idx])
        plt.title(label_dic[y_pred[idx]])
    plt.savefig("predict.png",dpi=300)
