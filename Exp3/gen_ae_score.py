import numpy as np
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

from DeepODModel.AE.run_ae import  run_just_one

if __name__ == '__main__':
    template_model_name = 'AE'
    training_epoch = 300
    g = os.walk(r"./data")
    cnt = 0
    Dict = dict()
    skip = -1
    dict_path = f'./od-predict-score/{template_model_name}_scores.npy'
    if skip !=-1:
        Dict = np.load(dict_path,allow_pickle=True)[()]

    for path,dir_list,file_list in g:
        for file_name in file_list:
            print(file_name)
            if cnt<=skip:
                cnt+=1
                continue
            _,_,scores = run_just_one(path,file_name,epoch=training_epoch)

            Dict[file_name] = scores
            print(cnt)
            if cnt %5==0:
                print('cnt save at ',cnt)
                np.save(dict_path,Dict)
            cnt+=1
    np.save(dict_path,Dict)
