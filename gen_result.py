from ssblazer import my_model
from torch import load, device
from gen_data import get_dataloader
import pandas as pd


def do_pred(url, batch_size):
    print('Making predictions...')
    model = my_model.res()
    model_checkpoint = load(
        '.\ssblazer\ssblazer.pkl', map_location=device('cpu'))
    model_checkpoint = model_checkpoint['model_state_dict']
    model.load_state_dict(model_checkpoint)
    model.eval()

    data_loader = get_dataloader(url, batch_size)

    pred_list = []

    for i in data_loader:
        aa = i['seq']
        pred = model(aa).detach().squeeze().numpy().tolist()
        pred_list = pred_list+pred

    pred_len = len(pred_list)
    end_pos = list(range(126, pred_len+126, 1))
    start_pos=[i - 1 for i in end_pos]

    df=pd.DataFrame({'start':start_pos,'end':end_pos,'score':pred_list})
    df.to_csv('./result.csv',header=True,index=False)
    print('Done! Predictions are saved to result.csv.')
