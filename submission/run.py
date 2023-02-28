from catboost import CatBoostClassifier
import pandas as pd
import sys

from model import reliable_predict


def main():
    source_file, output_path = sys.argv[1:]
    
    bins_path = "./nn_bins.pickle"
    model_path = "./nn_weights.ckpt"
    result = reliable_predict(source_file, bins_path, model_path)
    
    transactions = pd.read_csv(source_file, parse_dates=['transaction_dttm'])
    transactions['time'] = transactions.transaction_dttm.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    
    mcc_code_features = transactions.pivot_table(
        index='user_id', columns=['mcc_code'], values=['transaction_amt'], 
        aggfunc=['count'], fill_value=0
    )
    mcc_code_features.columns = [f'{i[0]}-mcc_code:{i[2]}' for i in mcc_code_features.columns]
    for col in mcc_code_features.columns:
        mcc_code_features[col] //= 20
        
    time_features = transactions.groupby('user_id')['time'].agg(['mean', 'std', 'min', 'max', 'median'])
    time_features.columns = [f'tr_time_{c}' for c in time_features.columns]
    
    df_test = pd.concat([
        result.set_index('user_id').target.rename('nn_predict'),
        time_features,
        mcc_code_features
    ], axis=1)
    
    model_cb = CatBoostClassifier().load_model("./model_cb.cbm")
    columns = model_cb.get_feature_importance(prettified=True)['Feature Id'].values
    
    for col in columns:
        if col not in df_test.columns:
            df_test[col] = 0
    predicts = model_cb.predict_proba(df_test[columns])[:,1]
    
    submission = pd.DataFrame({"user_id": df_test.index, "target": predicts})
    submission.to_csv(output_path, index=False)
    


if __name__ == "__main__":
    main()