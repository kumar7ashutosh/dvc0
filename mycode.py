import os,pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def create_dataframe()->pd.DataFrame:
    data={
        'id':[1,2,3,4,5,6,7,8,9,10],
        'review':[
            'Great food and ambiance.',
            'Terrible services.',
            'Amazing experience!',
            'Food was cold.',
            'Loved the desert.',
            'Not worth of money.',
            'Excellent customer service.',
            'The place was too crowded.',
            'Best restaurant in town.',
            'Average experience.'
        ]
    }
    df=pd.DataFrame(data)
    return df

def save_dataframe(df):
    os.makedirs('data',exist_ok=True)
    df.to_csv('data/data.csv',index=False)
    print('data.csv stored in data folder')
    
def process_data(k)->pd.DataFrame:
    df=pd.read_csv('data/data.csv')
    vec=CountVectorizer(max_features=k)
    vec_data=vec.fit_transform(df['review'])
    feature_names=vec.get_feature_names_out()
    vec_df=pd.DataFrame(vec_data.toarray(),columns=feature_names)
    new_df=pd.concat([df,vec_df],axis=1)
    new_df.to_csv('data/new_df.csv',index=False)
    return new_df

if __name__=='__main__':
    df=create_dataframe()
    save_dataframe(df)
    new_df=process_data(k=3)
    