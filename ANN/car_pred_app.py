import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
st.header("Car Price Prediction")
st.subheader("Using Artificial Neural Networks")
data=st.file_uploader("Upload your dataset here", type=["csv"])
if data is not None:
    df=pd.read_csv(data)
    st.dataframe(df.head())
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    X=df.drop("price",axis=1)
    y=df['price']
    X.drop(['car_ID','CarName'],axis=1,inplace=True)
    obj_cols=X.select_dtypes(include=['object'])
    num_cols=X.select_dtypes(include=['int64','float64'])
    ss=StandardScaler()
    ohe=OneHotEncoder()
    num=pd.DataFrame(ss.fit_transform(num_cols),columns=num_cols.columns)
    obj=pd.DataFrame(ohe.fit_transform(obj_cols).toarray(),columns=ohe.get_feature_names_out(obj_cols.columns))
    X=pd.concat([num,obj],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    X_train=ss.fit_transform(X_train)
    X_test=ss.transform(X_test)
    model=Sequential()
    model.add(Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.summary(print_fn=lambda x: st.text(x))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    history=model.fit(X_train,y_train,batch_size=32,epochs=50,verbose=1)
    y_pred=model.predict(X_test)
    st.write("R2 Score:", r2_score(y_test,y_pred))
    st.write("MAE:", mean_absolute_error(y_test,y_pred))
    st.write("MSE:", mean_squared_error(y_test,y_pred))
