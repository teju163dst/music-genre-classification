

import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')

from PIL import Image

#Set title

st.title('Spotify app')
image = Image.open('spotify.png')
st.image(image,use_column_width=True)



def main():
	activities=['EDA','Visualisation','model','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	


#DEALING WITH THE EDA PART


	if option=='EDA':
		st.subheader("Exploratory Data Analysis")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox("Display shape"):
				st.write(df.shape)
			if st.checkbox("Display columns"):
				st.write(df.columns)
			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect('Select preferred columns:',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox("Display summary"):
				st.write(df1.describe().T)

			if st.checkbox('Display Null Values'):
				st.write(df.isnull().sum())

			if st.checkbox("Display the data types"):
				st.write(df.dtypes)
			if st.checkbox('Display Correlation of data variuos columns'):
				st.write(df.corr())




#DEALING WITH THE VISUALISATION PART


	elif option=='Visualisation':
		st.subheader("Data Visualisation")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select Multiple columns to plot'):
				selected_columns=st.multiselect('Select your preferred columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox('Display Heatmap'):
				st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot()
			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()
			if st.checkbox('Display Pie Chart'):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox("select column to display",all_columns)
				pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()





	# DEALING WITH THE MODEL BUILDING PART

	elif option=='model':
		st.subheader("Model Building")

		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select Multiple columns'):
				new_data=st.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected",df.columns)
				df1=df[new_data]
				st.dataframe(df1)


				#Dividing my data into X and y variables

				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]


			seed=st.sidebar.slider('Seed',1,200)

			classifier_name=st.sidebar.selectbox('Select your preferred classifier:',('KNN','SVM','LR','naive_bayes','RandomForestClassifier'))


			def add_parameter(name_of_clf):
				params=dict()
				if name_of_clf=='SVM':
					C=st.sidebar.slider('C',0.01, 15.0)
					params['C']=C
				else:
					name_of_clf=='KNN'
					K=st.sidebar.slider('K',1,15)
					params['K']=K
					return params

			#calling the function

			params=add_parameter(classifier_name)



			#defing a function for our classifier

			def get_classifier(name_of_clf,params):
				clf= None
				if name_of_clf=='SVM':
					clf=SVC(C=params['C'])
				elif name_of_clf=='KNN':
					clf=KNeighborsClassifier(n_neighbors=params['K'])
				elif name_of_clf=='LR':
					clf=LogisticRegression()
				elif name_of_clf=='naive_bayes':
					clf=GaussianNB()
				elif name_of_clf=='RandomForestClassifier':
					clf=RandomForestClassifier()
				
				else:
					st.warning('Select your choice of algorithm')

				return clf

			clf=get_classifier(classifier_name,params)


			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=seed)

			clf.fit(X_train,y_train)

			

			y_pred=clf.predict(X_test)
			st.write('Predictions:',y_pred)

			accuracy=accuracy_score(y_test,y_pred)

			st.write('Nmae of classifier:',classifier_name)
			st.write('Accuracy',accuracy)








#DELING WITH THE ABOUT US PAGE



	elif option=='About us':

		st.markdown('This is Spotify genre classifier app. The dataset is taken from Kaggle.com') 


		st.balloons()
	# 	..............


if __name__ == '__main__':
	main()