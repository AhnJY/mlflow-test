import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# github actions 환경변수에서 가져오거나 직접 입력
tracking_rui = "http://52.79.234.78:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Automated_Training")

def train_model(alpha, l1_ratio):
	with mlflow.start_run():
		mlflow.log_param("storage", "s3")
		data = load_diabetes()
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

        	# 모델 학습
		lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
		lr.fit(X_train, y_train)
        
        	# 예측 및 평가
		predictions = lr.predict(X_test)
		mse = mean_squared_error(y_test, predictions)
		rmse = np.sqrt(mse)

        	# MLflow에 파라미터 및 메트릭 기록
		mlflow.log_param("alpha", alpha)
		mlflow.log_param("l1_ratio", l1_ratio)
		mlflow.log_metric("rmse", rmse)
        
		# 모델 저장
		mlflow.sklearn.log_model(lr, "model")
		print(f"Run finished with RMSE: {rmse}")

if __name__ == "__main__":
	# 다양한 하이퍼파라미터로 시도
	train_model(0.1, 0.1)
	train_model(0.5, 0.5)
