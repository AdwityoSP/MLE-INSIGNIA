# MLE-INSIGNIA

In QnA, i used decision tree because of the EDA and requirement to implement feature selection
But after the math calculation, code, and training
The results model not works really well to the validation data

So for the deployment i used lazy regressor to get a rapid benchmark for every possible methods
After run lazy regressor, the best one is LassoLars
And the results is far more better than decision tree

How to use:
1. go to terminal:
git clone https://github.com/AdwityoSP/MLE-INSIGNIA.git
2. move dir:
cd .\MLE-INSIGNIA\house-price-insignia\
3. installation:
pip install -r requirements.txt
4. training:
python train_model.py
5. docker build:
docker build -t house-price-prediction
6. docker run:
docker run -d -p 7860:7860 --name house-price-app house-price-prediction
7. docker check:
docker ps
8. access gradio interface:
http://localhost:7860/
