# mlops

This small project takes an image of a handwritten chinese character digit from 0 to 9 and predicts it.

Download the dataset here:  
https://www.kaggle.com/datasets/gpreda/chinese-mnist/data

Run the server with:
`uvicorn app:app`

Send a image to the app with:
`curl -F file=@test0.png http://127.0.0.1:8000/upload/`
