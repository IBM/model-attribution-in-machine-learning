# MLMAT API

API that replicates functionality of [mlmac.io](mlmac.io) 

Currently only for internal development. To install and begin running:
```
pip install -r requirements.txt
python app.py
```
To use the API:

 1. Register a username
 2. Login with said username to receive an API token 
 3. Happy querying!

Use the ```Models.py``` file to query the models with a simplified interface:
```
from Models import Model
IBM_MLMAT_TOKEN = TOKEN

ft_models = [Model("ibm", IBM_MLMAT_TOKEN, idx) for idx in range(len(ft_models))]

input = "This is a prompt for a model"
output = ft_models[0](input)
```


# Endpoints
http://localhost:8080/db-init 

- call this endpoint to populate the database or reset it. 

http://localhost:8080/v1/register

- register (only requires a username) to interact with the api 

http://localhost:8080/v1/login

- login (with username) to generate an API token

http://localhost:8080/v1/status

- view number of requests made by the user so far

http://localhost:8080/v1/query/{model}

- query a model


