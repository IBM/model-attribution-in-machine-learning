# Machine Learning Model Attribution

This repository contains code for the paper [Model Attribution in Machine Learning](https://arxiv.org/abs/....) and instructions for reproducing the environmental setup.

## Starting the API server

The API server is a Flask application that runs on port 5001. To start the server, using docker-compose, run the following command:

```bash
docker-compose -f api/docker-compose.yml up
```

TODO: add startup without docker

The server will be available at [http://localhost:5001](http://localhost:5001).

## Prompting a model trough the API

Run the following script to derive responses from a custom set of prompts:

   ```bash
   python compute_responses/compute_responses.py
   ``` 

Run the following script to derive responses from the pile prompts:

   ```bash
   python compute_responses/compute_pile_responses.py
   ``` 

Both scripts will prompt the model and save the responses in a json file.

## Run the classifier on the responses

To run the experiments, use the following command:

   ```bash
   python perform_attribution.py
   ```

The script will output the model attribution results.

### Fine-grained evaluation

   ```bash
   python k_auc_prec_recall.py
   ```

The report will contain Table 1 and Table 2 from the paper. TODO: Add reference to the paper