import requests
import time

class Model:
    def __init__(self, api, api_token, model_id, use_cache=True):
        self.api = api
        self.api_token = api_token
        self.model_id = model_id
        self.use_cache = use_cache
        self.cache = {}

        if api == "hf":
            self.api_url = f"https://api-inference.huggingface.co/models/model-attribution-challenge/{model_id}"
        elif api == "mlmac":
            self.api_url = f"https://api.mlmac.io:8080/query?model={model_id}"
        elif api == "ibm":
            self.api_url = f"http://localhost:8080/v1/query/{model_id}"

    def __call__(self, input, max_retries=10, params={}, options={}):
        if self.use_cache and input in self.cache:
            return self.cache[input]

        if self.api == "hf":
            payload = {"inputs": input, "parameters": params, "options": options}
        elif self.api == "mlmac":
            payload = {"input": input}
        elif self.api == "ibm":
            payload = {"prompt": input}

        headers = {"Authorization": f"Bearer {self.api_token}"}

        for retry in range(max_retries):
            response = requests.post(self.api_url, json=payload, headers=headers)

            if response.status_code == 200:
                if self.api == "hf":
                    result = response.json()
                elif self.api == "mlmac":
                    result = response.json().get("result")
                elif self.api == "ibm":
                    result = response.json().get("result")
                self.cache[input] = result

                return result
            elif response.status_code == 503:
                print(response.json())
                print(f"attempt {retry + 1}/{max_retries}; waiting for 20 seconds")
                time.sleep(20.0)
            else:  # error
                raise Exception(response.text)

        raise Exception(f"Failed after {max_retries} attempts")