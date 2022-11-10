import json
import datetime
from flask import Response
from sqlalchemy import inspect
from api_views.users import *
from finetune_zoo.models import *
#from api_models.query_model import FineTuneQuery
import re
def populate_db():
    db.drop_all()
    db.create_all()
    response_text = '{ "message": "Database reset." }'
    response = Response(response_text, 200, mimetype='application/json')
    return response

def basic():
    response_text = '{ "message": "MLMAC IBM", "Help": "MLMAC IBM is a developmental API of the machine learning model attribution challenge (MLMAC)." } '
    response = Response(response_text, 200, mimetype='application/json')
    return response


def query(model):
    request_data = request.get_json()
    try:
        jsonschema.validate(request_data, query_schema)
    except:
        return Response(error_message_helper("Please provide a proper JSON body."), 400, mimetype="application/json")
    resp, _ = token_validator(request.headers.get('Authorization'))

    if "expired" in resp:
        return Response(error_message_helper(resp), 401, mimetype="application/json")
    elif "Invalid token" in resp:
        return Response(error_message_helper(resp), 401, mimetype="application/json")
    else:
        #Query.__table__.columns.keys()
        user = User.query.filter_by(username=resp).first()
        model = str(model)
        try:
            ft_response = ft_models[model](request_data.get('prompt'))
        except:
            try:
                # 1024 prompt
                prompt = ft_models[model].tokenizer.decode(ft_models[model].tokenizer(request_data.get('prompt')).data['input_ids'][:1024])
                ft_response = ft_models[model](prompt)
            except:
                # 512 prompt
                prompt = ft_models[model].tokenizer.decode(ft_models[model].tokenizer(request_data.get('prompt')).data['input_ids'][:512])
                ft_response = ft_models[model](prompt)
        model_queries = {}
        for attr, column in inspect(user.__class__).c.items():
            num_queries = getattr(user, attr)
            if re.search('query_' + model, column.name):
                setattr(user, attr, num_queries + 1)
                db.session.commit()

            if num_queries != 0 and 'query' in column.name:
                model_queries[column.name[-1]] = num_queries

        return Response(json.dumps({'status': 'success',
                                    'result': ft_response[0],
                                    'queries': model_queries}),
                        200,
                        mimetype="application/json")


def status():
    #request_data = request.get_json()
    api_key = request.headers.get('Authorization')
    print(token_validator(api_key))
    resp, token_made = token_validator(api_key)
    if "expired" in resp:
        return Response(error_message_helper(resp), 401, mimetype="application/json")
    elif "Invalid token" in resp:
        return Response(error_message_helper(resp), 401, mimetype="application/json")
    else:
        user = User.query.filter_by(username=resp).first()

        model_queries = {}
        print(user)
        for attr, column in inspect(user.__class__).c.items():
            if getattr(user, attr) != 0 and 'query' in column.name:
                model_queries[column.name[-1]] = getattr(user, attr)

        total_queries = sum([num_q for num_q in model_queries.values()])
        token_made = datetime.datetime.fromtimestamp(token_made).strftime("%H:%M:%S %d-%m-%Y")

        return Response(json.dumps({'api_key': api_key,
                                    'name': user.username,
                                    'created': token_made,
                                    'total_queries': total_queries,
                                    'queries': model_queries}),
                        200,
                        mimetype="application/json")