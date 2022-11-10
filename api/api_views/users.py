import re
import jsonschema
import jwt

from config import db
from api_views.json_schemas import *
from flask import jsonify, Response, request, json
from api_models.user_model import User



def error_message_helper(msg):
    return '{ "status": "fail", "message": "' + msg + '"}'


def register_user():
    request_data = request.get_json()
    # check if user already exists
    user = User.query.filter_by(username=request_data.get('username')).first()
    if not user:
        try:
            # validate the data are in the correct form
            jsonschema.validate(request_data, register_user_schema)
            user = User(username=request_data['username'], query_0=0)
            db.session.add(user)
            db.session.commit()

            responseObject = {
                'status': 'success',
                'message': 'Successfully registered. Login to receive an auth token.'
            }

            return Response(json.dumps(responseObject), 200, mimetype="application/json")
        except jsonschema.exceptions.ValidationError as exc:
            return Response(error_message_helper(exc.message), 400, mimetype="application/json")
    else:
        return Response(error_message_helper("User already exists. Please Log in."), 200, mimetype="application/json")


def generate_token():
    request_data = request.get_json()
    try:
        # validate the data are in the correct form
        jsonschema.validate(request_data, generate_token_schema)
        # fetching user data if the user exists
        user = User.query.filter_by(username=request_data.get('username')).first()
        if user:
            auth_token = user.encode_auth_token(user.username)
            responseObject = {
                'status': 'success',
                'message': 'Successfully logged in.',
                'auth_token': auth_token
            }
            return Response(json.dumps(responseObject), 200, mimetype="application/json")
        if (user and request_data.get('password') != user.password) or (not user):
            return Response(error_message_helper("Username or Password Incorrect!"), 200, mimetype="application/json")
    except jsonschema.exceptions.ValidationError as exc:
        return Response(error_message_helper(exc.message), 400, mimetype="application/json")


def token_validator(auth_header):
    if auth_header:
        try:
            auth_token = auth_header.split(" ")[1]
        except:
            auth_token = ""
    else:
        auth_token = ""
    if auth_token:
        # if auth_token is valid we get back the username of the user
        print(User.decode_auth_token(auth_token))
        return User.decode_auth_token(auth_token)
    else:
        return "Invalid token", None
