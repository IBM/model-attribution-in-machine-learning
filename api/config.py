import os
import connexion
from flask_sqlalchemy import SQLAlchemy
api = connexion.App(__name__, specification_dir='./openapi_specs')

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(api.root_path, 'database/database.db')
api.app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
api.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

api.app.config['SECRET_KEY'] = 'mlmac'
# start the db
db = SQLAlchemy(api.app)


api.add_api('openapi3.yml')


