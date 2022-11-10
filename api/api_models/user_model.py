import datetime
import jwt
from config import db, api
from app import alive
#from api_models.query_model import FineTuneQuery
from sqlalchemy.orm import relationship

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, unique=True, autoincrement=True)
    username = db.Column(db.String(128), unique=True, nullable=False)
    query_0 = db.Column(db.Integer, nullable=False, default=int(0))
    query_1 = db.Column(db.Integer, nullable=False)
    query_2 = db.Column(db.Integer, nullable=False)
    query_3 = db.Column(db.Integer, nullable=False)
    query_4 = db.Column(db.Integer, nullable=False)
    query_5 = db.Column(db.Integer, nullable=False)
    query_6 = db.Column(db.Integer, nullable=False)
    query_7 = db.Column(db.Integer, nullable=False)

    #finetune_query = relationship("FineTuneQuery", order_by=FineTuneQuery.id, back_populates="user")

    def __init__(self, username, query_0):
        self.username = username
        self.query_0 = query_0
        self.query_1 = 0
        self.query_2 = 0
        self.query_3 = 0
        self.query_4 = 0
        self.query_5 = 0
        self.query_6 = 0
        self.query_7 = 0

    def __repr__(self):
        return f"<User(name={self.username})>"

    def encode_auth_token(self, user_id):
        try:
            payload = {
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=alive),
                'iat': datetime.datetime.utcnow(),
                'sub': user_id
            }
            return jwt.encode(
                payload,
                api.app.config.get('SECRET_KEY'),
                algorithm="HS256"
            )
        except Exception as e:
            return e

    @staticmethod
    def decode_auth_token(auth_token):
        try:
            print(auth_token)
            print(type(auth_token))
            payload = jwt.decode(auth_token, api.app.config.get('SECRET_KEY'), algorithms=["HS256"])
            return payload['sub'], payload['iat']
        except jwt.ExpiredSignatureError:
            return 'Signature expired. Please log in again.', None
        except jwt.InvalidTokenError:
            return 'Invalid token. Please log in again.', None

    def json(self):
        return{'username': self.username}

    @staticmethod
    def get_all_users():
        return [User.json(user) for user in User.query.all()]

    @staticmethod
    def register_user(username):
        new_user = User(username=username, query_0=0)
        db.session.add(new_user)
        db.session.commit()
