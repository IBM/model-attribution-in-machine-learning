from config import api, db
import os


# token alive for how many days?
alive = 365

# start the api
if __name__ == '__main__':
    api.run(host='0.0.0.0', port=8080, debug=True)
