from pathlib import Path

import yaml
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader

with open(Path(__file__).parent / 'users.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


def get_authenticator():
    return Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
