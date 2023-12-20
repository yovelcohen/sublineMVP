from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGO_FULL_URL: str = 'mongodb+srv://subline:9QPLIvm9Qup1oHn1@cluster1.bb0fppm.mongodb.net/?retryWrites=true&w=majority'

    # to get a string like this run:
    # openssl rand -hex 32
    SECRET_KEY: str = "cc3743c07011d1ba45ab5cbfeb31328c1550583c24bc6c4d6aa0a668fea65c2c"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 180
    SSL_CA_CERTS: bool = True
    DATABASE_NAME: str = 'projects'
    OPENAI_KEY: str = ''
    AZURE_OPENAI_ENDPOINT: str = 'https://glixdev.openai.azure.com/'
    SYSTEM_EMAIL: str = 'yovell04@gmail.com'
    SENDGRID_API_KEY: str = 'Bearer 123'

    BLOB_ACCOUNT_KEY: str = ''
    BLOB_ACCOUNT_NAME: str = ''
    AZURE_STORAGE_CONNECTION_STRING: str = 'DefaultEndpointsProtocol=https;AccountName=translatesubs;AccountKey=0lCm8sn9Ics7O8p7SMPskrax6tSvZE3RMpoEc84A4ta6DzaMdMlE5BbeyKhX4E36Y9K1mT5cGqbi+AStpf0gwA==;EndpointSuffix=core.windows.net'

    PROJECT_BLOB_CONTAINER: str = 'projects'
    AVATAR_CONTAINER: str = 'avatars'
    TRANSLATION_CONTAINER: str = ''

    GEMINI_API_KEY: str = 'AIzaSyCQkvKpN-gSkK3pZGuiuhmZY-A-ilUea2k'


settings = Settings()
