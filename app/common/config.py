from pydantic_settings import BaseSettings


class _MongoDBSettings(BaseSettings):
    # MONGO_FULL_URL: str = 'mongodb+srv://cluster1.bb0fppm.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority'
    MONGO_FULL_URL: str = 'mongodb+srv://cluster1.bb0fppm.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority'
    # MONGO_FULL_URL: str = 'mongodb://localhost:27017'
    SSL_CA_CERTS: bool = True
    DATABASE_NAME: str = 'projects'


class _OpenAISettings(BaseSettings):
    OPENAI_KEY: str = ''

    AZURE_TEXT_OPENAI_KEY: str = ''
    AZURE_TEXT_OPENAI_ENDPOINT: str = 'https://glixbackup.openai.azure.com/'

    AZURE_VISION_OPENAI_KEY: str = ''
    AZURE_VISION_OPENAI_ENDPOINT: str = 'https://glix.openai.azure.com/'

    AZURE_TEXT_OPENAI_KEY_BACKUP: str = ''
    AZURE_TEXT_OPENAI_ENDPOINT_BACKUP: str = 'https://glix.openai.azure.com/'

    VERSION: str = "2023-12-01-preview"


class _GeneralSettings(BaseSettings):
    DEBUG: bool = True

    SYSTEM_EMAIL: str = 'yovell04@gmail.com'
    SENDGRID_API_KEY: str = 'Bearer 123'

    BLOB_ACCOUNT_KEY: str = 'U2VAPwlpZFc7hzx6qq8Z6TttY/vss1BFdaqmtEMq8GXWwiyHT3pq2Rh94Rx3Dfbu2c1k7cJ4S54n+AStH/cWgQ=='
    BLOB_ACCOUNT_NAME: str = 'glixbe'
    AZURE_STORAGE_CONNECTION_STRING: str = 'DefaultEndpointsProtocol=https;AccountName=glixbe;AccountKey=U2VAPwlpZFc7hzx6qq8Z6TttY/vss1BFdaqmtEMq8GXWwiyHT3pq2Rh94Rx3Dfbu2c1k7cJ4S54n+AStH/cWgQ==;EndpointSuffix=core.windows.net'

    PROJECT_BLOB_CONTAINER: str = 'projects'

    DEEPGRAM_API_KEY: str = 'd82ffb1b78dbec71fd6c953e297f69aac54cad2c'
    SMODIN_API_KEY: str = "Sydb1bgyb896b20e86Z0a2fd04c4767L4fbegpbfbeLE95f3uq4130Y08fcewI6fd6EJ0235wU4e77rr4055g7abe2naa009"


settings = _GeneralSettings()
mongodb_settings = _MongoDBSettings()
openai_settings = _OpenAISettings()
