from enum import Enum


class TranslationSteps(str, Enum):
    """
    PENDING ->  IN_PROGRESS -> AUDIO_EXTRACTION -> AUDIO_ANALYSIS ->
        GENERATE_EPISODE_SUMMARY -> GENERATE_TRANSLATION_SUGGESTIONS -> RANK_AND_JUDGE -> COMPLETED
    |||--> FAILED
    """
    PENDING = 'pe'
    REQUESTED = 're'
    IN_PROGRESS = 'ip'
    AUDIO_EXTRACTION = 'fae'
    AUDIO_ANALYSIS = 'aa'
    GENERATE_EPISODE_SUMMARY = 'ges'
    GETTING_SIMILAR_EXAMPLES = 'gse'
    GENERATE_TRANSLATION_SUGGESTIONS = 'gts'
    RANK_AND_JUDGE = 'rj'
    COMPLETED = 'co'

    FAILED = 'fa'

    RUNNING_STATES = [
        IN_PROGRESS, AUDIO_EXTRACTION, AUDIO_ANALYSIS, GENERATE_EPISODE_SUMMARY,
        GENERATE_TRANSLATION_SUGGESTIONS, RANK_AND_JUDGE
    ]

    MIGHT_BE_STALE = [REQUESTED, *RUNNING_STATES]
    FINAL_STATES = [COMPLETED, FAILED]


class Ages(int, Enum):
    ZERO = 0
    THREE = 3
    SIX = 6
    TWELVE = 12
    SIXTEEN = 16
    EIGHTEEN = 18


class Genres(str, Enum):
    NOOP = 'noop'
    ACTION = 'action'
    ADVENTURE = 'adventure'
    ANIMATION = 'animation'
    BIOGRAPHY = 'biography'
    COMEDY = 'comedy'
    CRIME = 'crime'
    DOCUMENTARY = 'documentary'
    DRAMA = 'drama'
    FAMILY = 'family'
    FANTASY = 'fantasy'
    FILM_NOIR = 'film-noir'
    HISTORY = 'history'
    HORROR = 'horror'
    MUSIC = 'music'
    MUSICAL = 'musical'
    MYSTERY = 'mystery'
    ROMANCE = 'romance'
    SCI_FI = 'sci-fi'
    SHORT_FILM = 'short-film'
    SPORT = 'sport'
    SUPERHERO = 'superhero'
    THRILLER = 'thriller'
    WAR = 'war'
    WESTERN = 'western'
    BASED_ON_REAL_STORY = 'based-on-real-story'


class ModelVersions(str, Enum):
    V1 = 'v1'
    V3 = 'v3'
    V037 = 'v0.3.7'
    V038 = 'v0.3.8'
    V039 = 'v0.3.9'
    V039_CL = 'v0.3.9-cl'
    V0310 = 'v0.3.10'
    V0311 = 'v0.3.11'
    V0312 = 'v0.3.12'
    V0313 = 'v0.3.13'
    V0314 = 'v0.3.14'

    LATEST = V0312
