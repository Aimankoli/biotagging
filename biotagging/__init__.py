from .services.api import tag_sentence, tag_sentences
from .utils.tag_json import tag_from_json
from .utils.tag_pd import tag_from_dataframe

__version__ = "0.1.0"
__all__ = ["tag_sentence", "tag_sentences", "tag_from_json", "tag_from_dataframe", "__version__"]
