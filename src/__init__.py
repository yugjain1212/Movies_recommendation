"""Movie Recommendation System := A hybrid recommendation system combinig content based and collaborative filtering  """


__version__ = '1.0.0'
__author__ = 'Yug jain'


from . import data_preprocessing
from . import content_based
from . import collaborative
from . import utils
from . import evaluation


__all__ = [
    'data_preprocessing',
    'content_based',
    'collaborative',
    'utils',
    'evaluation'

]