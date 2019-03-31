from .cbow.base.module import CBOWBaseModule
from .cbow.hierarchical_softmax.module import CBOWHierarchicalSoftmaxModule
from .cbow.negative_sampling.module import CBOWNegativeSamplingModule
from .skip_gram.base.module import SkipGramBaseModule
from .skip_gram.negative_sampling.module import SkipGramNegativeSamplingModule
from .skip_gram.hierarchical_softmax.module import SkipGramHierarchicalSoftmaxModule
__all__ = ['CBOWBaseModule', 'CBOWNegativeSamplingModule', 'CBOWHierarchicalSoftmaxModule', 'SkipGramBaseModule',
           'SkipGramNegativeSamplingModule', 'SkipGramHierarchicalSoftmaxModule']
