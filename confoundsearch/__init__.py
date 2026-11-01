"""ConfoundSearch: an agent that builds a confound graph as a Markov Decision Process."""

from __future__ import annotations

from confoundsearch.data.corpus import Corpus, InMemoryCorpus
from confoundsearch.data.es import ElasticsearchCorpus
from confoundsearch.data.keyword_index import KeywordIndex, KeywordNotFound, slugify, without_echoes
from confoundsearch.data.keyword_service import HttpCorpus, HttpVectorStore, KeywordService
from confoundsearch.data.sources import OpenAlexCorpus, SemanticScholarCorpus
from confoundsearch.data.vectors import (
    InMemoryVectorStore,
    Match,
    PineconeVectorStore,
    VectorRecord,
    VectorStore,
)
from confoundsearch.errors import CorpusError, InvalidAction, LLMError, RemoteError, VectorStoreError
from confoundsearch.interface.export import export_zip, save_export, to_share_data
from confoundsearch.logic.actions import (
    Action,
    AddConfound,
    AddEdge,
    AttachPassage,
    Merge,
    Retrieve,
    Stop,
    enumerate_actions,
)
from confoundsearch.logic.audit import AuditedClient, AuditLog, AuditTrail, verify
from confoundsearch.logic.deterministic import DeterministicPolicy
from confoundsearch.logic.episode import Step, Trajectory, run_episode
from confoundsearch.logic.expand import ExpandedCorpus
from confoundsearch.logic.graph import Graph, Passage, Vertex
from confoundsearch.logic.policy import EpsilonGreedy, LookaheadQ, Softmax
from confoundsearch.logic.reward import phi, reward
from confoundsearch.logic.state import State
from confoundsearch.logic.transition import Environment, Judge
from confoundsearch.model.embeddings import OpenAIEmbedder
from confoundsearch.model.llm import (
    LLMConfounds,
    LLMJudge,
    LLMProposer,
    LLMRewordings,
    LLMSynonyms,
    OpenAICompatibleClient,
    Proposer,
)

__version__ = "0.4.0"

__all__ = [
    "Action",
    "AddConfound",
    "AddEdge",
    "AttachPassage",
    "AuditLog",
    "AuditTrail",
    "AuditedClient",
    "Corpus",
    "CorpusError",
    "DeterministicPolicy",
    "ElasticsearchCorpus",
    "Environment",
    "EpsilonGreedy",
    "ExpandedCorpus",
    "Graph",
    "HttpCorpus",
    "HttpVectorStore",
    "InMemoryCorpus",
    "InMemoryVectorStore",
    "InvalidAction",
    "Judge",
    "KeywordIndex",
    "KeywordNotFound",
    "KeywordService",
    "LLMConfounds",
    "LLMError",
    "LLMJudge",
    "LLMProposer",
    "LLMRewordings",
    "LLMSynonyms",
    "LookaheadQ",
    "Match",
    "Merge",
    "OpenAICompatibleClient",
    "OpenAIEmbedder",
    "OpenAlexCorpus",
    "Passage",
    "PineconeVectorStore",
    "Proposer",
    "RemoteError",
    "Retrieve",
    "SemanticScholarCorpus",
    "Softmax",
    "State",
    "Step",
    "Stop",
    "Trajectory",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "Vertex",
    "enumerate_actions",
    "export_zip",
    "phi",
    "reward",
    "run_episode",
    "save_export",
    "slugify",
    "to_share_data",
    "verify",
    "without_echoes",
]
