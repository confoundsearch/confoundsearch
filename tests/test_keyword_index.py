"""KeywordIndex: ask for the keywords near a name, and let spaces, hyphens and capitals not matter."""

import pytest

from confoundsearch import InMemoryVectorStore, KeywordIndex, VectorRecord, VectorStoreError, slugify


def store():
    rows = {
        "firm-size": ("firm size", 900, [1.0, 0.0]),
        "large-firms": ("large firms", 600, [0.9, 0.1]),
        "rare-firms": ("rare firms", 5, [0.95, 0.05]),
        "pt3sn": ("Pt3Sn", 700, [0.0, 1.0]),
        "tobin-s-q": ("Tobin's Q", 16252, [0.8, 0.6]),
        "female-dogs": ("bitches", 100, [0.2, 0.5]),  # a keyword renamed after its slug was fixed
        "montanuniversitat-leoben": ("Montanuniversit\u00e4t Leoben", 50, [0.3, 0.3]),
    }
    made = InMemoryVectorStore("kw")
    made.create_index(2)
    made.upsert([VectorRecord(i, v, {"name": n, "works": w}) for i, (n, w, v) in rows.items()])
    return made


@pytest.mark.parametrize(
    "name,slug",
    [
        ("firm size", "firm-size"),
        ("Firm  Size", "firm-size"),
        ("firm-size", "firm-size"),
        ("Tobin's q", "tobin-s-q"),
        ("Montanuniversit\u00e4t Leoben", "montanuniversitat-leoben"),
        ("foo_bar (baz)", "foo-bar-baz"),
        ("  --  ", ""),
    ],
)
def test_slugify_follows_openalexs_slugs(name, slug):
    assert slugify(name) == slug


@pytest.mark.parametrize("asked", ["firm size", "Firm Size", "firm-size", "FIRM_SIZE", " firm  size "])
def test_any_spelling_of_a_keyword_name_finds_it(asked):
    found = KeywordIndex(store()).query(asked, 2)
    assert [m.metadata["name"] for m in found] == ["rare firms", "large firms"]  # not firm size itself


def test_a_keyword_name_needs_no_embedder():
    def embedder(text):
        raise AssertionError("should not be called for a keyword")

    assert KeywordIndex(store(), embedder).query("firm size", 1)


def test_min_works_hides_rare_keywords_and_k_is_respected():
    found = KeywordIndex(store()).query("firm size", 10, min_works=100)
    names = [m.metadata["name"] for m in found]
    assert (
        "rare firms" not in names
        and names[0] == "large firms"
        and len(KeywordIndex(store()).query("firm size", 2)) == 2
    )


def test_accents_and_apostrophes_are_handled():
    index = KeywordIndex(store())
    assert index.keyword("Tobin's q").id == "tobin-s-q"
    assert index.keyword("Montanuniversitat Leoben").id == "montanuniversitat-leoben"


def test_a_slug_that_belongs_to_a_renamed_keyword_is_not_mistaken_for_it():
    index = KeywordIndex(store())
    assert index.keyword("female dogs") is None  # the slug exists, but that keyword's name is 'bitches' now
    assert index.keyword("bitches") is None  # and its own slug is not what the name spells


def test_nothing_to_look_up():
    index = KeywordIndex(store())
    assert index.keyword("---") is None and index.keyword("") is None
    assert index.keyword("no such thing") is None


def test_a_phrase_that_is_not_a_keyword_is_embedded_then_searched():
    seen = []

    def embedder(text):
        seen.append(text)
        return [0.0, 1.0]

    found = KeywordIndex(store()).query("a small plasma device", 1, embedder=embedder)
    assert seen == ["a small plasma device"] and found[0].metadata["name"] == "Pt3Sn"
    assert (
        KeywordIndex(store(), embedder).query("something else", 1, min_works=800)[0].metadata["name"]
        == "Tobin's Q"
    )


def test_without_an_embedder_a_non_keyword_is_an_error():
    with pytest.raises(VectorStoreError, match="no keyword named 'a small plasma device'"):
        KeywordIndex(store()).query("a small plasma device")


def echo_store():
    """Eight keywords: five echo 'firm size', three are other ideas, all pointing roughly the same way."""
    made = InMemoryVectorStore("kw")
    made.create_index(2)
    rows = [
        ("firm-size", "firm size", 1.0),
        ("firm-size-and-growth", "firm size and growth", 0.99),
        ("firm-size-classes", "firm size classes", 0.98),
        ("firm-size-distribution", "Firm Size Distribution", 0.97),
        ("organizational-size", "organizational size", 0.9),
        ("economies-of-scale", "economies of scale", 0.8),
        ("enterprise-scale", "enterprise scale", 0.7),
        ("pt3sn", "Pt3Sn", 0.0),
    ]
    made.upsert([VectorRecord(i, [x, 1.0 - x], {"name": n, "works": 100}) for i, n, x in rows])
    return made


def test_without_echoes_drops_names_that_contain_the_phrase_whatever_the_case():
    from confoundsearch import Match, without_echoes

    matches = [
        Match("a", 1.0, {"name": "Firm Size classes"}),
        Match("b", 0.9, {"name": "scale"}),
        Match("c", 0.8),
    ]
    assert [m.id for m in without_echoes(matches, " FIRM size ", 5)] == [
        "b",
        "c",
    ]  # no name: judged by its id
    assert [m.id for m in without_echoes(matches, "nothing", 2)] == ["a", "b"]  # and the cut at k


def test_distinct_leaves_out_the_echoes_so_other_ideas_come_up():
    index = KeywordIndex(echo_store())
    plain = [m.metadata["name"] for m in index.query("firm size", 3)]
    assert plain == ["firm size and growth", "firm size classes", "Firm Size Distribution"]
    distinct = [m.metadata["name"] for m in index.query("firm size", 3, distinct=True)]
    assert distinct == ["organizational size", "economies of scale", "enterprise scale"]


def test_distinct_asks_for_more_than_k_and_a_phrase_that_is_not_a_keyword_works_too():
    asked = []
    store = echo_store()
    original = store.query
    store.query = lambda vector, k, flt=None: asked.append(k) or original(vector, k, flt)
    KeywordIndex(store).query("firm size", 3, distinct=True)
    assert asked == [101]  # the pool of 100, and one more for the keyword itself
    names = [
        m.metadata["name"]
        for m in KeywordIndex(store).query("size of firms", 2, embedder=lambda t: [1.0, 0.0], distinct=True)
    ]
    assert names == ["firm size", "firm size and growth"]  # 'size of firms' is in none of these names


def test_pool_size_grows_with_k_and_is_capped():
    from confoundsearch.data.keyword_index import pool_size

    assert pool_size(5) == 100 and pool_size(50) == 200 and pool_size(10_000) == 1000
