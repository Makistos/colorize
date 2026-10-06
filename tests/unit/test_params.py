import pytest

from colorizer.core.params import Param, parse_params, validate_params

SCHEMA = (
    Param("size", "int", 512, min=256, max=1024, step=64),
    Param("strength", "float", 0.5, min=0.0, max=1.0),
    Param("fast", "bool", False),
    Param("variant", "choice", "tiny", choices=("tiny", "large")),
    Param("prompt", "str", ""),
    Param("seed", "seed", 0, min=0),
    Param("hints", "points", []),
)


def test_defaults_filled():
    assert validate_params(SCHEMA, {}) == {p.name: p.default for p in SCHEMA}


def test_valid_values_pass_through():
    out = validate_params(
        SCHEMA, {"size": 768, "strength": 1, "variant": "large", "hints": [[1, 2, [255, 0, 0]]]}
    )
    assert out["size"] == 768
    assert out["strength"] == 1.0 and isinstance(out["strength"], float)
    assert out["hints"] == [(1, 2, (255, 0, 0))]


@pytest.mark.parametrize(
    "values",
    [
        {"nope": 1},
        {"size": 128},  # < min
        {"size": 2048},  # > max
        {"size": 300},  # off step grid
        {"size": "512"},  # wrong type
        {"size": True},  # bool is not an int here
        {"size": 512.5},
        {"strength": 1.5},
        {"strength": float("nan")},
        {"fast": 1},
        {"variant": "huge"},
        {"prompt": 3},
        {"seed": -1},
        {"hints": [[1, 2]]},
        {"hints": [[-1, 2, [0, 0, 0]]]},
        {"hints": [[1, 2, [0, 0, 256]]]},
        {"hints": "x"},
    ],
)
def test_invalid_values_raise(values):
    with pytest.raises(ValueError):
        validate_params(SCHEMA, values)


def test_parse_strings():
    out = parse_params(
        SCHEMA,
        {
            "size": "640",
            "strength": "0.25",
            "fast": "yes",
            "variant": "large",
            "hints": "[[3, 4, [1, 2, 3]]]",
        },
    )
    assert out == {
        "size": 640,
        "strength": 0.25,
        "fast": True,
        "variant": "large",
        "hints": [(3, 4, (1, 2, 3))],
    }


@pytest.mark.parametrize(
    "pair", [{"size": "abc"}, {"fast": "maybe"}, {"hints": "[["}, {"unknown": "1"}]
)
def test_parse_errors(pair):
    with pytest.raises(ValueError):
        parse_params(SCHEMA, pair)


def test_invalid_schema_rejected():
    with pytest.raises(ValueError):
        Param("x", "choice", "a")
    with pytest.raises(ValueError):
        Param("x", "int", 5, min=10)
