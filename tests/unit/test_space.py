"""Tests for llmtuna.space — Float, Int, Choice."""

import pytest

import llmtuna as lt


# ============================================================
# Float — construction
# ============================================================

@pytest.mark.parametrize("bounds", [(0.5, 0.5), (1.0, 0.5), (-1.0, -2.0)])
def test_float_construction_rejects_bad_bounds(bounds):
    with pytest.raises(ValueError, match="lo < hi"):
        lt.Float(name="lr", description="learning rate", bounds=bounds)


@pytest.mark.parametrize("initial", [0.5, 5.0])
def test_float_construction_rejects_initial_outside_bounds(initial):
    with pytest.raises(ValueError, match="outside bounds"):
        lt.Float(name="lr", description="learning rate", bounds=(1.0, 2.0), initial=initial)


def test_float_construction_accepts_no_bounds_no_initial():
    p = lt.Float(name="lr", description="learning rate")
    assert p.bounds is None and p.initial is None


def test_float_construction_accepts_bounds_only():
    p = lt.Float(name="lr", description="learning rate", bounds=(1e-6, 1.0))
    assert p.bounds == (1e-6, 1.0)


def test_float_construction_accepts_initial_only():
    p = lt.Float(name="lr", description="learning rate", initial=1e-3)
    assert p.initial == 1e-3


# ============================================================
# Float — validate
# ============================================================

def test_float_validate_accepts_float_in_bounds():
    p = lt.Float(name="lr", description="...", bounds=(0.0, 1.0))
    assert p.validate(value=0.5) == 0.5


def test_float_validate_coerces_int_to_float():
    p = lt.Float(name="lr", description="...", bounds=(0, 10))
    result = p.validate(value=5)
    assert result == 5.0
    assert isinstance(result, float)


def test_float_validate_accepts_exact_bounds():
    p = lt.Float(name="lr", description="...", bounds=(1.0, 2.0))
    assert p.validate(value=1.0) == 1.0
    assert p.validate(value=2.0) == 2.0


def test_float_validate_accepts_any_number_with_no_bounds():
    p = lt.Float(name="lr", description="...")
    assert p.validate(value=-1e9) == -1e9
    assert p.validate(value=1e9) == 1e9


@pytest.mark.parametrize("bad_value", [True, False])
def test_float_validate_rejects_bool(bad_value):
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="bool not accepted"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", ["0.5", None, [0.5], {"v": 0.5}])
def test_float_validate_rejects_non_numbers(bad_value):
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="expected number"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", [0.5, 2.5])
def test_float_validate_rejects_out_of_bounds(bad_value):
    p = lt.Float(name="lr", description="...", bounds=(1.0, 2.0))
    with pytest.raises(ValueError, match="outside bounds"):
        p.validate(value=bad_value)


def test_float_validate_rejects_nan():
    p = lt.Float(name="lr", description="...")
    with pytest.raises(ValueError, match="NaN"):
        p.validate(value=float("nan"))


# ============================================================
# Float — to_schema
# ============================================================

def test_float_to_schema_returns_number_type():
    p = lt.Float(name="lr", description="learning rate")
    assert p.to_schema()["type"] == "number"


def test_float_to_schema_includes_description():
    p = lt.Float(name="lr", description="learning rate")
    assert "learning rate" in p.to_schema()["description"]


# ============================================================
# Int — construction
# ============================================================

@pytest.mark.parametrize("bounds", [(1.5, 10), (1, 10.5), (1.0, 10.0)])
def test_int_construction_rejects_float_bounds(bounds):
    with pytest.raises(ValueError, match="must be integers"):
        lt.Int(name="depth", description="...", bounds=bounds)


@pytest.mark.parametrize("bounds", [(True, 10), (1, False)])
def test_int_construction_rejects_bool_bounds(bounds):
    with pytest.raises(ValueError, match="must be integers"):
        lt.Int(name="depth", description="...", bounds=bounds)


@pytest.mark.parametrize("initial", [True, False, 1.5, "5"])
def test_int_construction_rejects_non_int_initial(initial):
    with pytest.raises(ValueError, match="initial must be int"):
        lt.Int(name="depth", description="...", initial=initial)


def test_int_construction_rejects_initial_outside_bounds():
    with pytest.raises(ValueError, match="outside bounds"):
        lt.Int(name="depth", description="...", bounds=(1, 10), initial=20)


# ============================================================
# Int — validate
# ============================================================

def test_int_validate_accepts_int():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    assert p.validate(value=5) == 5


def test_int_validate_coerces_integer_valued_float():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    result = p.validate(value=5.0)
    assert result == 5
    assert isinstance(result, int)


def test_int_validate_rejects_non_integer_float():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    with pytest.raises(ValueError, match="not integer-valued"):
        p.validate(value=5.5)


@pytest.mark.parametrize("bad_value", [True, False])
def test_int_validate_rejects_bool(bad_value):
    p = lt.Int(name="depth", description="...")
    with pytest.raises(ValueError, match="bool not accepted"):
        p.validate(value=bad_value)


@pytest.mark.parametrize("bad_value", ["5", None, [5]])
def test_int_validate_rejects_non_numbers(bad_value):
    p = lt.Int(name="depth", description="...")
    with pytest.raises(ValueError, match="expected integer"):
        p.validate(value=bad_value)


def test_int_validate_bounds_inclusive():
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    assert p.validate(value=1) == 1
    assert p.validate(value=10) == 10


@pytest.mark.parametrize("bad_value", [0, 11])
def test_int_validate_rejects_out_of_bounds(bad_value):
    p = lt.Int(name="depth", description="...", bounds=(1, 10))
    with pytest.raises(ValueError, match="outside bounds"):
        p.validate(value=bad_value)


# ============================================================
# Int — to_schema
# ============================================================

def test_int_to_schema_returns_integer_type():
    p = lt.Int(name="depth", description="...")
    assert p.to_schema()["type"] == "integer"


# ============================================================
# Choice — construction
# ============================================================

def test_choice_construction_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one option"):
        lt.Choice(name="activation", description="...", options=[])


def test_choice_construction_rejects_initial_not_in_options():
    with pytest.raises(ValueError, match="not in options"):
        lt.Choice(
            name="activation",
            description="...",
            options=["relu", "gelu"],
            initial="silu",
        )


def test_choice_construction_accepts_mixed_type_options():
    p = lt.Choice(name="misc", description="...", options=[1, "two", 3.0])
    assert p.options == [1, "two", 3.0]


# ============================================================
# Choice — validate
# ============================================================

def test_choice_validate_accepts_value_in_options():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu", "silu"])
    assert p.validate(value="gelu") == "gelu"


def test_choice_validate_rejects_value_not_in_options():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu"])
    with pytest.raises(ValueError, match="not in options"):
        p.validate(value="tanh")


def test_choice_validate_strict_type_equality():
    """String '5' must not match int 5 — type-strict membership."""
    p = lt.Choice(name="misc", description="...", options=["5", "10"])
    with pytest.raises(ValueError, match="not in options"):
        p.validate(value=5)


# ============================================================
# Choice — to_schema
# ============================================================

def test_choice_to_schema_has_enum_no_type():
    p = lt.Choice(name="activation", description="...", options=["relu", "gelu", "silu"])
    schema = p.to_schema()
    assert schema["enum"] == ["relu", "gelu", "silu"]
    assert "type" not in schema
