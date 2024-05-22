from typing import Annotated

import pandas as pd
import pydantic

from pydantic_pandas import DataFrameType


class MyDFType(DataFrameType):
    expected_schema = {
        "int_column": int,
        "float_column": float,
        "str_column": str,
    }


MyDF = Annotated[pd.DataFrame, MyDFType]


class MyModel(pydantic.BaseModel):
    df: MyDF


# Create a DataFrame with the expected schema
data = {
    "int_column": [1, 2, 3],
    "float_column": [1.1, 2.2, 3.3],
    "str_column": ["a", "b", "c"],
}
df = pd.DataFrame(data)
model = MyModel(df=df)


def test_serialize_json():
    # Create a DataFrame with the expected schema
    data = {
        "int_column": [1, 2, 3],
        "float_column": [1.1, 2.2, 3.3],
        "str_column": ["a", "b", "c"],
    }
    df = pd.DataFrame(data)

    model = MyModel(df=df)
    serialized = model.model_dump(mode="json")

    assert serialized == {
        "df": {
            "schema": {
                "int_column": "int",
                "float_column": "float",
                "str_column": "str",
            },
            "columns": {
                "int_column": [1, 2, 3],
                "float_column": [1.1, 2.2, 3.3],
                "str_column": ["a", "b", "c"],
            },
        }
    }


def test_serialize_python():
    assert model.df.equals(df)
    serialized = model.model_dump(mode="python")
    assert isinstance(serialized["df"], pd.DataFrame)
    assert serialized["df"].equals(df)


def test_schema():
    schema = model.model_json_schema()
    assert schema == {
        "type": "object",
        "required": [
            "df",
        ],
        "title": "MyModel",
        "properties": {
            "df": {
                "type": "object",
                "description": "DataFrame with columns: int_column, float_column, str_column",
                # "format": "dataframe-json",
                "title": "Df",
                "properties": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "int_column": {"type": "integer"},
                            "float_column": {"type": "number"},
                            "str_column": {"type": "string"},
                        },
                    },
                    "columns": {
                        "type": "object",
                        "properties": {
                            "int_column": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "float_column": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "str_column": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            }
        },
    }
