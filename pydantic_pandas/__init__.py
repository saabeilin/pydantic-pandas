import typing

import pandas as pd
import pydantic


def dataframe(pydantic_model: pydantic.BaseModel) -> pd.DataFrame:
    cols_types = {
        name: fi.annotation for name, fi in pydantic_model.model_fields.items()
    }

    def dtype(type_: typing.Any) -> typing.Any:
        return type_ if type_ in (int, float, str, bool) else "object"

    return pd.DataFrame({c: pd.Series(dtype=dtype(t)) for c, t in cols_types.items()})


def df_validate(df: pd.DataFrame, model: pydantic.BaseModel):
    def _validate(row):
        try:
            model.model_validate(row.to_dict())
        except pydantic.ValidationError:
            return row

    return df.apply(lambda row: _validate(row), axis=1)
    #
    # for row in df.to_dict("records"):
    #     try:
    #         model.model_validate(
    #             obj=row,
    #         )
    #     except pydantic.ValidationError:
    #         raise

