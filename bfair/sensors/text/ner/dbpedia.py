import requests
from urllib.parse import quote
from pathlib import Path

from typing import List, Dict, Any, Tuple
from bfair.sensors.base import Sensor, P_GENDER


class DBPediaSensor(Sensor):
    pass


class DBPediaWrapper:
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    DEFAULT_GRAPH_URI = "http://dbpedia.org"
    FORMAT = "json"
    DEFAULT_PARAMS = {"default-graph-uri": DEFAULT_GRAPH_URI, "format": FORMAT}
    LIMIT = 10000

    def __init__(self):
        pass

    def _get_params(self, **others):
        return dict(self.DEFAULT_PARAMS, **others)

    def _build_payload(self, params: Dict[str, Any]):
        return "&".join(f"{k}={quote(v, safe='+{}')}" for k, v in params.items())

    def _do_query(self, query, default=None):
        params = self._get_params(query=query)
        payload = self._build_payload(params)
        response = requests.get(self.SPARQL_ENDPOINT, params=payload)
        if response.ok:
            return response.json()["results"]
        if default is None:
            raise RuntimeError(response.content)

    def _get_values(self, results, *keys) -> List[Tuple]:
        bindings = results["bindings"]
        values = [tuple(row[key]["value"] for key in keys) for row in bindings]
        return values

    def _do_large_query_and_merge_values(self, query, key, *keys):
        limit = self.LIMIT
        values = []
        offset = 0
        while True:
            partial = self._do_query(
                query=query
                + f"""+
                ORDER+BY+ASC(?{key})+
                LIMIT+{limit}+
                OFFSET+{offset * limit}
                """,
                default=(),
            )
            if not partial:
                break
            values.extend(self._get_values(partial, key, *keys))
            offset += 1
        return values

    def _merge_at_index(self, values, index=0):
        return [
            value
            for row in values
            for value in (self._default_transformation(row[index]),)
            if value
        ]

    def _default_transformation(self, value):
        return Path(value.title()).name

    def get_property_of(self, entity, property):
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{property}+
            WHERE+{{
                dbr:{entity}+dbp:{property}+?{property}+.
            }}
            """,
            property,
        )
        values = self._merge_at_index(values)
        return set(values)

    def get_people_with_property(self, property):
        var_name = "who"
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{var_name},+?{property}+
            WHERE+{{
                ?{var_name}+a+dbo:Person+.+
                ?{var_name}+dbp:{property}+?{property}+.
            }}
            """,
            var_name,
        )

        values = self._merge_at_index(values)
        return set(values)

    def get_all_of_type(self, type_name):
        var_name = "who"
        values = self._do_large_query_and_merge_values(
            f"""
            SELECT+?{var_name}+
            WHERE+{{
                ?{var_name}+a+dbo:{type_name}+.
            }}""",
            var_name,
        )
        values = self._merge_at_index(values)
        return set(values)