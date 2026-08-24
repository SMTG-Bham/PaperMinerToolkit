"""Shared HTTP test doubles for the data-source clients.

Every source client takes an injected HTTP session, so its tests need the same
two doubles: a prepared response and a session that hands them out in order
while recording what was asked for. They were written once per source and
drifted, so they live here instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import requests


class FakeResponse:
    """Prepared provider response with a configurable status code and body.

    Parameters
    ----------
    text : str, default=''
        Raw response body. Decoded as JSON when no ``payload`` is given.
    payload : Any, optional
        Pre-decoded JSON payload, for providers whose tests do not care about
        the raw text. ``json()`` raises when neither this nor a decodable
        ``text`` is available, which is how a malformed body is simulated.
    status_code : int, default=200
        HTTP status code to report.
    headers : Mapping[str, str] or None, optional
        Response headers, such as ``Retry-After``.
    """

    def __init__(self,
                 text: str = '',
                 payload: Any = None,
                 status_code: int = 200,
                 headers: Mapping[str, str] | None = None) -> None:
        """Store the prepared response.

        Parameters
        ----------
        text : str, default=''
            Raw response body.
        payload : Any, optional
            Pre-decoded JSON payload.
        status_code : int, default=200
            HTTP status code to report.
        headers : Mapping[str, str] or None, optional
            Response headers.

        Returns
        -------
        None
            The double is initialized in place.
        """
        self.text = text
        self.payload = payload
        self.status_code = status_code
        self.headers = dict(headers or {})

    def json(self) -> Any:
        """Decode the prepared body as JSON.

        Returns
        -------
        Any
            The prepared payload, or the body decoded as JSON.

        Raises
        ------
        ValueError
            If no payload was prepared and the body is not valid JSON.
        """
        if self.payload is not None:
            return self.payload
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        """Validate the prepared response status.

        Returns
        -------
        None
            Nothing is returned for a successful status.

        Raises
        ------
        requests.HTTPError
            If the prepared status is an error status.
        """
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)


class FakeSession:
    """Return prepared responses in order and record request arguments.

    Parameters
    ----------
    responses : Iterable[FakeResponse]
        Responses to hand out, one per ``get`` call.
    """

    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        """Store the prepared responses.

        Parameters
        ----------
        responses : Iterable[FakeResponse]
            Responses to hand out, one per ``get`` call.

        Returns
        -------
        None
            The double is initialized in place.
        """
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> FakeResponse:
        """Return the next prepared response and record the request.

        Parameters
        ----------
        url : str
            Endpoint the client asked for.
        params : Mapping[str, Any]
            Query parameters the client sent.
        headers : Mapping[str, str]
            Headers the client sent.
        timeout : float
            Timeout the client asked for.

        Returns
        -------
        FakeResponse
            The next prepared response.
        """
        self.calls.append({
            'url': url,
            'params': dict(params),
            'headers': dict(headers),
            'timeout': timeout,
        })
        return next(self.responses)
