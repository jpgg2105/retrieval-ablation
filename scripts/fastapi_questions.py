"""The hand-written question set over FastAPI's docs, as (kind, question, doc, evidence).

kind: "exact"   - the question contains a literal term from the docs (a parameter name, a header,
                  a CLI flag, a status code), the case where keyword search should help.
      "paraphrase" - the question describes the need in other words and avoids the docs' terms.
evidence: a short verbatim passage that answers the question. A retrieved chunk counts as relevant
when it contains the passage, so the labels don't depend on how the docs are chunked.

    python scripts/fastapi_questions.py      # checks every passage and writes data/fastapi/questions.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path

Q = [
    # ---- exact-term questions ------------------------------------------------------------
    ("exact", "What does exclude_unset do in model_dump for PATCH updates?", "tutorial/body-updates.md",
     "That would generate a `dict` with only the data that was set when creating the `item` model, excluding default values."),
    ("exact", "Why use secrets.compare_digest() to check HTTP Basic credentials?", "advanced/security/http-basic-auth.md",
     "it will be secure against a type of attacks called \"timing attacks\""),
    ("exact", "What does TrustedHostMiddleware protect against?", "advanced/middleware.md",
     "Enforces that all incoming requests have a correctly set `Host` header, in order to guard against HTTP Host Header attacks."),
    ("exact", "What is the default minimum_size of GZipMiddleware?", "advanced/middleware.md",
     "Do not GZip responses that are smaller than this minimum size in bytes. Defaults to `500`."),
    ("exact", "Can allow_origins be ['*'] when allow_credentials is True?", "tutorial/cors.md",
     "None of `allow_origins`, `allow_methods` and `allow_headers` can be set to `['*']` if `allow_credentials` is set to `True`."),
    ("exact", "What is the default max_age for CORS responses in CORSMiddleware?", "tutorial/cors.md",
     "Sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`."),
    ("exact", "How do I reset app.dependency_overrides after a test?", "advanced/testing-dependencies.md",
     "Then you can reset your overrides (remove them) by setting `app.dependency_overrides` to be an empty `dict`"),
    ("exact", "When should I add --proxy-headers to fastapi run in Docker?", "deployment/docker.md",
     "add the option `--proxy-headers`, this will tell Uvicorn (through the FastAPI CLI) to trust the headers sent by that proxy"),
    ("exact", "What is root_path in the ASGI specification used for?", "advanced/behind-a-proxy.md",
     "The `root_path` is a mechanism provided by the ASGI specification (that FastAPI is built on, through Starlette)."),
    ("exact", "How do I set convert_underscores to False on Header and what is the risk?", "tutorial/header-params.md",
     "bear in mind that some HTTP proxies and servers disallow the usage of headers with underscores"),
    ("exact", "What object does UploadFile.file expose, a SpooledTemporaryFile?", "tutorial/request-files.md",
     "This is the actual Python file object that you can pass directly to other functions or libraries"),
    ("exact", "How do I mark a query parameter deprecated=True in Query?", "tutorial/query-params-str-validations.md",
     "Then pass the parameter `deprecated=True` to `Query`"),
    ("exact", "How do I set the OpenAPI operationId of a path operation with operation_id?", "advanced/path-operation-advanced-configuration.md",
     "You can set the OpenAPI `operationId` to be used in your *path operation* with the parameter `operation_id`."),
    ("exact", "What happens to extra query parameters when a query model has extra=\"forbid\"?", "tutorial/query-param-models.md",
     "With `extra=\"forbid\"`, any parameter outside that model is rejected."),
    ("exact", "Should I use --workers inside a container on Kubernetes?", "deployment/docker.md",
     "You would want to have just a **single Uvicorn process** per container (but probably multiple containers)."),
    ("exact", "Why do I need python-multipart to receive uploaded files?", "tutorial/request-files.md",
     "This is because uploaded files are sent as \"form data\"."),
    ("exact", "Which pwdlib algorithm is recommended for password hashing?", "tutorial/security/oauth2-jwt.md",
     "The recommended algorithm is \"Argon2\"."),
    ("exact", "How do I resume a server-sent events stream with the Last-Event-ID header?", "tutorial/server-sent-events.md",
     "When a browser reconnects after a connection drop, it sends the last received `id` in the `Last-Event-ID` header."),
    ("exact", "Why does AsyncClient not trigger lifespan events, and what is LifespanManager?", "advanced/async-tests.md",
     "If your application relies on lifespan events, the `AsyncClient` won't trigger these events."),
    ("exact", "Which status code does an invalid boolean like ?short=maybe produce, 422?", "tutorial/query-params.md",
     "An empty value (`?short=`) or an unrecognized value (for example, `?short=maybe`) produces a `422` validation error."),
    ("exact", "How do I generate a SECRET_KEY with openssl rand -hex 32 for JWT?", "tutorial/security/oauth2-jwt.md",
     "To generate a secure random secret key use the command:"),
    ("exact", "Why use @lru_cache on get_settings with a .env file?", "advanced/settings.md",
     "the `Settings` object will be created only once, the first time it's called."),
    ("exact", "How do I disable docs by setting OPENAPI_URL to an empty string?", "how-to/conditional-openapi.md",
     "Then you could disable OpenAPI (including the UI docs) by setting the environment variable `OPENAPI_URL` to the empty string"),
    ("exact", "What value does the WWW-Authenticate header have in HTTP Basic auth?", "advanced/security/http-basic-auth.md",
     "And returns a header `WWW-Authenticate` with a value of `Basic`, and an optional `realm` parameter."),
    ("exact", "What does HTTPSRedirectMiddleware do with http and ws requests?", "advanced/middleware.md",
     "Any incoming request to `http` or `ws` will be redirected to the secure scheme instead."),
    ("exact", "Since version 0.122.0, which status code do security utilities return instead of 403?", "how-to/authentication-error-status-code.md",
     "Starting with FastAPI version `0.122.0`, they use the more appropriate HTTP status code `401 Unauthorized`"),
    ("exact", "When does exit code after yield run with Depends(scope=\"function\")?", "advanced/advanced-dependencies.md",
     "the exit code after `yield` is executed right after the *path operation function* is finished, before the response is sent back to the client."),
    ("exact", "How do I set syntaxHighlight to disable syntax highlighting in Swagger UI?", "how-to/configure-swagger-ui.md",
     "For example, you could disable syntax highlighting in Swagger UI."),
    ("exact", "How do I cache the generated schema with app.openapi_schema?", "how-to/extending-openapi.md",
     "You can use the property `.openapi_schema` as a \"cache\", to store your generated schema."),
    ("exact", "What does fastapi run listen on, 0.0.0.0, and is auto-reload on?", "fastapi-cli.md",
     "By default, **auto-reload** is disabled. It also listens on the IP address `0.0.0.0`"),
    # ---- paraphrased questions -----------------------------------------------------------
    ("paraphrase", "If I replace a stored record with a full update and leave out a field, what value does that field end up with?", "tutorial/body-updates.md",
     "the input model would take the default value of `\"tax\": 10.5`"),
    ("paraphrase", "Why could checking a login by plain string equality let an attacker guess it character by character?", "advanced/security/http-basic-auth.md",
     "the attackers would have guessed the correct username and password, with the \"help\" of our application, just using the time taken to answer"),
    ("paraphrase", "When I register several wrappers around the app, which one sees the incoming call first?", "tutorial/middleware.md",
     "The last middleware added is the *outermost*, and the first is the *innermost*."),
    ("paraphrase", "Is handling many things at once always better than doing several things simultaneously on multiple cores?", "async.md",
     "Concurrency is different than parallelism. And it is better in **specific** scenarios that involve a lot of waiting."),
    ("paraphrase", "My endpoint for the logged-in person's own profile is being captured by the endpoint that takes an id. Why?", "tutorial/path-params.md",
     "you need to make sure that the path for `/users/me` is declared before the one for `/users/{user_id}`"),
    ("paraphrase", "For heavy jobs that run after the reply is sent, when should I use a separate job queue tool instead?", "tutorial/background-tasks.md",
     "you might benefit from using other bigger tools like [Celery](https://docs.celeryq.dev)"),
    ("paraphrase", "I'm not sure whether my function should be a coroutine or a regular function. What's the safe default?", "async.md",
     "If you just don't know, use normal `def`."),
    ("paraphrase", "How can my tests swap a dependency that calls an outside provider for a fake one?", "advanced/testing-dependencies.md",
     "To override a dependency for testing, you put as a key the original dependency (a function), and as the value, your dependency override (another function)."),
    ("paraphrase", "How do I keep a server program alive after the machine reboots, without a person starting it?", "deployment/concepts.md",
     "you will probably want the server program (e.g. Uvicorn) to be started automatically on server startup, and without needing any **human intervention**"),
    ("paraphrase", "What's the first thing to do so a library upgrade doesn't break my app unexpectedly?", "deployment/versions.md",
     "The first thing you should do is to \"pin\" the version of **FastAPI** you are using"),
    ("paraphrase", "How do I let a web page on another domain call my API from the browser?", "tutorial/cors.md",
     "A list of origins that should be permitted to make cross-origin requests."),
    ("paraphrase", "What does the declared output type do to the data my handler returns, beyond documentation?", "tutorial/response-model.md",
     "**convert and filter the output data** to its type declaration"),
    ("paraphrase", "How do I make a URL argument mandatory so the request fails without it?", "tutorial/query-params.md",
     "But when you want to make a query parameter required, you can just not declare any default value"),
    ("paraphrase", "Where should I keep the database connection so every request gets its own one?", "tutorial/sql-databases.md",
     "This is what ensures that we use a single session per request."),
    ("paraphrase", "Does the order in which I list my endpoint groups' descriptions change how the docs page shows them?", "tutorial/metadata.md",
     "The order of each tag metadata dictionary also defines the order shown in the docs UI."),
    ("paraphrase", "What does the development command do when I edit my code, and why not use it in production?", "fastapi-cli.md",
     "By default, **auto-reload** is enabled, automatically reloading the server when you make changes to your code."),
    ("paraphrase", "How can I avoid re-reading my configuration file from disk on every single request?", "advanced/settings.md",
     "Reading a file from disk is normally a costly (slow) operation, so you probably want to do it only once"),
    ("paraphrase", "How do I hide the interactive documentation in production using configuration?", "how-to/conditional-openapi.md",
     "You can easily use the same Pydantic settings to configure your generated OpenAPI and the docs UIs."),
    ("paraphrase", "How do I make the browser show its built-in username and password prompt again after wrong credentials?", "advanced/security/http-basic-auth.md",
     "add the header `WWW-Authenticate` to make the browser show the login prompt again"),
    ("paraphrase", "How do I tell clients that the bytes I stream are a picture?", "advanced/stream-data.md",
     "you can create a `PNGStreamingResponse` that sets the `Content-Type` header to `image/png` using the `media_type` attribute"),
    ("paraphrase", "Why shouldn't I store a user's plain password or send it back to clients?", "tutorial/response-model.md",
     "Never store the plain password of a user or send it in a response like this"),
    ("paraphrase", "Why is an uploaded file kept in memory a problem for large uploads?", "tutorial/request-files.md",
     "Keep in mind that this means that the whole contents will be stored in memory. This will work well for small files."),
    ("paraphrase", "Can my stored JSON hold dates directly, or do I need to turn them into text first?", "tutorial/body-updates.md",
     "You can use the `jsonable_encoder` to convert the input data to data that can be stored as JSON (e.g. with a NoSQL database)."),
    ("paraphrase", "Which permissions does a third-party app get if the user grants only some of the requested ones?", "advanced/security/oauth2-scopes.md",
     "you will be able to access `/users/me/` but not `/users/me/items/`"),
    ("paraphrase", "When I raise an error from a helper function deep inside my handler, does the rest of the handler still run?", "tutorial/handling-errors.md",
     "it won't run the rest of the code in the *path operation function*, it will terminate that request right away"),
    ("paraphrase", "Why might a cleanup step in a dependency break a streamed reply in older versions?", "advanced/advanced-dependencies.md",
     "the `StreamingResponse` would not be able to use that session"),
    ("paraphrase", "If a cloud service already replicates my app, how many server processes should each instance run?", "deployment/concepts.md",
     "it would most probably be **a single Uvicorn process**, and the cloud service would be in charge of replicating it"),
    ("paraphrase", "How do I add a company logo to the alternative documentation page by changing the generated schema?", "how-to/extending-openapi.md",
     "Now you can add the ReDoc extension, adding a custom `x-logo` to the `info` \"object\" in the OpenAPI schema"),
    ("paraphrase", "Why would a client get a 'not found' error with a small JSON message when an item is missing?", "tutorial/handling-errors.md",
     "\"detail\": \"Item not found\""),
    ("paraphrase", "What should I pass to a library that expects an ordinary file object when handling an upload?", "tutorial/request-files.md",
     "It exposes an actual Python [`SpooledTemporaryFile`]"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "fastapi"
    docs = {}
    for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        docs[d["id"]] = norm(d["text"])
    out, bad = [], 0
    for i, (kind, q, doc, ev) in enumerate(Q, 1):
        if doc not in docs or norm(ev) not in docs[doc]:
            print(f"MISSING q{i:02d}: {doc}: {ev[:60]}")
            bad += 1
        out.append({"id": f"fa{i:02d}", "kind": kind, "query": q, "doc": doc, "evidence": [ev]})
    assert not bad, f"{bad} passages not found"
    with (root / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {k: sum(r["kind"] == k for r in out) for k in ("exact", "paraphrase")}
    print(f"{len(out)} questions ({kinds}) -> {root / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
