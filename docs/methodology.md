# How would I approach this problem

This documentation describes my thought process during the coding of this case.

## Defining requirements

Before going ahead and coding the entire thing, or perhaps trying to one shot it with one prompt in claude, some careful planning is required to avoid later frustrations or even refactors. By reading the problem statement one can infer some functional requirements:

* A functional lib that abstracts some data science work (data exploration, training and inference). This lib should be published to pypi to later be consumed by external services, such as an API.
* An inference API deployed on K8s that consumes the abstraction described above and uses it to serve a model.
* Both components should have a robust CICD pipeline. They ship differently, though: the lib is published as a package to test.pypi, while the API is containerized and served on K8s.
* Documentation explaining how would I architect everything on AWS.

From the description alone it is necessary to decouple the lib from the API since they serve inherently different purposes. The API is concerned with only loading and serving the model and the lib needs to be a robust solution for the Data Scientist using it. For this reason, I opt to split the services in two different uv projects, each one with their unique configuration.

## Repository layout

Since the two deliverables are decoupled but graded as a single submission, the repository is a **monorepo with independent uv projects** behind a single remote, as the delivery requires one repository.

For CI I'm going with **GitHub Actions**. In a production environment I'd avoid creating a CI dependency on GHA, since GitHub is not really a great example of availability, but I'm following with the technology because it feels familiar to me and time is limited.

```
desafio-inter-monorepo/
├── README.md                 # front door: what this is, how to run it, index of everything
├── .github/
│   └── workflows/            # per-component pipelines, each with its own path filter
│
├── dolores-lib/              # lib used to abstract common data science work
├── data/                     # data used to test lib/api, it should not be packaged with the services 
├── inference-server/         # fastapi APP
├── client_code/              # consumer proof: notebooks exercising the published lib
│
└── docs/                     # cross-cutting documentation
```

Both server and lib get their own pipeline, gated by a path filter so the pipeline is triggered only when the relevant directory is changed. `lib-ci.yml` and `server-ci.yml` handle quality on pushes and PRs; `lib-release.yml` triggers only on `v*` tags. Only the lib is versioned and its dynamic, derived from hatch-vcs, so nothing is hardcoded and the tag is the single source of truth.

All lockfiles are commited, so CI fails if any dependency edit skips a relock.


## Why the lib and the API are structured differently

The library uses the standard `src/` layout and is packaged using Hatch's build system (hatchling) and is versioned with git tags. Releases are made manually with `git tag v0.1.0 && git push origin v0.1.0`, which is what triggers `lib-release.yml`. That means releasing is manual and deliberate, even for pre-releases or dev releases. Adding automatic version bumping is one more moving part in the CD pipeline so for this case I'll keep it simple. 

The API is flat `app/` with `[tool.uv] package = false` as per recommended in FastAPI docs. It is also the standard used in many other FastAPI apps. It is also possible to distribute the wheel of the API and use the wheel in the Dockerfile, but I decided to follow the industry convention. The API's version is inert and deliberately never tagged. I'll use the commit SHA to identify the image. Tags become worthwhile when you want human-meaningful release notes for the API. In the case of a simple serving prototype, I'll leave as it is.


## Linting, formatting and typing

Case asks for black but `ruff format` already replaces black with near-identical output. Adding black means adding one more dependency and I judge it to not be worthy since ruff already does the job.


## What is the purporse of the client_code dir?

This is where I want to make sure the lib imports and abstractions are worked as intended. It's basically a consumer proof. 

The `client_code` project is deliberately **not** a path dependency on `dolores-lib`. It installs the published artifact from test.pypi, which turns it into an end-to-end assertion that Stage 3 actually worked, if the wheel is
broken, the notebook fails.


## Running MLflow locally

MLflow is not a case requirement but I find it to be a good addition to any MLOps workflow since it allow us to treat models as software, meaning the models and experiments are versioned and audited. The client code includes experiment tracking and logging so a submodule in the lib is dedicated to abstract this behavior.

MLflow is mandatory for the data scientist but optional for the API, so it ships as a `dolores-lib[tracking]` extra and stays out of the container image. The API never talks to MLflow: it loads a serialized artifact, which is what keeps the K8s stage small.

Why not Docker or K8s? uv plus the committed lockfile already makes it reproducible. Commands provided in the makefile also help.


